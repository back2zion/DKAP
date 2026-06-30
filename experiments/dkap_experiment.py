#!/usr/bin/env python3
"""
DKAP Finance Text-to-SQL Experiment
====================================
Preliminary experiment for IEEE Access paper:
"DKAP: Domain Knowledge Adaptation Pattern for Cross-Domain Industrial AI System Design"

Three conditions:
  B1: Vanilla LLM zero-shot (no schema, no domain knowledge)
  B2: Single-stage RAG with flat chunk retrieval
  DKAP: Full L2 structured domain artifacts (schema + glossary + transform rules)

Model: Qwen3-32B-AWQ via vLLM (localhost:8000)
Domain: Finance (Korean card/credit data, derived from ai-governance-bridge PoC)

Author: Dooil Kwak (automated experiment runner)
Date: 2026-03-17
"""

import json
import time
import sqlite3
import hashlib
import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip install requests --break-system-packages -q")
    import requests

# ============================================================
# Configuration
# ============================================================
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "default-model"  # Qwen3-32B-AWQ served by vLLM
MAX_TOKENS = 2048
TEMPERATURE = 0.0  # deterministic for reproducibility
NUM_RUNS = 3  # repeat each condition N times for variance
RESULTS_DIR = Path("/home/aigen/dkap_results")
DB_PATH = Path("/tmp/finance_benchmark.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dkap_experiment.log")
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Synthetic Finance Database Schema & Data
# ============================================================
SCHEMA_DDL = """
-- Korean Financial Card Data Warehouse (ODS Layer)
-- Derived from ai-governance-bridge PoC real schema

CREATE TABLE IF NOT EXISTS ods_card_member (
    기준년월        TEXT NOT NULL,          -- 데이터 기준 연월 (YYYYMM)
    발급회원번호    TEXT NOT NULL,          -- 카드 회원 고유 식별 번호
    고객명          TEXT,                   -- 고객 성명
    VIP등급코드     TEXT,                   -- VIP 등급 (01:VVIP, 02:VIP, 03:Gold, 04:일반)
    입회일자        TEXT,                   -- 카드 입회 일자 (YYYYMMDD)
    최종이용일자    TEXT,                   -- 최종 카드 이용 일자
    신용등급        INTEGER,               -- 개인 신용 등급 (1~10, 1이 최우수)
    남녀구분코드    TEXT,                   -- 성별 (M/F)
    연령대코드      TEXT,                   -- 연령대 (20/30/40/50/60)
    PRIMARY KEY (기준년월, 발급회원번호)
);

CREATE TABLE IF NOT EXISTS ods_card_transaction (
    거래일련번호    INTEGER PRIMARY KEY AUTOINCREMENT,
    기준년월        TEXT NOT NULL,
    발급회원번호    TEXT NOT NULL,
    승인번호        TEXT,
    가맹점명        TEXT,                   -- 가맹점 상호명
    업종코드        TEXT,                   -- 업종 분류 코드
    이용금액        INTEGER,               -- 카드 이용 금액 (원)
    이용일자        TEXT,                   -- 이용 일자 (YYYYMMDD)
    할부개월수      INTEGER DEFAULT 0,     -- 할부 개월 수 (0=일시불)
    FOREIGN KEY (발급회원번호) REFERENCES ods_card_member(발급회원번호)
);

CREATE TABLE IF NOT EXISTS ods_card_credit (
    기준년월        TEXT NOT NULL,
    발급회원번호    TEXT NOT NULL,
    카드이용한도금액 INTEGER,               -- 카드 이용 한도 금액 (원)
    잔액            INTEGER,               -- 현재 잔액 (원)
    연체일수        INTEGER DEFAULT 0,     -- 연체 일수
    연체잔액        INTEGER DEFAULT 0,     -- 연체 잔액 (원)
    한도소진율      REAL,                   -- 한도 소진율 (0.0~1.0)
    PRIMARY KEY (기준년월, 발급회원번호),
    FOREIGN KEY (발급회원번호) REFERENCES ods_card_member(발급회원번호)
);

CREATE TABLE IF NOT EXISTS ods_card_billing (
    기준년월        TEXT NOT NULL,
    발급회원번호    TEXT NOT NULL,
    청구금액        INTEGER,               -- 청구 금액 (원)
    결제일          TEXT,                   -- 결제 예정일 (DD)
    납부상태코드    TEXT,                   -- 납부 상태 (01:완납, 02:미납, 03:부분납)
    PRIMARY KEY (기준년월, 발급회원번호),
    FOREIGN KEY (발급회원번호) REFERENCES ods_card_member(발급회원번호)
);

CREATE TABLE IF NOT EXISTS ods_fin_product (
    상품코드        TEXT PRIMARY KEY,
    상품명          TEXT,                   -- 금융상품명
    상품유형        TEXT,                   -- 유형 (카드/대출/펀드/보험)
    기준금리        REAL,                   -- 기준 금리 (%)
    가입건수        INTEGER DEFAULT 0      -- 누적 가입 건수
);
"""

# Seed data for realistic queries
SEED_DATA_SQL = """
-- Members
INSERT INTO ods_card_member VALUES ('202501','M001','김철수','01','20200115','20250301',2,'M','40');
INSERT INTO ods_card_member VALUES ('202501','M002','이영희','03','20180620','20250228',4,'F','30');
INSERT INTO ods_card_member VALUES ('202501','M003','박지성','02','20190301','20250305',1,'M','50');
INSERT INTO ods_card_member VALUES ('202501','M004','최수연','04','20220801','20250210',6,'F','20');
INSERT INTO ods_card_member VALUES ('202501','M005','정민호','01','20170510','20250310',3,'M','60');
INSERT INTO ods_card_member VALUES ('202501','M006','한소희','02','20210315','20250308',2,'F','30');
INSERT INTO ods_card_member VALUES ('202501','M007','강동원','03','20190722','20250225',5,'M','40');
INSERT INTO ods_card_member VALUES ('202501','M008','윤세리','04','20230101','20250201',7,'F','20');
INSERT INTO ods_card_member VALUES ('202501','M009','송중기','01','20160815','20250312',1,'M','30');
INSERT INTO ods_card_member VALUES ('202501','M010','김태리','02','20200901','20250307',3,'F','40');
INSERT INTO ods_card_member VALUES ('202502','M001','김철수','01','20200115','20250401',2,'M','40');
INSERT INTO ods_card_member VALUES ('202502','M002','이영희','03','20180620','20250320',4,'F','30');
INSERT INTO ods_card_member VALUES ('202502','M003','박지성','02','20190301','20250415',1,'M','50');
INSERT INTO ods_card_member VALUES ('202502','M004','최수연','04','20220801','20250325',6,'F','20');
INSERT INTO ods_card_member VALUES ('202502','M005','정민호','01','20170510','20250410',3,'M','60');

-- Transactions (202501)
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M001','A001','스타벅스 강남점','카페',5500,'20250115',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M001','A002','삼성전자 직영점','전자제품',1890000,'20250120',12);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M001','A003','이마트 역삼점','대형마트',156000,'20250122',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M002','A004','올리브영 홍대점','화장품',45000,'20250110',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M002','A005','배달의민족','음식배달',32000,'20250118',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M003','A006','현대백화점 본점','백화점',2450000,'20250105',6);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M003','A007','GS칼텍스 주유소','주유',85000,'20250112',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M004','A008','넷플릭스','구독',17000,'20250101',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M004','A009','다이소 신촌점','생활용품',8500,'20250125',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M005','A010','서울아산병원','의료',350000,'20250108',3);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M005','A011','하나투어','여행',4200000,'20250115',6);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M006','A012','쿠팡','온라인쇼핑',89000,'20250120',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M006','A013','CGV 용산점','영화',28000,'20250122',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M007','A014','교보문고 광화문','도서',52000,'20250118',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M007','A015','파리바게뜨','베이커리',15000,'20250125',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M008','A016','유니클로 명동점','의류',129000,'20250110',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M009','A017','대한항공','항공',890000,'20250105',3);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M009','A018','롯데호텔','숙박',320000,'20250106',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M010','A019','애플스토어 가로수길','전자제품',1590000,'20250115',12);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M010','A020','스시조','음식점',78000,'20250128',0);
-- 202502 transactions
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M001','B001','스타벅스 강남점','카페',6000,'20250215',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M001','B002','쿠팡','온라인쇼핑',234000,'20250218',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M002','B003','올리브영 홍대점','화장품',67000,'20250210',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M003','B004','현대백화점 본점','백화점',1280000,'20250220',3);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M005','B005','하나투어','여행',2100000,'20250225',6);

-- Credit info
INSERT INTO ods_card_credit VALUES ('202501','M001',50000000,12500000,0,0,0.25);
INSERT INTO ods_card_credit VALUES ('202501','M002',10000000,3200000,0,0,0.32);
INSERT INTO ods_card_credit VALUES ('202501','M003',80000000,45000000,0,0,0.5625);
INSERT INTO ods_card_credit VALUES ('202501','M004',5000000,1800000,15,250000,0.36);
INSERT INTO ods_card_credit VALUES ('202501','M005',100000000,22000000,0,0,0.22);
INSERT INTO ods_card_credit VALUES ('202501','M006',30000000,8500000,0,0,0.2833);
INSERT INTO ods_card_credit VALUES ('202501','M007',15000000,6700000,5,120000,0.4467);
INSERT INTO ods_card_credit VALUES ('202501','M008',3000000,2100000,0,0,0.70);
INSERT INTO ods_card_credit VALUES ('202501','M009',70000000,15000000,0,0,0.2143);
INSERT INTO ods_card_credit VALUES ('202501','M010',40000000,18000000,0,0,0.45);
INSERT INTO ods_card_credit VALUES ('202502','M001',50000000,11000000,0,0,0.22);
INSERT INTO ods_card_credit VALUES ('202502','M002',10000000,2800000,0,0,0.28);
INSERT INTO ods_card_credit VALUES ('202502','M003',80000000,42000000,0,0,0.525);
INSERT INTO ods_card_credit VALUES ('202502','M004',5000000,2200000,30,450000,0.44);
INSERT INTO ods_card_credit VALUES ('202502','M005',100000000,20000000,0,0,0.20);

-- Billing
INSERT INTO ods_card_billing VALUES ('202501','M001',2051500,'15','01');
INSERT INTO ods_card_billing VALUES ('202501','M002',77000,'25','01');
INSERT INTO ods_card_billing VALUES ('202501','M003',2535000,'10','01');
INSERT INTO ods_card_billing VALUES ('202501','M004',25500,'20','02');
INSERT INTO ods_card_billing VALUES ('202501','M005',4550000,'05','01');
INSERT INTO ods_card_billing VALUES ('202501','M006',117000,'15','01');
INSERT INTO ods_card_billing VALUES ('202501','M007',67000,'25','01');
INSERT INTO ods_card_billing VALUES ('202501','M008',129000,'10','01');
INSERT INTO ods_card_billing VALUES ('202501','M009',1210000,'20','01');
INSERT INTO ods_card_billing VALUES ('202501','M010',1668000,'05','01');

-- Financial products
INSERT INTO ods_fin_product VALUES ('P001','프리미엄 골드카드','카드',0.0,15200);
INSERT INTO ods_fin_product VALUES ('P002','스마트 신용대출','대출',4.5,8900);
INSERT INTO ods_fin_product VALUES ('P003','글로벌 주식펀드','펀드',0.0,3200);
INSERT INTO ods_fin_product VALUES ('P004','무배당 종합보험','보험',0.0,12100);
INSERT INTO ods_fin_product VALUES ('P005','직장인 체크카드','카드',0.0,45000);
"""


# ============================================================
# 2. Benchmark Questions (50 NL→SQL pairs)
# ============================================================
# Difficulty: Easy (E), Medium (M), Hard (H)
BENCHMARK = [
    # --- Easy: single table, simple WHERE/aggregation ---
    {
        "id": "Q01", "difficulty": "E",
        "nl": "전체 회원 수를 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT COUNT(DISTINCT 발급회원번호) FROM ods_card_member WHERE 기준년월='202501'"
    },
    {
        "id": "Q02", "difficulty": "E",
        "nl": "VIP등급이 VVIP(01)인 회원의 이름을 모두 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT 고객명 FROM ods_card_member WHERE VIP등급코드='01' AND 기준년월='202501'"
    },
    {
        "id": "Q03", "difficulty": "E",
        "nl": "2025년 1월 전체 거래 건수를 구하시오.",
        "gold_sql": "SELECT COUNT(*) FROM ods_card_transaction WHERE 기준년월='202501'"
    },
    {
        "id": "Q04", "difficulty": "E",
        "nl": "신용등급이 3 이하(우수)인 회원 수를 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT COUNT(*) FROM ods_card_member WHERE 신용등급<=3 AND 기준년월='202501'"
    },
    {
        "id": "Q05", "difficulty": "E",
        "nl": "2025년 1월 가장 큰 이용금액의 거래를 조회하시오.",
        "gold_sql": "SELECT * FROM ods_card_transaction WHERE 기준년월='202501' ORDER BY 이용금액 DESC LIMIT 1"
    },
    {
        "id": "Q06", "difficulty": "E",
        "nl": "연체일수가 0보다 큰 회원번호를 모두 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT 발급회원번호 FROM ods_card_credit WHERE 연체일수>0 AND 기준년월='202501'"
    },
    {
        "id": "Q07", "difficulty": "E",
        "nl": "카드 유형 상품의 누적 가입건수 합계를 구하시오.",
        "gold_sql": "SELECT SUM(가입건수) FROM ods_fin_product WHERE 상품유형='카드'"
    },
    {
        "id": "Q08", "difficulty": "E",
        "nl": "2025년 1월 할부 거래(할부개월수 > 0)의 건수를 구하시오.",
        "gold_sql": "SELECT COUNT(*) FROM ods_card_transaction WHERE 기준년월='202501' AND 할부개월수>0"
    },
    {
        "id": "Q09", "difficulty": "E",
        "nl": "기준년월 202501 기준 청구금액이 100만원 이상인 회원번호를 조회하시오.",
        "gold_sql": "SELECT 발급회원번호 FROM ods_card_billing WHERE 기준년월='202501' AND 청구금액>=1000000"
    },
    {
        "id": "Q10", "difficulty": "E",
        "nl": "여성(F) 회원의 수를 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT COUNT(*) FROM ods_card_member WHERE 남녀구분코드='F' AND 기준년월='202501'"
    },

    # --- Medium: JOIN, GROUP BY, subquery ---
    {
        "id": "Q11", "difficulty": "M",
        "nl": "2025년 1월 기준 회원별 총 이용금액을 구하시오. 이용금액 내림차순 정렬.",
        "gold_sql": "SELECT 발급회원번호, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호 ORDER BY 총이용금액 DESC"
    },
    {
        "id": "Q12", "difficulty": "M",
        "nl": "VIP등급별 평균 신용등급을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT VIP등급코드, AVG(신용등급) AS 평균신용등급 FROM ods_card_member WHERE 기준년월='202501' GROUP BY VIP등급코드"
    },
    {
        "id": "Q13", "difficulty": "M",
        "nl": "2025년 1월 업종코드별 거래 건수와 총 이용금액을 구하시오.",
        "gold_sql": "SELECT 업종코드, COUNT(*) AS 거래건수, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 업종코드"
    },
    {
        "id": "Q14", "difficulty": "M",
        "nl": "회원별 이용한도 대비 잔액 비율(한도소진율)이 0.5 이상인 회원의 이름과 한도소진율을 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.고객명, c.한도소진율 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.기준년월='202501' AND c.한도소진율>=0.5"
    },
    {
        "id": "Q15", "difficulty": "M",
        "nl": "납부상태가 미납(02)인 회원의 이름과 청구금액을 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.고객명, b.청구금액 FROM ods_card_member m JOIN ods_card_billing b ON m.발급회원번호=b.발급회원번호 AND m.기준년월=b.기준년월 WHERE b.납부상태코드='02' AND m.기준년월='202501'"
    },
    {
        "id": "Q16", "difficulty": "M",
        "nl": "2025년 1월 이용금액 상위 5건의 거래에 대해 회원이름, 가맹점명, 이용금액을 조회하시오.",
        "gold_sql": "SELECT m.고객명, t.가맹점명, t.이용금액 FROM ods_card_transaction t JOIN ods_card_member m ON t.발급회원번호=m.발급회원번호 AND t.기준년월=m.기준년월 WHERE t.기준년월='202501' ORDER BY t.이용금액 DESC LIMIT 5"
    },
    {
        "id": "Q17", "difficulty": "M",
        "nl": "연령대별 평균 이용금액을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.연령대코드, AVG(t.이용금액) AS 평균이용금액 FROM ods_card_member m JOIN ods_card_transaction t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 WHERE m.기준년월='202501' GROUP BY m.연령대코드"
    },
    {
        "id": "Q18", "difficulty": "M",
        "nl": "한도소진율이 가장 높은 회원의 이름과 한도소진율을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.고객명, c.한도소진율 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.기준년월='202501' ORDER BY c.한도소진율 DESC LIMIT 1"
    },
    {
        "id": "Q19", "difficulty": "M",
        "nl": "성별(남녀구분코드)별 총 거래금액과 평균 거래금액을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.남녀구분코드, SUM(t.이용금액) AS 총거래금액, AVG(t.이용금액) AS 평균거래금액 FROM ods_card_member m JOIN ods_card_transaction t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 WHERE m.기준년월='202501' GROUP BY m.남녀구분코드"
    },
    {
        "id": "Q20", "difficulty": "M",
        "nl": "2025년 1월과 2월 모두 거래가 있는 회원번호를 조회하시오.",
        "gold_sql": "SELECT 발급회원번호 FROM ods_card_transaction WHERE 기준년월='202501' INTERSECT SELECT 발급회원번호 FROM ods_card_transaction WHERE 기준년월='202502'"
    },

    # --- Hard: multi-join, window, subquery, CASE ---
    {
        "id": "Q21", "difficulty": "H",
        "nl": "2025년 1월 기준 회원별 총 이용금액과 해당 회원의 VIP등급, 신용등급을 함께 조회하되, 총 이용금액이 100만원 이상인 회원만 출력하시오.",
        "gold_sql": "SELECT m.발급회원번호, m.고객명, m.VIP등급코드, m.신용등급, SUM(t.이용금액) AS 총이용금액 FROM ods_card_member m JOIN ods_card_transaction t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 WHERE m.기준년월='202501' GROUP BY m.발급회원번호, m.고객명, m.VIP등급코드, m.신용등급 HAVING SUM(t.이용금액)>=1000000"
    },
    {
        "id": "Q22", "difficulty": "H",
        "nl": "각 회원의 2025년 1월 총 이용금액 대비 전체 평균 이용금액과의 차이를 구하시오.",
        "gold_sql": "SELECT 발급회원번호, SUM(이용금액) AS 총이용금액, SUM(이용금액) - (SELECT AVG(total) FROM (SELECT SUM(이용금액) AS total FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호)) AS 평균대비차이 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호"
    },
    {
        "id": "Q23", "difficulty": "H",
        "nl": "2025년 1월 기준 연체가 있는 회원(연체일수>0)의 이름, 연체일수, 연체잔액, 그리고 해당 회원의 총 이용금액을 조회하시오.",
        "gold_sql": "SELECT m.고객명, c.연체일수, c.연체잔액, COALESCE(t.총이용금액,0) AS 총이용금액 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 LEFT JOIN (SELECT 발급회원번호, 기준년월, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction GROUP BY 발급회원번호, 기준년월) t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 WHERE c.연체일수>0 AND m.기준년월='202501'"
    },
    {
        "id": "Q24", "difficulty": "H",
        "nl": "회원을 이용금액 기준으로 상위 30% (High), 중위 40% (Mid), 하위 30% (Low)로 분류하시오. (기준년월 202501)",
        "gold_sql": "SELECT 발급회원번호, 총이용금액, CASE WHEN rn <= cnt*0.3 THEN 'High' WHEN rn <= cnt*0.7 THEN 'Mid' ELSE 'Low' END AS 등급 FROM (SELECT 발급회원번호, 총이용금액, ROW_NUMBER() OVER (ORDER BY 총이용금액 DESC) AS rn, COUNT(*) OVER () AS cnt FROM (SELECT 발급회원번호, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호))"
    },
    {
        "id": "Q25", "difficulty": "H",
        "nl": "VIP등급(01)이면서 한도소진율이 0.3 미만인 '우량고객'의 이름, 총이용금액, 한도소진율을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.고객명, COALESCE(SUM(t.이용금액),0) AS 총이용금액, c.한도소진율 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 LEFT JOIN ods_card_transaction t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 WHERE m.VIP등급코드='01' AND c.한도소진율<0.3 AND m.기준년월='202501' GROUP BY m.고객명, c.한도소진율"
    },
    {
        "id": "Q26", "difficulty": "H",
        "nl": "2025년 1월 대비 2월의 회원별 이용금액 증감액을 구하시오. 두 달 모두 거래가 있는 회원만 대상.",
        "gold_sql": "SELECT a.발급회원번호, a.금액_01 AS 이용금액_1월, b.금액_02 AS 이용금액_2월, b.금액_02-a.금액_01 AS 증감액 FROM (SELECT 발급회원번호, SUM(이용금액) AS 금액_01 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호) a JOIN (SELECT 발급회원번호, SUM(이용금액) AS 금액_02 FROM ods_card_transaction WHERE 기준년월='202502' GROUP BY 발급회원번호) b ON a.발급회원번호=b.발급회원번호"
    },
    {
        "id": "Q27", "difficulty": "H",
        "nl": "업종코드별 이용금액의 전체 대비 비중(%)을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT 업종코드, SUM(이용금액) AS 업종이용금액, ROUND(SUM(이용금액)*100.0/(SELECT SUM(이용금액) FROM ods_card_transaction WHERE 기준년월='202501'),2) AS 비중_pct FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 업종코드 ORDER BY 비중_pct DESC"
    },
    {
        "id": "Q28", "difficulty": "H",
        "nl": "신용등급 1~3등급 회원과 4~10등급 회원 그룹 간의 평균 한도소진율 차이를 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT (SELECT AVG(c.한도소진율) FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.신용등급<=3 AND m.기준년월='202501') AS 우수등급_평균한도소진율, (SELECT AVG(c.한도소진율) FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.신용등급>=4 AND m.기준년월='202501') AS 일반등급_평균한도소진율"
    },
    {
        "id": "Q29", "difficulty": "H",
        "nl": "각 회원의 이용금액 순위(RANK)를 매기고, 순위별 누적 이용금액을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT 발급회원번호, 총이용금액, RANK() OVER (ORDER BY 총이용금액 DESC) AS 순위, SUM(총이용금액) OVER (ORDER BY 총이용금액 DESC ROWS UNBOUNDED PRECEDING) AS 누적이용금액 FROM (SELECT 발급회원번호, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호)"
    },
    {
        "id": "Q30", "difficulty": "H",
        "nl": "이용금액 기준 상위 3명 회원의 이름, VIP등급, 신용등급, 총이용금액, 한도소진율을 모두 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.고객명, m.VIP등급코드, m.신용등급, t.총이용금액, c.한도소진율 FROM ods_card_member m JOIN (SELECT 발급회원번호, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호 ORDER BY 총이용금액 DESC LIMIT 3) t ON m.발급회원번호=t.발급회원번호 JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.기준년월='202501'"
    },
    # --- Additional Medium/Hard for 50 total ---
    {
        "id": "Q31", "difficulty": "M",
        "nl": "2025년 1월 거래가 없는 회원의 이름을 조회하시오.",
        "gold_sql": "SELECT 고객명 FROM ods_card_member WHERE 기준년월='202501' AND 발급회원번호 NOT IN (SELECT DISTINCT 발급회원번호 FROM ods_card_transaction WHERE 기준년월='202501')"
    },
    {
        "id": "Q32", "difficulty": "M",
        "nl": "가맹점별 평균 이용금액을 구하고 100만원 이상인 가맹점만 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT 가맹점명, AVG(이용금액) AS 평균이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 가맹점명 HAVING AVG(이용금액)>=1000000"
    },
    {
        "id": "Q33", "difficulty": "M",
        "nl": "2025년 1월 전체 청구금액 합계를 구하시오.",
        "gold_sql": "SELECT SUM(청구금액) FROM ods_card_billing WHERE 기준년월='202501'"
    },
    {
        "id": "Q34", "difficulty": "E",
        "nl": "기준금리가 가장 높은 금융상품의 이름과 금리를 조회하시오.",
        "gold_sql": "SELECT 상품명, 기준금리 FROM ods_fin_product ORDER BY 기준금리 DESC LIMIT 1"
    },
    {
        "id": "Q35", "difficulty": "M",
        "nl": "연령대별 회원 수를 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT 연령대코드, COUNT(*) AS 회원수 FROM ods_card_member WHERE 기준년월='202501' GROUP BY 연령대코드"
    },
    {
        "id": "Q36", "difficulty": "H",
        "nl": "각 VIP등급별로 평균 이용금액과 평균 한도소진율을 함께 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.VIP등급코드, AVG(t.이용금액) AS 평균이용금액, AVG(c.한도소진율) AS 평균한도소진율 FROM ods_card_member m JOIN ods_card_transaction t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.기준년월='202501' GROUP BY m.VIP등급코드"
    },
    {
        "id": "Q37", "difficulty": "H",
        "nl": "완납(01) 회원과 미납(02) 회원의 평균 신용등급을 비교하시오. (기준년월 202501)",
        "gold_sql": "SELECT b.납부상태코드, AVG(m.신용등급) AS 평균신용등급 FROM ods_card_billing b JOIN ods_card_member m ON b.발급회원번호=m.발급회원번호 AND b.기준년월=m.기준년월 WHERE b.기준년월='202501' GROUP BY b.납부상태코드"
    },
    {
        "id": "Q38", "difficulty": "E",
        "nl": "금융상품 중 상품유형별 상품 수를 구하시오.",
        "gold_sql": "SELECT 상품유형, COUNT(*) AS 상품수 FROM ods_fin_product GROUP BY 상품유형"
    },
    {
        "id": "Q39", "difficulty": "M",
        "nl": "2025년 1월 일시불(할부개월수=0) 거래의 평균 이용금액을 구하시오.",
        "gold_sql": "SELECT AVG(이용금액) FROM ods_card_transaction WHERE 기준년월='202501' AND 할부개월수=0"
    },
    {
        "id": "Q40", "difficulty": "H",
        "nl": "2025년 1월 기준 회원별로 거래 건수, 총 이용금액, 평균 이용금액, 최대 이용금액을 모두 구하시오.",
        "gold_sql": "SELECT 발급회원번호, COUNT(*) AS 거래건수, SUM(이용금액) AS 총이용금액, AVG(이용금액) AS 평균이용금액, MAX(이용금액) AS 최대이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호"
    },
    {
        "id": "Q41", "difficulty": "M",
        "nl": "2025년 1월 이용금액이 50만원 이상인 거래의 회원이름과 가맹점명을 조회하시오.",
        "gold_sql": "SELECT m.고객명, t.가맹점명, t.이용금액 FROM ods_card_transaction t JOIN ods_card_member m ON t.발급회원번호=m.발급회원번호 AND t.기준년월=m.기준년월 WHERE t.기준년월='202501' AND t.이용금액>=500000"
    },
    {
        "id": "Q42", "difficulty": "H",
        "nl": "연체 이력이 있는 회원(연체일수>0)의 비율을 전체 회원 대비 퍼센트로 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT ROUND(COUNT(CASE WHEN 연체일수>0 THEN 1 END)*100.0/COUNT(*),2) AS 연체회원비율_pct FROM ods_card_credit WHERE 기준년월='202501'"
    },
    {
        "id": "Q43", "difficulty": "M",
        "nl": "VIP등급코드별 회원 수와 전체 대비 비율을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT VIP등급코드, COUNT(*) AS 회원수, ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM ods_card_member WHERE 기준년월='202501'),2) AS 비율_pct FROM ods_card_member WHERE 기준년월='202501' GROUP BY VIP등급코드"
    },
    {
        "id": "Q44", "difficulty": "H",
        "nl": "2025년 1월과 2월 사이 한도소진율이 증가한 회원의 이름과 증감폭을 구하시오.",
        "gold_sql": "SELECT m.고객명, c2.한도소진율-c1.한도소진율 AS 소진율증감 FROM ods_card_credit c1 JOIN ods_card_credit c2 ON c1.발급회원번호=c2.발급회원번호 JOIN ods_card_member m ON c1.발급회원번호=m.발급회원번호 AND m.기준년월='202501' WHERE c1.기준년월='202501' AND c2.기준년월='202502' AND c2.한도소진율>c1.한도소진율"
    },
    {
        "id": "Q45", "difficulty": "E",
        "nl": "2025년 1월 거래에서 사용된 업종코드의 종류 수를 구하시오.",
        "gold_sql": "SELECT COUNT(DISTINCT 업종코드) FROM ods_card_transaction WHERE 기준년월='202501'"
    },
    {
        "id": "Q46", "difficulty": "M",
        "nl": "입회일자가 2020년 이전인 회원의 이름과 입회일자를 조회하시오. (기준년월 202501)",
        "gold_sql": "SELECT 고객명, 입회일자 FROM ods_card_member WHERE 기준년월='202501' AND 입회일자<'20200101'"
    },
    {
        "id": "Q47", "difficulty": "H",
        "nl": "각 회원의 청구금액 대비 실제 이용금액의 비율을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.고객명, b.청구금액, COALESCE(t.총이용금액,0) AS 총이용금액, CASE WHEN b.청구금액>0 THEN ROUND(COALESCE(t.총이용금액,0)*1.0/b.청구금액,2) ELSE 0 END AS 이용대비청구비율 FROM ods_card_member m JOIN ods_card_billing b ON m.발급회원번호=b.발급회원번호 AND m.기준년월=b.기준년월 LEFT JOIN (SELECT 발급회원번호, 기준년월, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction GROUP BY 발급회원번호, 기준년월) t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 WHERE m.기준년월='202501'"
    },
    {
        "id": "Q48", "difficulty": "M",
        "nl": "2025년 2월에만 거래가 있고 1월에는 거래가 없는 회원번호를 조회하시오.",
        "gold_sql": "SELECT DISTINCT 발급회원번호 FROM ods_card_transaction WHERE 기준년월='202502' AND 발급회원번호 NOT IN (SELECT DISTINCT 발급회원번호 FROM ods_card_transaction WHERE 기준년월='202501')"
    },
    {
        "id": "Q49", "difficulty": "H",
        "nl": "신용등급과 한도소진율 간의 상관관계를 파악하기 위해, 신용등급별 평균 한도소진율을 구하시오. (기준년월 202501)",
        "gold_sql": "SELECT m.신용등급, AVG(c.한도소진율) AS 평균한도소진율, COUNT(*) AS 회원수 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.기준년월='202501' GROUP BY m.신용등급 ORDER BY m.신용등급"
    },
    {
        "id": "Q50", "difficulty": "H",
        "nl": "전체 회원의 종합 리스크 점수를 계산하시오. 리스크 점수 = (연체일수 × 0.4) + (한도소진율 × 100 × 0.3) + (신용등급 × 0.3). 리스크 점수 내림차순 정렬. (기준년월 202501)",
        "gold_sql": "SELECT m.발급회원번호, m.고객명, m.신용등급, c.연체일수, c.한도소진율, ROUND(c.연체일수*0.4 + c.한도소진율*100*0.3 + m.신용등급*0.3, 2) AS 리스크점수 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.기준년월='202501' ORDER BY 리스크점수 DESC"
    },
]


# ============================================================
# 3. Prompt Templates (B1, B2, DKAP)
# ============================================================

# --- B1: Vanilla zero-shot (no schema, no domain knowledge) ---
PROMPT_B1 = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.
Output ONLY the SQL query, nothing else. Do not include explanation.

Question: {question}

SQL:"""


# --- B2: Single-stage RAG with flat chunk retrieval ---
# Simulates a naive RAG that retrieves relevant "chunks" of documentation
PROMPT_B2 = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.
You have access to the following database documentation chunks retrieved via similarity search:

---
{retrieved_chunks}
---

Output ONLY the SQL query, nothing else. Do not include explanation.

Question: {question}

SQL:"""


def get_b2_chunks(question: str) -> str:
    """Simulate flat-chunk retrieval for B2 condition.
    Returns concatenated raw text chunks without structural hierarchy."""
    # All table/column info as flat unstructured text
    chunks = []

    # Simple keyword matching to simulate retrieval
    keywords_table_map = {
        "회원": "ods_card_member 테이블: 기준년월(TEXT), 발급회원번호(TEXT PK), 고객명(TEXT), VIP등급코드(TEXT), 입회일자(TEXT), 최종이용일자(TEXT), 신용등급(INTEGER), 남녀구분코드(TEXT), 연령대코드(TEXT)",
        "거래": "ods_card_transaction 테이블: 거래일련번호(INTEGER PK), 기준년월(TEXT), 발급회원번호(TEXT FK), 승인번호(TEXT), 가맹점명(TEXT), 업종코드(TEXT), 이용금액(INTEGER), 이용일자(TEXT), 할부개월수(INTEGER)",
        "이용금액": "ods_card_transaction 테이블: 거래일련번호(INTEGER PK), 기준년월(TEXT), 발급회원번호(TEXT FK), 승인번호(TEXT), 가맹점명(TEXT), 업종코드(TEXT), 이용금액(INTEGER), 이용일자(TEXT), 할부개월수(INTEGER)",
        "신용": "ods_card_credit 테이블: 기준년월(TEXT), 발급회원번호(TEXT), 카드이용한도금액(INTEGER), 잔액(INTEGER), 연체일수(INTEGER), 연체잔액(INTEGER), 한도소진율(REAL)",
        "한도": "ods_card_credit 테이블: 기준년월(TEXT), 발급회원번호(TEXT), 카드이용한도금액(INTEGER), 잔액(INTEGER), 연체일수(INTEGER), 연체잔액(INTEGER), 한도소진율(REAL)",
        "연체": "ods_card_credit 테이블: 기준년월(TEXT), 발급회원번호(TEXT), 카드이용한도금액(INTEGER), 잔액(INTEGER), 연체일수(INTEGER), 연체잔액(INTEGER), 한도소진율(REAL)",
        "청구": "ods_card_billing 테이블: 기준년월(TEXT), 발급회원번호(TEXT), 청구금액(INTEGER), 결제일(TEXT), 납부상태코드(TEXT)",
        "납부": "ods_card_billing 테이블: 기준년월(TEXT), 발급회원번호(TEXT), 청구금액(INTEGER), 결제일(TEXT), 납부상태코드(TEXT)",
        "상품": "ods_fin_product 테이블: 상품코드(TEXT PK), 상품명(TEXT), 상품유형(TEXT), 기준금리(REAL), 가입건수(INTEGER)",
        "금리": "ods_fin_product 테이블: 상품코드(TEXT PK), 상품명(TEXT), 상품유형(TEXT), 기준금리(REAL), 가입건수(INTEGER)",
        "VIP": "VIP등급코드: 01=VVIP, 02=VIP, 03=Gold, 04=일반",
        "업종": "업종코드 종류: 카페, 전자제품, 대형마트, 화장품, 음식배달, 백화점, 주유, 구독, 생활용품 등",
    }

    matched = set()
    for kw, chunk in keywords_table_map.items():
        if kw in question:
            matched.add(chunk)

    # Always include at least the member and transaction table info
    if not matched:
        matched.add(keywords_table_map["회원"])
        matched.add(keywords_table_map["거래"])

    return "\n".join(matched)


# --- DKAP: Full L2 structured domain artifacts ---
PROMPT_DKAP = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.

## Database Schema (structured)

### Table: ods_card_member (카드 회원정보)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 기준년월 | TEXT NOT NULL | 데이터 기준 연월 (YYYYMM) | date |
| 발급회원번호 | TEXT PRIMARY KEY | 카드 회원 고유 식별 번호 | identifier |
| 고객명 | TEXT | 고객 성명 | PII |
| VIP등급코드 | TEXT | VIP 등급 (01:VVIP, 02:VIP, 03:Gold, 04:일반) | category |
| 입회일자 | TEXT | 카드 입회 일자 (YYYYMMDD) | date |
| 최종이용일자 | TEXT | 최종 카드 이용 일자 | date |
| 신용등급 | INTEGER | 개인 신용 등급 (1~10, 1이 최우수) | category |
| 남녀구분코드 | TEXT | 성별 (M:남성, F:여성) | category |
| 연령대코드 | TEXT | 연령대 (20/30/40/50/60) | category |

### Table: ods_card_transaction (카드 승인매출정보)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 거래일련번호 | INTEGER PRIMARY KEY AUTOINCREMENT | 거래 고유 번호 | identifier |
| 기준년월 | TEXT NOT NULL | 데이터 기준 연월 (YYYYMM) | date |
| 발급회원번호 | TEXT NOT NULL (FK→ods_card_member) | 카드 회원 번호 | identifier |
| 승인번호 | TEXT | 거래 승인 번호 | identifier |
| 가맹점명 | TEXT | 가맹점 상호명 | text |
| 업종코드 | TEXT | 업종 분류 코드 | category |
| 이용금액 | INTEGER | 카드 이용 금액 (원) | financial_amount |
| 이용일자 | TEXT | 이용 일자 (YYYYMMDD) | date |
| 할부개월수 | INTEGER DEFAULT 0 | 할부 개월 수 (0=일시불) | count |

### Table: ods_card_credit (카드 신용정보)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 기준년월 | TEXT NOT NULL (PK) | 데이터 기준 연월 | date |
| 발급회원번호 | TEXT NOT NULL (PK, FK→ods_card_member) | 카드 회원 번호 | identifier |
| 카드이용한도금액 | INTEGER | 카드 이용 한도 금액 (원) | financial_amount |
| 잔액 | INTEGER | 현재 잔액 (원) | financial_amount |
| 연체일수 | INTEGER DEFAULT 0 | 연체 일수 | count |
| 연체잔액 | INTEGER DEFAULT 0 | 연체 잔액 (원) | financial_amount |
| 한도소진율 | REAL | 한도 소진율 (0.0~1.0) | ratio |

### Table: ods_card_billing (카드 청구정보)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 기준년월 | TEXT NOT NULL (PK) | 데이터 기준 연월 | date |
| 발급회원번호 | TEXT NOT NULL (PK, FK→ods_card_member) | 카드 회원 번호 | identifier |
| 청구금액 | INTEGER | 청구 금액 (원) | financial_amount |
| 결제일 | TEXT | 결제 예정일 (DD) | date |
| 납부상태코드 | TEXT | 납부 상태 (01:완납, 02:미납, 03:부분납) | category |

### Table: ods_fin_product (금융상품정보)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 상품코드 | TEXT PRIMARY KEY | 상품 고유 코드 | identifier |
| 상품명 | TEXT | 금융상품명 | text |
| 상품유형 | TEXT | 유형 (카드/대출/펀드/보험) | category |
| 기준금리 | REAL | 기준 금리 (%) | ratio |
| 가입건수 | INTEGER DEFAULT 0 | 누적 가입 건수 | count |

## Domain Glossary (금융 용어 사전)
- 기준년월: 데이터 기준 연월, YYYYMM 형식. 파티션 키로 사용.
- 발급회원번호: 카드 회원 고유 식별 번호. 모든 테이블의 조인 키.
- VIP등급코드: 01=VVIP, 02=VIP, 03=Gold, 04=일반. 숫자가 작을수록 상위 등급.
- 신용등급: 1~10 등급, 1이 최우수. 금감원 기준.
- 한도소진율: 잔액/카드이용한도금액. 0.0~1.0 범위. 높을수록 위험.
- 납부상태코드: 01=완납, 02=미납, 03=부분납.
- 이용금액: 카드 이용 금액, 원(KRW) 단위.
- 할부개월수: 0이면 일시불, 1 이상이면 할부.
- 연체일수: 0이면 정상, 양수이면 연체 중.

## Join Rules
- ods_card_member ↔ ods_card_transaction: ON 발급회원번호 AND 기준년월
- ods_card_member ↔ ods_card_credit: ON 발급회원번호 AND 기준년월
- ods_card_member ↔ ods_card_billing: ON 발급회원번호 AND 기준년월
- ods_fin_product: 독립 테이블 (상품코드로 조인 가능하나 현재 FK 없음)

## Column Transform Rules (from ai-governance-bridge L2 artifacts)
- 금액 컬럼 (이용금액, 청구금액, 잔액 등): 원(KRW) 단위 정수. 비교 시 숫자 직접 사용.
- 일자 컬럼 (입회일자, 이용일자 등): YYYYMMDD 문자열. 비교 시 문자열 비교 가능.
- 등급/코드 컬럼: 문자열 비교. 숫자처럼 보이지만 TEXT 타입.
- 비율 컬럼 (한도소진율, 기준금리): REAL 타입 0.0~1.0 범위.

Output ONLY the SQL query, nothing else. Do not include explanation.

Question: {question}

SQL:"""


# ============================================================
# 4. LLM Inference
# ============================================================
def call_vllm(prompt: str, max_retries: int = 3) -> str:
    """Call vLLM OpenAI-compatible API."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stop": ["Question:", "---"],
        "chat_template_kwargs": {"enable_thinking": False}
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Remove Qwen3 <think>...</think> reasoning blocks
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            # Remove incomplete <think> block (if truncated)
            content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL).strip()
            # Clean up: remove markdown code fences if present
            content = re.sub(r'^```sql\s*', '', content)
            content = re.sub(r'^```\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()
            # Rejoin multi-line SQL into single line
            content = ' '.join(content.split())
            # Remove trailing semicolons for consistency
            content = content.rstrip(';').strip()
            return content
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"All retries failed for prompt")
                return ""
    return ""


# ============================================================
# 5. Evaluation Functions
# ============================================================
def setup_database(db_path: Path) -> sqlite3.Connection:
    """Create and populate the benchmark database."""
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_DDL)
    conn.executescript(SEED_DATA_SQL)
    conn.commit()
    return conn


def execute_sql(conn: sqlite3.Connection, sql: str) -> Tuple[bool, Optional[List]]:
    """Execute SQL and return (success, results)."""
    try:
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        return True, results
    except Exception as e:
        return False, None


def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact match comparison."""
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';').strip()
    return sql


def evaluate_ex(conn: sqlite3.Connection, pred_sql: str, gold_sql: str) -> bool:
    """Execution Accuracy: do predicted and gold SQL produce same results?"""
    pred_ok, pred_results = execute_sql(conn, pred_sql)
    gold_ok, gold_results = execute_sql(conn, gold_sql)

    if not pred_ok or not gold_ok:
        return pred_ok == gold_ok  # Both fail = technically match, but we count as False

    if pred_results is None or gold_results is None:
        return False

    # Convert to sets of tuples for order-independent comparison
    # (unless ORDER BY is in gold SQL, then order matters)
    if "order by" in gold_sql.lower():
        return pred_results == gold_results
    else:
        return set(map(tuple, pred_results)) == set(map(tuple, gold_results))


def evaluate_em(pred_sql: str, gold_sql: str) -> bool:
    """Exact Match: normalized SQL string comparison."""
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


# ============================================================
# 6. Main Experiment Runner
# ============================================================
@dataclass
class ExperimentResult:
    condition: str
    run_id: int
    question_id: str
    difficulty: str
    nl_question: str
    gold_sql: str
    pred_sql: str
    ex_score: bool
    em_score: bool
    execution_ok: bool
    latency_ms: float


def run_experiment():
    """Run the full B1/B2/DKAP experiment."""
    logger.info("=" * 60)
    logger.info("DKAP Finance Text-to-SQL Experiment")
    logger.info("=" * 60)

    # Setup database
    logger.info("Setting up benchmark database...")
    conn = setup_database(DB_PATH)

    # Verify database
    cursor = conn.execute("SELECT COUNT(*) FROM ods_card_member")
    logger.info(f"  Members: {cursor.fetchone()[0]} rows")
    cursor = conn.execute("SELECT COUNT(*) FROM ods_card_transaction")
    logger.info(f"  Transactions: {cursor.fetchone()[0]} rows")
    cursor = conn.execute("SELECT COUNT(*) FROM ods_card_credit")
    logger.info(f"  Credit: {cursor.fetchone()[0]} rows")

    # Test vLLM connectivity
    logger.info("Testing vLLM connectivity...")
    test_resp = call_vllm("SELECT 1")
    if not test_resp:
        logger.error("Cannot connect to vLLM. Aborting.")
        sys.exit(1)
    logger.info(f"  vLLM OK. Test response: {test_resp[:50]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results: List[ExperimentResult] = []

    conditions = {
        "B1": lambda q: PROMPT_B1.format(question=q),
        "B2": lambda q: PROMPT_B2.format(question=q, retrieved_chunks=get_b2_chunks(q)),
        "DKAP": lambda q: PROMPT_DKAP.format(question=q),
    }

    total_queries = len(BENCHMARK) * len(conditions) * NUM_RUNS
    progress = 0

    for run_id in range(1, NUM_RUNS + 1):
        logger.info(f"\n--- Run {run_id}/{NUM_RUNS} ---")
        for cond_name, prompt_fn in conditions.items():
            logger.info(f"  Condition: {cond_name}")
            for item in BENCHMARK:
                progress += 1
                qid = item["id"]
                prompt = prompt_fn(item["nl"])

                start_time = time.time()
                pred_sql = call_vllm(prompt)
                latency_ms = (time.time() - start_time) * 1000

                # Evaluate
                ex_ok, _ = execute_sql(conn, pred_sql)
                ex_score = evaluate_ex(conn, pred_sql, item["gold_sql"]) if ex_ok else False
                em_score = evaluate_em(pred_sql, item["gold_sql"])

                result = ExperimentResult(
                    condition=cond_name,
                    run_id=run_id,
                    question_id=qid,
                    difficulty=item["difficulty"],
                    nl_question=item["nl"],
                    gold_sql=item["gold_sql"],
                    pred_sql=pred_sql,
                    ex_score=ex_score,
                    em_score=em_score,
                    execution_ok=ex_ok,
                    latency_ms=latency_ms
                )
                all_results.append(result)

                status = "EX" if ex_score else ("EXEC" if ex_ok else "FAIL")
                logger.info(f"    [{progress}/{total_queries}] {qid} ({item['difficulty']}) -> {status} ({latency_ms:.0f}ms)")

    conn.close()

    # ============================================================
    # 7. Results Analysis
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS ANALYSIS")
    logger.info("=" * 60)

    # Save raw results
    raw_path = RESULTS_DIR / "raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    logger.info(f"Raw results saved to {raw_path}")

    # Aggregate by condition
    summary = {}
    for cond in ["B1", "B2", "DKAP"]:
        cond_results = [r for r in all_results if r.condition == cond]
        n = len(cond_results)
        ex_total = sum(1 for r in cond_results if r.ex_score)
        em_total = sum(1 for r in cond_results if r.em_score)
        exec_total = sum(1 for r in cond_results if r.execution_ok)
        avg_latency = sum(r.latency_ms for r in cond_results) / n if n > 0 else 0

        # By difficulty
        diff_breakdown = {}
        for diff in ["E", "M", "H"]:
            diff_results = [r for r in cond_results if r.difficulty == diff]
            dn = len(diff_results)
            if dn > 0:
                diff_breakdown[diff] = {
                    "n": dn,
                    "EX": sum(1 for r in diff_results if r.ex_score),
                    "EX_pct": round(sum(1 for r in diff_results if r.ex_score) * 100 / dn, 1),
                    "EM": sum(1 for r in diff_results if r.em_score),
                    "EM_pct": round(sum(1 for r in diff_results if r.em_score) * 100 / dn, 1),
                }

        # By run (for variance)
        run_ex = {}
        for run_id in range(1, NUM_RUNS + 1):
            run_results = [r for r in cond_results if r.run_id == run_id]
            rn = len(run_results)
            run_ex[f"run_{run_id}"] = round(sum(1 for r in run_results if r.ex_score) * 100 / rn, 1) if rn > 0 else 0

        summary[cond] = {
            "n": n,
            "EX_total": ex_total,
            "EX_pct": round(ex_total * 100 / n, 1) if n > 0 else 0,
            "EM_total": em_total,
            "EM_pct": round(em_total * 100 / n, 1) if n > 0 else 0,
            "EXEC_total": exec_total,
            "EXEC_pct": round(exec_total * 100 / n, 1) if n > 0 else 0,
            "avg_latency_ms": round(avg_latency, 1),
            "by_difficulty": diff_breakdown,
            "by_run": run_ex,
        }

    # Print summary table
    logger.info("\n=== Overall Results ===")
    logger.info(f"{'Condition':<10} {'EX(%)':<10} {'EM(%)':<10} {'EXEC(%)':<10} {'Latency(ms)':<12}")
    logger.info("-" * 52)
    for cond in ["B1", "B2", "DKAP"]:
        s = summary[cond]
        logger.info(f"{cond:<10} {s['EX_pct']:<10} {s['EM_pct']:<10} {s['EXEC_pct']:<10} {s['avg_latency_ms']:<12}")

    logger.info("\n=== By Difficulty ===")
    for cond in ["B1", "B2", "DKAP"]:
        logger.info(f"\n  {cond}:")
        for diff in ["E", "M", "H"]:
            d = summary[cond]["by_difficulty"].get(diff, {})
            if d:
                logger.info(f"    {diff}: EX={d['EX_pct']}% ({d['EX']}/{d['n']}), EM={d['EM_pct']}% ({d['EM']}/{d['n']})")

    logger.info("\n=== Run Variance (EX%) ===")
    for cond in ["B1", "B2", "DKAP"]:
        runs = summary[cond]["by_run"]
        logger.info(f"  {cond}: {runs}")

    # DKAP improvement delta
    if summary["B1"]["EX_pct"] > 0:
        delta_b1 = summary["DKAP"]["EX_pct"] - summary["B1"]["EX_pct"]
        delta_b2 = summary["DKAP"]["EX_pct"] - summary["B2"]["EX_pct"]
        logger.info(f"\n=== DKAP Improvement ===")
        logger.info(f"  DKAP vs B1: +{delta_b1:.1f}pp")
        logger.info(f"  DKAP vs B2: +{delta_b2:.1f}pp")

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    # Save failed queries for analysis
    failed_path = RESULTS_DIR / "failed_queries.json"
    failed = [asdict(r) for r in all_results if not r.ex_score and r.run_id == 1]
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)
    logger.info(f"Failed queries saved to {failed_path}")

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    summary = run_experiment()
