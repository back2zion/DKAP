#!/usr/bin/env python3
"""
B1_DDL Condition Experiment
============================
Adds a B1_DDL intermediate condition to the finance ablation:
  - B1:     zero-shot (no schema, no domain knowledge)         → 0% EX [existing]
  - B1_DDL: raw DDL schema only (table/column names + types)   → ??? [this script]
  - B2:     flat-chunk RAG                                     → 54% EX [existing]
  - DKAP:   full structured L2                                 → 80% EX [existing]

B1_DDL isolates the effect of having schema structure
without semantic annotations (glossary, join rules, value mappings).
This addresses the reviewer concern that B1=0% may reflect the absence of
ANY schema information, not the absence of structured domain knowledge.

Author: Dooil Kwak, 2026-05-22
"""

import json, time, sqlite3, re, sys, os
from pathlib import Path
try:
    import requests
except ImportError:
    import subprocess; subprocess.run([sys.executable,"-m","pip","install","requests","-q"])
    import requests

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME    = "Qwen3.6-27B-AWQ"
TEMPERATURE   = 0.0
NUM_RUNS      = 3
DB_PATH       = Path("/tmp/finance_b1ddl.db")
RESULTS_DIR   = Path("results/b1_ddl")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Schema (DDL only, no semantic comments)
# ============================================================
DDL_ONLY = """
CREATE TABLE ods_card_member (
    기준년월 TEXT NOT NULL,
    발급회원번호 TEXT NOT NULL,
    고객명 TEXT,
    VIP등급코드 TEXT,
    입회일자 TEXT,
    최종이용일자 TEXT,
    신용등급 INTEGER,
    남녀구분코드 TEXT,
    연령대코드 TEXT,
    PRIMARY KEY (기준년월, 발급회원번호)
);

CREATE TABLE ods_card_transaction (
    거래일련번호 INTEGER PRIMARY KEY AUTOINCREMENT,
    기준년월 TEXT NOT NULL,
    발급회원번호 TEXT NOT NULL,
    승인번호 TEXT,
    가맹점명 TEXT,
    업종코드 TEXT,
    이용금액 INTEGER,
    이용일자 TEXT,
    할부개월수 INTEGER DEFAULT 0,
    FOREIGN KEY (발급회원번호) REFERENCES ods_card_member(발급회원번호)
);

CREATE TABLE ods_card_credit (
    기준년월 TEXT NOT NULL,
    발급회원번호 TEXT NOT NULL,
    카드이용한도금액 INTEGER,
    잔액 INTEGER,
    연체일수 INTEGER DEFAULT 0,
    연체잔액 INTEGER DEFAULT 0,
    한도소진율 REAL,
    PRIMARY KEY (기준년월, 발급회원번호),
    FOREIGN KEY (발급회원번호) REFERENCES ods_card_member(발급회원번호)
);

CREATE TABLE ods_card_billing (
    기준년월 TEXT NOT NULL,
    발급회원번호 TEXT NOT NULL,
    청구금액 INTEGER,
    결제일 TEXT,
    납부상태코드 TEXT,
    PRIMARY KEY (기준년월, 발급회원번호),
    FOREIGN KEY (발급회원번호) REFERENCES ods_card_member(발급회원번호)
);

CREATE TABLE ods_fin_product (
    상품코드 TEXT PRIMARY KEY,
    상품명 TEXT,
    상품유형 TEXT,
    기준금리 REAL,
    가입건수 INTEGER DEFAULT 0
);
"""

SEED_DATA_SQL = """
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
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M001','A001','스타벅스 강남점','카페',5500,'20250115',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M001','A002','삼성전자 직영점','전자제품',1890000,'20250120',12);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M002','A004','올리브영 홍대점','화장품',45000,'20250110',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M003','A006','현대백화점 본점','백화점',2450000,'20250105',6);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M003','A007','GS칼텍스 주유소','주유',85000,'20250112',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M004','A008','넷플릭스','구독',17000,'20250101',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M005','A010','서울아산병원','의료',350000,'20250108',3);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M005','A011','하나투어','여행',4200000,'20250115',6);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M006','A012','쿠팡','온라인쇼핑',89000,'20250120',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M007','A014','교보문고 광화문','도서',52000,'20250118',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M008','A016','유니클로 명동점','의류',129000,'20250110',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M009','A017','대한항공','항공',890000,'20250105',3);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M009','A018','롯데호텔','숙박',320000,'20250106',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M010','A019','애플스토어 가로수길','전자제품',1590000,'20250115',12);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202501','M010','A020','스시조','음식점',78000,'20250128',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M001','B001','스타벅스 강남점','카페',6000,'20250215',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M001','B002','쿠팡','온라인쇼핑',234000,'20250218',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M002','B003','올리브영 홍대점','화장품',67000,'20250210',0);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M003','B004','현대백화점 본점','백화점',1280000,'20250220',3);
INSERT INTO ods_card_transaction (기준년월,발급회원번호,승인번호,가맹점명,업종코드,이용금액,이용일자,할부개월수) VALUES ('202502','M005','B005','하나투어','여행',2100000,'20250225',6);
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
INSERT INTO ods_fin_product VALUES ('P001','프리미엄 골드카드','카드',0.0,15200);
INSERT INTO ods_fin_product VALUES ('P002','스마트 신용대출','대출',4.5,8900);
INSERT INTO ods_fin_product VALUES ('P003','글로벌 주식펀드','펀드',0.0,3200);
INSERT INTO ods_fin_product VALUES ('P004','무배당 종합보험','보험',0.0,12100);
INSERT INTO ods_fin_product VALUES ('P005','직장인 체크카드','카드',0.0,45000);
"""

# Load benchmark from existing experiment (Q01-Q13 Easy, Q14-Q32 Medium, Q33-Q50 Hard)
# We use Q01-Q10 (E), Q14-Q22 (M), Q33-Q40 (H) as representative subset = 27 questions
# matching the structure used in the original experiment
BENCHMARK_SUBSET = [
    # Easy
    {"id":"Q01","difficulty":"E","nl":"전체 회원 수를 구하시오. (기준년월 202501)","gold_sql":"SELECT COUNT(DISTINCT 발급회원번호) FROM ods_card_member WHERE 기준년월='202501'"},
    {"id":"Q02","difficulty":"E","nl":"VIP등급이 VVIP(01)인 회원의 이름을 모두 조회하시오. (기준년월 202501)","gold_sql":"SELECT 고객명 FROM ods_card_member WHERE VIP등급코드='01' AND 기준년월='202501'"},
    {"id":"Q03","difficulty":"E","nl":"2025년 1월 전체 거래 건수를 구하시오.","gold_sql":"SELECT COUNT(*) FROM ods_card_transaction WHERE 기준년월='202501'"},
    {"id":"Q04","difficulty":"E","nl":"신용등급이 3 이하(우수)인 회원 수를 구하시오. (기준년월 202501)","gold_sql":"SELECT COUNT(*) FROM ods_card_member WHERE 신용등급<=3 AND 기준년월='202501'"},
    {"id":"Q05","difficulty":"E","nl":"2025년 1월 가장 큰 이용금액의 거래를 조회하시오.","gold_sql":"SELECT * FROM ods_card_transaction WHERE 기준년월='202501' ORDER BY 이용금액 DESC LIMIT 1"},
    {"id":"Q06","difficulty":"E","nl":"연체일수가 0보다 큰 회원번호를 모두 조회하시오. (기준년월 202501)","gold_sql":"SELECT 발급회원번호 FROM ods_card_credit WHERE 연체일수>0 AND 기준년월='202501'"},
    {"id":"Q07","difficulty":"E","nl":"카드 유형 상품의 누적 가입건수 합계를 구하시오.","gold_sql":"SELECT SUM(가입건수) FROM ods_fin_product WHERE 상품유형='카드'"},
    {"id":"Q08","difficulty":"E","nl":"2025년 1월 여성(F) 회원 수를 구하시오.","gold_sql":"SELECT COUNT(*) FROM ods_card_member WHERE 남녀구분코드='F' AND 기준년월='202501'"},
    {"id":"Q09","difficulty":"E","nl":"납부상태가 미납(02)인 회원의 청구금액을 조회하시오. (기준년월 202501)","gold_sql":"SELECT 발급회원번호, 청구금액 FROM ods_card_billing WHERE 납부상태코드='02' AND 기준년월='202501'"},
    {"id":"Q10","difficulty":"E","nl":"기준금리가 0보다 큰 금융상품의 상품명과 금리를 조회하시오.","gold_sql":"SELECT 상품명, 기준금리 FROM ods_fin_product WHERE 기준금리>0"},
    {"id":"Q11","difficulty":"E","nl":"2025년 1월 30대 회원 수를 조회하시오.","gold_sql":"SELECT COUNT(*) FROM ods_card_member WHERE 연령대코드='30' AND 기준년월='202501'"},
    {"id":"Q12","difficulty":"E","nl":"할부 거래(할부개월수>0)의 건수와 총 이용금액을 조회하시오. (기준년월 202501)","gold_sql":"SELECT COUNT(*), SUM(이용금액) FROM ods_card_transaction WHERE 할부개월수>0 AND 기준년월='202501'"},
    {"id":"Q13","difficulty":"E","nl":"한도소진율이 0.5 이상인 회원번호를 조회하시오. (기준년월 202501)","gold_sql":"SELECT 발급회원번호 FROM ods_card_credit WHERE 한도소진율>=0.5 AND 기준년월='202501'"},
    # Medium
    {"id":"Q14","difficulty":"M","nl":"2025년 1월 회원별 총 이용금액을 조회하시오. (내림차순)","gold_sql":"SELECT 발급회원번호, SUM(이용금액) AS 총이용금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 발급회원번호 ORDER BY 총이용금액 DESC"},
    {"id":"Q15","difficulty":"M","nl":"VVIP 회원(VIP등급코드=01)의 2025년 1월 카드 이용금액 합계를 구하시오.","gold_sql":"SELECT SUM(t.이용금액) FROM ods_card_transaction t JOIN ods_card_member m ON t.발급회원번호=m.발급회원번호 WHERE m.VIP등급코드='01' AND t.기준년월='202501' AND m.기준년월='202501'"},
    {"id":"Q16","difficulty":"M","nl":"2025년 1월 업종코드별 이용금액 합계를 조회하시오.","gold_sql":"SELECT 업종코드, SUM(이용금액) AS 합계 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 업종코드 ORDER BY 합계 DESC"},
    {"id":"Q17","difficulty":"M","nl":"신용등급 1인 회원의 카드이용한도금액을 조회하시오. (기준년월 202501)","gold_sql":"SELECT c.발급회원번호, c.카드이용한도금액 FROM ods_card_credit c JOIN ods_card_member m ON c.발급회원번호=m.발급회원번호 AND c.기준년월=m.기준년월 WHERE m.신용등급=1 AND m.기준년월='202501'"},
    {"id":"Q18","difficulty":"M","nl":"2025년 1월과 2025년 2월 두 달 모두 거래가 있는 회원번호를 조회하시오.","gold_sql":"SELECT DISTINCT t1.발급회원번호 FROM ods_card_transaction t1 JOIN ods_card_transaction t2 ON t1.발급회원번호=t2.발급회원번호 WHERE t1.기준년월='202501' AND t2.기준년월='202502'"},
    {"id":"Q19","difficulty":"M","nl":"각 회원의 청구금액 대비 이용금액 비율을 구하시오. (기준년월 202501)","gold_sql":"SELECT t.발급회원번호, SUM(t.이용금액)*1.0/b.청구금액 AS 비율 FROM ods_card_transaction t JOIN ods_card_billing b ON t.발급회원번호=b.발급회원번호 AND t.기준년월=b.기준년월 WHERE t.기준년월='202501' GROUP BY t.발급회원번호, b.청구금액"},
    {"id":"Q20","difficulty":"M","nl":"평균 이용금액이 100,000원 이상인 업종코드를 조회하시오. (기준년월 202501)","gold_sql":"SELECT 업종코드, AVG(이용금액) AS 평균금액 FROM ods_card_transaction WHERE 기준년월='202501' GROUP BY 업종코드 HAVING AVG(이용금액)>=100000"},
    {"id":"Q21","difficulty":"M","nl":"연체잔액이 있는 회원의 이름과 신용등급을 조회하시오. (기준년월 202501)","gold_sql":"SELECT m.고객명, m.신용등급 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE c.연체잔액>0 AND c.기준년월='202501'"},
    {"id":"Q22","difficulty":"M","nl":"카드상품과 대출상품의 총 가입건수 차이를 구하시오.","gold_sql":"SELECT ABS(SUM(CASE WHEN 상품유형='카드' THEN 가입건수 ELSE 0 END)-SUM(CASE WHEN 상품유형='대출' THEN 가입건수 ELSE 0 END)) AS 차이 FROM ods_fin_product"},
    # Hard
    {"id":"Q33","difficulty":"H","nl":"2025년 1월 VVIP 회원의 업종별 이용금액 비중을 구하시오. (비중=업종금액/회원총금액)","gold_sql":"SELECT t.발급회원번호, t.업종코드, SUM(t.이용금액) AS 업종금액, SUM(t.이용금액)*1.0/SUM(SUM(t.이용금액)) OVER (PARTITION BY t.발급회원번호) AS 비중 FROM ods_card_transaction t JOIN ods_card_member m ON t.발급회원번호=m.발급회원번호 AND t.기준년월=m.기준년월 WHERE m.VIP등급코드='01' AND t.기준년월='202501' GROUP BY t.발급회원번호, t.업종코드"},
    {"id":"Q34","difficulty":"H","nl":"한도소진율이 높을수록 연체일수가 많은지 상관관계를 확인할 수 있는 데이터를 추출하시오. (기준년월 202501)","gold_sql":"SELECT 발급회원번호, 한도소진율, 연체일수 FROM ods_card_credit WHERE 기준년월='202501' ORDER BY 한도소진율 DESC"},
    {"id":"Q35","difficulty":"H","nl":"2025년 1월과 2025년 2월 이용금액 증감률이 가장 높은 회원을 조회하시오.","gold_sql":"SELECT t1.발급회원번호, (SUM(t2.이용금액)-SUM(t1.이용금액))*1.0/SUM(t1.이용금액) AS 증감률 FROM ods_card_transaction t1 JOIN ods_card_transaction t2 ON t1.발급회원번호=t2.발급회원번호 WHERE t1.기준년월='202501' AND t2.기준년월='202502' GROUP BY t1.발급회원번호 ORDER BY 증감률 DESC LIMIT 1"},
    {"id":"Q36","difficulty":"H","nl":"각 회원의 신용등급별 평균 한도소진율을 구하고 신용등급이 낮을수록 한도소진율이 높은지 확인하시오. (기준년월 202501)","gold_sql":"SELECT m.신용등급, AVG(c.한도소진율) AS 평균한도소진율 FROM ods_card_member m JOIN ods_card_credit c ON m.발급회원번호=c.발급회원번호 AND m.기준년월=c.기준년월 WHERE m.기준년월='202501' GROUP BY m.신용등급 ORDER BY m.신용등급"},
    {"id":"Q37","difficulty":"H","nl":"미납 회원의 총 청구금액과 미납 회원의 연체잔액 합계를 함께 조회하시오. (기준년월 202501)","gold_sql":"SELECT b.발급회원번호, b.청구금액, c.연체잔액 FROM ods_card_billing b JOIN ods_card_credit c ON b.발급회원번호=c.발급회원번호 AND b.기준년월=c.기준년월 WHERE b.납부상태코드='02' AND b.기준년월='202501'"},
    {"id":"Q38","difficulty":"H","nl":"2025년 1월 기준 30대이면서 신용등급 3 이하이고 이용금액 상위 3명을 구하시오.","gold_sql":"SELECT m.발급회원번호, m.고객명, SUM(t.이용금액) AS 총이용금액 FROM ods_card_member m JOIN ods_card_transaction t ON m.발급회원번호=t.발급회원번호 AND m.기준년월=t.기준년월 WHERE m.연령대코드='30' AND m.신용등급<=3 AND m.기준년월='202501' GROUP BY m.발급회원번호, m.고객명 ORDER BY 총이용금액 DESC LIMIT 3"},
]

B1_DDL_PROMPT = """Generate a SQLite SQL query to answer the following question.

## Database Schema (DDL)
{ddl}

## Question
{question}

Output only the SQL query, no explanation."""


def setup_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL_ONLY)
    conn.executescript(SEED_DATA_SQL)
    conn.commit()
    return conn


def extract_sql(raw: str) -> str:
    raw = raw.strip()
    # Remove markdown code blocks
    raw = re.sub(r'```sql\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```\s*', '', raw)
    # Take first statement
    lines = [l for l in raw.split('\n') if l.strip() and not l.strip().startswith('--')]
    return ' '.join(lines).strip().rstrip(';')


def execute_sql(conn, sql: str, gold_sql: str) -> dict:
    try:
        cur = conn.execute(sql)
        result = sorted([str(tuple(r)) for r in cur.fetchall()])
        gold_cur = conn.execute(gold_sql)
        gold_result = sorted([str(tuple(r)) for r in gold_cur.fetchall()])
        ex_pass = (result == gold_result)
        em_pass = (sql.strip().lower() == gold_sql.strip().lower())
        return {"EX": ex_pass, "EM": em_pass, "EXEC": True, "result": result[:3]}
    except Exception as e:
        return {"EX": False, "EM": False, "EXEC": False, "error": str(e)}


def call_vllm(user_content: str) -> tuple:
    # Use /completions with pre-filled empty think block so Qwen3 skips thinking mode
    # This approach avoids the vLLM 0.18.0 chat_template_kwargs limitation
    filled_prompt = (
        "<|im_start|>system\n"
        "You are a SQL expert. Output ONLY valid SQLite SQL. No explanations.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )
    t0 = time.time()
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/completions",
            json={
                "model": MODEL_NAME,
                "prompt": filled_prompt,
                "max_tokens": 256,
                "temperature": TEMPERATURE,
                "stop": ["<|im_end|>", "<|im_start|>"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["text"].strip()
        return text, int((time.time() - t0) * 1000)
    except Exception as e:
        return f"ERROR: {e}", 0


def main():
    print("=" * 60)
    print("B1_DDL Condition Experiment (Finance, n=27 queries)")
    print("=" * 60)

    # Wait for vLLM
    for attempt in range(30):
        try:
            r = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
            if r.status_code == 200:
                print(f"vLLM ready.")
                break
        except:
            pass
        print(f"Waiting for vLLM... ({attempt+1}/30)")
        time.sleep(20)
    else:
        print("ERROR: vLLM timeout.")
        sys.exit(1)

    conn = setup_db()
    print(f"DB ready: {DB_PATH}, {len(BENCHMARK_SUBSET)} queries")

    all_runs = []
    for run_idx in range(NUM_RUNS):
        print(f"\n--- Run {run_idx+1}/{NUM_RUNS} ---")
        run_results = []
        for q in BENCHMARK_SUBSET:
            prompt = B1_DDL_PROMPT.format(ddl=DDL_ONLY, question=q["nl"])
            raw_sql, lat = call_vllm(prompt)
            sql = extract_sql(raw_sql)
            eval_result = execute_sql(conn, sql, q["gold_sql"])
            record = {
                "run": run_idx + 1,
                "id": q["id"],
                "difficulty": q["difficulty"],
                "generated_sql": sql,
                "latency_ms": lat,
                **eval_result,
            }
            run_results.append(record)
            status = "✓EX" if eval_result["EX"] else ("✓EXEC" if eval_result["EXEC"] else "✗")
            print(f"  {q['id']} ({q['difficulty']}): {status} {lat}ms")
        all_runs.extend(run_results)

    conn.close()

    # Save raw
    out_path = RESULTS_DIR / "b1_ddl_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved: {out_path}")

    # Aggregate
    diffs = {"E": [], "M": [], "H": []}
    for r in all_runs:
        diffs[r["difficulty"]].append(r)

    print("\n" + "=" * 60)
    print("B1_DDL RESULTS SUMMARY")
    print("=" * 60)
    totals = {"EX": 0, "EXEC": 0, "total": 0, "lat": []}
    for diff, label in [("E","Easy"), ("M","Medium"), ("H","Hard")]:
        records = diffs[diff]
        ex = sum(1 for r in records if r["EX"])
        exc = sum(1 for r in records if r["EXEC"])
        n = len(records)
        lats = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
        print(f"{label}: EX={ex}/{n} ({100*ex/n:.1f}%), EXEC={exc}/{n} ({100*exc/n:.1f}%), "
              f"AvgLat={sum(lats)//len(lats) if lats else 0}ms")
        totals["EX"] += ex; totals["EXEC"] += exc; totals["total"] += n
        totals["lat"].extend(lats)

    n = totals["total"]
    print(f"Overall: EX={totals['EX']}/{n} ({100*totals['EX']/n:.1f}%), "
          f"EXEC={totals['EXEC']}/{n} ({100*totals['EXEC']/n:.1f}%)")
    avglat = sum(totals["lat"]) // len(totals["lat"]) if totals["lat"] else 0
    print(f"AvgLatency: {avglat}ms")

    summary = {
        "condition": "B1_DDL",
        "description": "Raw DDL schema provided, no semantic annotations, no RAG",
        "n_queries": n // NUM_RUNS,
        "n_runs": NUM_RUNS,
        "EX_total": totals["EX"],
        "EX_pct": round(100 * totals["EX"] / n, 1),
        "EXEC_total": totals["EXEC"],
        "EXEC_pct": round(100 * totals["EXEC"] / n, 1),
        "avg_latency_ms": avglat,
        "by_difficulty": {
            d: {
                "EX_pct": round(100 * sum(1 for r in diffs[d] if r["EX"]) / len(diffs[d]), 1),
                "EXEC_pct": round(100 * sum(1 for r in diffs[d] if r["EXEC"]) / len(diffs[d]), 1),
                "n": len(diffs[d]),
            }
            for d in ["E","M","H"]
        },
    }
    with open(RESULTS_DIR / "b1_ddl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {RESULTS_DIR/'b1_ddl_summary.json'}")
    print("\nContext for paper:")
    print(f"  B1 (zero-shot):  0.0% EX  [existing]")
    print(f"  B1_DDL (schema): {summary['EX_pct']}% EX  [new]")
    print(f"  B2 (flat RAG):  54.0% EX  [existing]")
    print(f"  DKAP:           80.0% EX  [existing]")


if __name__ == "__main__":
    main()
