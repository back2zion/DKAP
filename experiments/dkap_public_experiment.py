#!/usr/bin/env python3
"""
DKAP Public/Legal Domain Experiment (Preliminary)
==================================================
Legal QA over Korean public administration schema.
Three conditions: B1 / B2 / DKAP.
Model: Qwen3-32B-AWQ via vLLM (localhost:8000)
"""
import json, time, sqlite3, os, sys, re, logging
from pathlib import Path
from dataclasses import dataclass, asdict
try:
    import requests
except ImportError:
    os.system("pip install requests --break-system-packages -q")
    import requests

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "default-model"
MAX_TOKENS = 2048
TEMPERATURE = 0.0
NUM_RUNS = 3
RESULTS_DIR = Path("/home/aigen/dkap_public_results")
DB_PATH = Path("/tmp/public_benchmark.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dkap_public_experiment.log")])
logger = logging.getLogger(__name__)

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS bid_announcement (
    공고번호        TEXT PRIMARY KEY,
    공고명          TEXT NOT NULL,
    발주기관명      TEXT NOT NULL,
    공고일자        TEXT NOT NULL,
    마감일자        TEXT NOT NULL,
    추정가격        INTEGER,
    공고유형        TEXT,
    입찰방식        TEXT,
    업종제한        TEXT,
    지역제한        TEXT
);

CREATE TABLE IF NOT EXISTS bid_detail (
    공고번호        TEXT NOT NULL,
    세부항목번호    INTEGER NOT NULL,
    품목명          TEXT,
    수량            INTEGER,
    단위            TEXT,
    예정단가        INTEGER,
    규격            TEXT,
    PRIMARY KEY (공고번호, 세부항목번호),
    FOREIGN KEY (공고번호) REFERENCES bid_announcement(공고번호)
);

CREATE TABLE IF NOT EXISTS agency (
    기관코드        TEXT PRIMARY KEY,
    기관명          TEXT NOT NULL,
    기관유형        TEXT,
    상위기관코드    TEXT,
    지역            TEXT,
    설립연도        INTEGER
);

CREATE TABLE IF NOT EXISTS contractor (
    사업자번호      TEXT PRIMARY KEY,
    업체명          TEXT NOT NULL,
    대표자명        TEXT,
    업종            TEXT,
    소재지          TEXT,
    설립연도        INTEGER,
    자본금          INTEGER
);

CREATE TABLE IF NOT EXISTS contract (
    계약번호        TEXT PRIMARY KEY,
    공고번호        TEXT NOT NULL,
    사업자번호      TEXT NOT NULL,
    계약금액        INTEGER,
    계약일자        TEXT,
    납품기한        TEXT,
    계약상태        TEXT,
    FOREIGN KEY (공고번호) REFERENCES bid_announcement(공고번호),
    FOREIGN KEY (사업자번호) REFERENCES contractor(사업자번호)
);

CREATE TABLE IF NOT EXISTS evaluation (
    평가번호        INTEGER PRIMARY KEY AUTOINCREMENT,
    공고번호        TEXT NOT NULL,
    사업자번호      TEXT NOT NULL,
    기술점수        REAL,
    가격점수        REAL,
    총점            REAL,
    순위            INTEGER,
    적격여부        TEXT,
    FOREIGN KEY (공고번호) REFERENCES bid_announcement(공고번호),
    FOREIGN KEY (사업자번호) REFERENCES contractor(사업자번호)
);
"""

SEED_DATA_SQL = """
INSERT INTO agency VALUES ('A001','국토교통부','중앙부처',NULL,'서울',1948);
INSERT INTO agency VALUES ('A002','한국도로공사','공기업','A001','경기',1969);
INSERT INTO agency VALUES ('A003','서울특별시','지방자치단체',NULL,'서울',1946);
INSERT INTO agency VALUES ('A004','한국전력공사','공기업',NULL,'전남',1961);
INSERT INTO agency VALUES ('A005','조달청','중앙부처',NULL,'대전',1949);
INSERT INTO agency VALUES ('A006','부산광역시','지방자치단체',NULL,'부산',1963);

INSERT INTO contractor VALUES ('B001','삼성SDS','홍길동','IT서비스','서울',1985,5000000000);
INSERT INTO contractor VALUES ('B002','LG CNS','김철수','IT서비스','서울',1987,3000000000);
INSERT INTO contractor VALUES ('B003','현대건설','박영희','건설','서울',1947,8000000000);
INSERT INTO contractor VALUES ('B004','대우건설','이민호','건설','서울',1966,4000000000);
INSERT INTO contractor VALUES ('B005','KT','정수연','통신','경기',1981,10000000000);
INSERT INTO contractor VALUES ('B006','네이버클라우드','최동원','클라우드','경기',2017,500000000);
INSERT INTO contractor VALUES ('B007','카카오엔터프라이즈','한소희','AI','경기',2020,300000000);
INSERT INTO contractor VALUES ('B008','한화시스템','강민수','방산/IT','서울',1978,2000000000);

INSERT INTO bid_announcement VALUES ('G2025-001','AI 기반 도로 안전관리 시스템 구축','한국도로공사','2025-01-05','2025-02-05',5000000000,'용역','제한경쟁','IT서비스','전국');
INSERT INTO bid_announcement VALUES ('G2025-002','스마트시티 통합플랫폼 구축','서울특별시','2025-01-10','2025-02-10',8000000000,'용역','일반경쟁','IT서비스','서울');
INSERT INTO bid_announcement VALUES ('G2025-003','전력망 AI 예측 시스템','한국전력공사','2025-01-15','2025-02-15',3000000000,'용역','제한경쟁','AI','전국');
INSERT INTO bid_announcement VALUES ('G2025-004','국도 4차선 확장공사','국토교통부','2025-01-20','2025-02-20',25000000000,'공사','일반경쟁','건설','경기');
INSERT INTO bid_announcement VALUES ('G2025-005','클라우드 전환 사업','조달청','2025-02-01','2025-03-01',2000000000,'용역','제한경쟁','클라우드','전국');
INSERT INTO bid_announcement VALUES ('G2025-006','부산 도시철도 신호제어 시스템','부산광역시','2025-02-05','2025-03-05',4500000000,'용역','제한경쟁','IT서비스','부산');
INSERT INTO bid_announcement VALUES ('G2025-007','공공데이터 개방 포털 고도화','조달청','2025-02-10','2025-03-10',1500000000,'용역','일반경쟁','IT서비스','전국');
INSERT INTO bid_announcement VALUES ('G2025-008','세종시 신청사 건축','국토교통부','2025-02-15','2025-03-15',35000000000,'공사','일반경쟁','건설','세종');

INSERT INTO bid_detail VALUES ('G2025-001',1,'AI 모델 개발',1,'식',2000000000,'딥러닝 기반');
INSERT INTO bid_detail VALUES ('G2025-001',2,'CCTV 영상분석 서버',10,'대',100000000,'GPU 탑재');
INSERT INTO bid_detail VALUES ('G2025-001',3,'통합 대시보드',1,'식',500000000,'웹 기반');
INSERT INTO bid_detail VALUES ('G2025-002',1,'플랫폼 SW 개발',1,'식',4000000000,'MSA 아키텍처');
INSERT INTO bid_detail VALUES ('G2025-002',2,'IoT 센서 연동',500,'개',3000000,'LoRaWAN');
INSERT INTO bid_detail VALUES ('G2025-003',1,'예측 엔진 개발',1,'식',1500000000,'Transformer 기반');
INSERT INTO bid_detail VALUES ('G2025-003',2,'데이터 수집 시스템',1,'식',800000000,'실시간 스트리밍');
INSERT INTO bid_detail VALUES ('G2025-004',1,'토공사',1,'식',10000000000,'L=12.5km');
INSERT INTO bid_detail VALUES ('G2025-004',2,'포장공사',1,'식',8000000000,'아스팔트');
INSERT INTO bid_detail VALUES ('G2025-005',1,'클라우드 이전',1,'식',1200000000,'IaaS/PaaS');
INSERT INTO bid_detail VALUES ('G2025-006',1,'신호제어 시스템',1,'식',3000000000,'CBTC');
INSERT INTO bid_detail VALUES ('G2025-007',1,'포털 개발',1,'식',900000000,'React+Spring');
INSERT INTO bid_detail VALUES ('G2025-008',1,'건축공사',1,'식',20000000000,'지상20층');

INSERT INTO contract VALUES ('C001','G2025-001','B001',4800000000,'2025-02-10','2025-12-31','진행중');
INSERT INTO contract VALUES ('C002','G2025-002','B002',7600000000,'2025-02-15','2026-06-30','진행중');
INSERT INTO contract VALUES ('C003','G2025-003','B005',2900000000,'2025-02-20','2025-10-31','진행중');
INSERT INTO contract VALUES ('C004','G2025-004','B003',24000000000,'2025-02-25','2027-02-28','진행중');
INSERT INTO contract VALUES ('C005','G2025-005','B006',1900000000,'2025-03-05','2025-09-30','진행중');
INSERT INTO contract VALUES ('C006','G2025-007','B007',1400000000,'2025-03-15','2025-11-30','진행중');

INSERT INTO evaluation VALUES (1,'G2025-001','B001',85.5,90.0,87.3,1,'적격');
INSERT INTO evaluation VALUES (2,'G2025-001','B002',80.0,88.0,83.2,2,'적격');
INSERT INTO evaluation VALUES (3,'G2025-001','B008',75.0,85.0,79.0,3,'적격');
INSERT INTO evaluation VALUES (4,'G2025-002','B002',90.0,85.0,88.0,1,'적격');
INSERT INTO evaluation VALUES (5,'G2025-002','B001',88.0,82.0,85.6,2,'적격');
INSERT INTO evaluation VALUES (6,'G2025-003','B005',92.0,88.0,90.4,1,'적격');
INSERT INTO evaluation VALUES (7,'G2025-003','B006',78.0,90.0,83.0,2,'적격');
INSERT INTO evaluation VALUES (8,'G2025-004','B003',88.0,95.0,90.8,1,'적격');
INSERT INTO evaluation VALUES (9,'G2025-004','B004',82.0,92.0,86.0,2,'적격');
INSERT INTO evaluation VALUES (10,'G2025-005','B006',90.0,85.0,88.0,1,'적격');
INSERT INTO evaluation VALUES (11,'G2025-005','B005',85.0,80.0,83.0,2,'적격');
INSERT INTO evaluation VALUES (12,'G2025-006','B008',86.0,88.0,86.8,1,'적격');
INSERT INTO evaluation VALUES (13,'G2025-007','B007',88.0,92.0,89.6,1,'적격');
INSERT INTO evaluation VALUES (14,'G2025-007','B006',82.0,88.0,84.4,2,'적격');
INSERT INTO evaluation VALUES (15,'G2025-008','B003',90.0,93.0,91.2,1,'적격');
INSERT INTO evaluation VALUES (16,'G2025-008','B004',85.0,90.0,87.0,2,'적격');
"""

BENCHMARK = [
    {"id":"P01","difficulty":"E","nl":"전체 입찰 공고 건수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM bid_announcement"},
    {"id":"P02","difficulty":"E","nl":"추정가격이 50억 이상인 공고명을 조회하시오.",
     "gold_sql":"SELECT 공고명 FROM bid_announcement WHERE 추정가격>=5000000000"},
    {"id":"P03","difficulty":"E","nl":"IT서비스 업종 업체의 수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM contractor WHERE 업종='IT서비스'"},
    {"id":"P04","difficulty":"E","nl":"제한경쟁 입찰 건수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM bid_announcement WHERE 입찰방식='제한경쟁'"},
    {"id":"P05","difficulty":"E","nl":"서울 소재 업체명을 모두 조회하시오.",
     "gold_sql":"SELECT 업체명 FROM contractor WHERE 소재지='서울'"},
    {"id":"P06","difficulty":"E","nl":"공기업 기관의 이름을 조회하시오.",
     "gold_sql":"SELECT 기관명 FROM agency WHERE 기관유형='공기업'"},
    {"id":"P07","difficulty":"E","nl":"진행중인 계약 건수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM contract WHERE 계약상태='진행중'"},
    {"id":"P08","difficulty":"E","nl":"기술점수가 90점 이상인 평가 결과를 조회하시오.",
     "gold_sql":"SELECT * FROM evaluation WHERE 기술점수>=90"},
    {"id":"P09","difficulty":"M","nl":"발주기관별 공고 건수를 구하시오.",
     "gold_sql":"SELECT 발주기관명, COUNT(*) AS 공고수 FROM bid_announcement GROUP BY 발주기관명"},
    {"id":"P10","difficulty":"M","nl":"업체별 낙찰(1순위) 횟수를 구하시오.",
     "gold_sql":"SELECT 사업자번호, COUNT(*) AS 낙찰수 FROM evaluation WHERE 순위=1 GROUP BY 사업자번호"},
    {"id":"P11","difficulty":"M","nl":"계약금액 합계가 가장 큰 업체의 이름과 합계를 구하시오.",
     "gold_sql":"SELECT c2.업체명, SUM(c.계약금액) AS 합계 FROM contract c JOIN contractor c2 ON c.사업자번호=c2.사업자번호 GROUP BY c2.업체명 ORDER BY 합계 DESC LIMIT 1"},
    {"id":"P12","difficulty":"M","nl":"입찰방식별 평균 추정가격을 구하시오.",
     "gold_sql":"SELECT 입찰방식, AVG(추정가격) AS 평균추정가격 FROM bid_announcement GROUP BY 입찰방식"},
    {"id":"P13","difficulty":"M","nl":"공고별 세부항목 수를 구하시오.",
     "gold_sql":"SELECT 공고번호, COUNT(*) AS 항목수 FROM bid_detail GROUP BY 공고번호"},
    {"id":"P14","difficulty":"M","nl":"2025년 2월 이후에 공고된 건의 공고명과 발주기관을 조회하시오.",
     "gold_sql":"SELECT 공고명, 발주기관명 FROM bid_announcement WHERE 공고일자>='2025-02-01'"},
    {"id":"P15","difficulty":"M","nl":"계약 체결된 공고의 공고명, 계약업체명, 계약금액을 조회하시오.",
     "gold_sql":"SELECT a.공고명, c2.업체명, c.계약금액 FROM contract c JOIN bid_announcement a ON c.공고번호=a.공고번호 JOIN contractor c2 ON c.사업자번호=c2.사업자번호"},
    {"id":"P16","difficulty":"H","nl":"추정가격 대비 계약금액의 낙찰률(%)을 구하시오.",
     "gold_sql":"SELECT a.공고명, a.추정가격, c.계약금액, ROUND(c.계약금액*100.0/a.추정가격,2) AS 낙찰률 FROM contract c JOIN bid_announcement a ON c.공고번호=a.공고번호"},
    {"id":"P17","difficulty":"H","nl":"2건 이상 입찰 참여한 업체의 이름과 평균 총점을 구하시오.",
     "gold_sql":"SELECT c2.업체명, AVG(e.총점) AS 평균총점, COUNT(*) AS 참여수 FROM evaluation e JOIN contractor c2 ON e.사업자번호=c2.사업자번호 GROUP BY c2.업체명 HAVING COUNT(*)>=2"},
    {"id":"P18","difficulty":"H","nl":"기관유형별 총 계약금액을 구하시오.",
     "gold_sql":"SELECT ag.기관유형, SUM(c.계약금액) AS 총계약금액 FROM contract c JOIN bid_announcement a ON c.공고번호=a.공고번호 JOIN agency ag ON a.발주기관명=ag.기관명 GROUP BY ag.기관유형"},
    {"id":"P19","difficulty":"H","nl":"각 공고의 1순위 업체명, 총점, 계약금액을 함께 조회하시오.",
     "gold_sql":"SELECT a.공고명, c2.업체명, e.총점, ct.계약금액 FROM evaluation e JOIN bid_announcement a ON e.공고번호=a.공고번호 JOIN contractor c2 ON e.사업자번호=c2.사업자번호 LEFT JOIN contract ct ON e.공고번호=ct.공고번호 AND e.사업자번호=ct.사업자번호 WHERE e.순위=1"},
    {"id":"P20","difficulty":"H","nl":"업종별 평균 자본금과 평균 낙찰 총점을 구하시오.",
     "gold_sql":"SELECT c2.업종, AVG(c2.자본금) AS 평균자본금, AVG(e.총점) AS 평균총점 FROM contractor c2 JOIN evaluation e ON c2.사업자번호=e.사업자번호 GROUP BY c2.업종"},
    {"id":"P21","difficulty":"E","nl":"용역 공고의 수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM bid_announcement WHERE 공고유형='용역'"},
    {"id":"P22","difficulty":"M","nl":"세부항목 중 예정단가가 10억 이상인 품목명과 공고번호를 조회하시오.",
     "gold_sql":"SELECT 공고번호, 품목명, 예정단가 FROM bid_detail WHERE 예정단가>=1000000000"},
    {"id":"P23","difficulty":"H","nl":"공고별 평가 참여 업체 수와 최고 총점, 최저 총점을 구하시오.",
     "gold_sql":"SELECT 공고번호, COUNT(*) AS 참여수, MAX(총점) AS 최고점, MIN(총점) AS 최저점 FROM evaluation GROUP BY 공고번호"},
    {"id":"P24","difficulty":"H","nl":"낙찰률이 95% 이상인 계약의 공고명, 업체명, 낙찰률을 구하시오.",
     "gold_sql":"SELECT a.공고명, c2.업체명, ROUND(c.계약금액*100.0/a.추정가격,2) AS 낙찰률 FROM contract c JOIN bid_announcement a ON c.공고번호=a.공고번호 JOIN contractor c2 ON c.사업자번호=c2.사업자번호 WHERE c.계약금액*100.0/a.추정가격>=95"},
    {"id":"P25","difficulty":"M","nl":"지역별 공고 건수와 총 추정가격을 구하시오.",
     "gold_sql":"SELECT 지역제한, COUNT(*) AS 공고수, SUM(추정가격) AS 총추정가격 FROM bid_announcement GROUP BY 지역제한"},
]

PROMPT_B1 = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.
Output ONLY the SQL query, nothing else.

Question: {question}

SQL:"""

PROMPT_B2 = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.
You have access to the following database documentation:

---
{retrieved_chunks}
---

Output ONLY the SQL query, nothing else.

Question: {question}

SQL:"""

def get_b2_chunks(q):
    kw = {
        "공고": "bid_announcement 테이블: 공고번호(TEXT PK), 공고명(TEXT), 발주기관명(TEXT), 공고일자(TEXT), 마감일자(TEXT), 추정가격(INTEGER), 공고유형(TEXT), 입찰방식(TEXT), 업종제한(TEXT), 지역제한(TEXT)",
        "입찰": "bid_announcement 테이블: 공고번호(TEXT PK), 공고명(TEXT), 발주기관명(TEXT), 공고일자(TEXT), 마감일자(TEXT), 추정가격(INTEGER), 공고유형(TEXT), 입찰방식(TEXT), 업종제한(TEXT), 지역제한(TEXT)",
        "세부": "bid_detail 테이블: 공고번호(TEXT FK), 세부항목번호(INTEGER), 품목명(TEXT), 수량(INTEGER), 단위(TEXT), 예정단가(INTEGER), 규격(TEXT)",
        "품목": "bid_detail 테이블: 공고번호(TEXT FK), 세부항목번호(INTEGER), 품목명(TEXT), 수량(INTEGER), 단위(TEXT), 예정단가(INTEGER), 규격(TEXT)",
        "기관": "agency 테이블: 기관코드(TEXT PK), 기관명(TEXT), 기관유형(TEXT), 상위기관코드(TEXT), 지역(TEXT), 설립연도(INTEGER)",
        "업체": "contractor 테이블: 사업자번호(TEXT PK), 업체명(TEXT), 대표자명(TEXT), 업종(TEXT), 소재지(TEXT), 설립연도(INTEGER), 자본금(INTEGER)",
        "계약": "contract 테이블: 계약번호(TEXT PK), 공고번호(TEXT FK), 사업자번호(TEXT FK), 계약금액(INTEGER), 계약일자(TEXT), 납품기한(TEXT), 계약상태(TEXT)",
        "평가": "evaluation 테이블: 평가번호(INTEGER PK), 공고번호(TEXT FK), 사업자번호(TEXT FK), 기술점수(REAL), 가격점수(REAL), 총점(REAL), 순위(INTEGER), 적격여부(TEXT)",
        "낙찰": "evaluation.순위=1이면 낙찰. contract 테이블과 연결.",
        "자본금": "contractor.자본금: 원(KRW) 단위",
    }
    m = set()
    for k,v in kw.items():
        if k in q: m.add(v)
    if not m:
        m.add(kw["공고"]); m.add(kw["업체"])
    return "\n".join(m)

PROMPT_DKAP = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.

## Database Schema (Korean Public Procurement System - 나라장터)

### Table: bid_announcement (입찰공고)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 공고번호 | TEXT PK | 공고 고유 번호 (G20XX-NNN) | identifier |
| 공고명 | TEXT NOT NULL | 사업/공사 명칭 | text |
| 발주기관명 | TEXT NOT NULL | 발주 기관 이름 (agency.기관명과 매칭) | text |
| 공고일자 | TEXT NOT NULL | 공고 게시일 (YYYY-MM-DD) | date |
| 마감일자 | TEXT NOT NULL | 입찰 마감일 (YYYY-MM-DD) | date |
| 추정가격 | INTEGER | 추정 사업 금액 (원, KRW) | financial_amount |
| 공고유형 | TEXT | 용역/공사/물품 | category |
| 입찰방식 | TEXT | 일반경쟁/제한경쟁/지명경쟁 | category |
| 업종제한 | TEXT | 참여 업종 제한 | category |
| 지역제한 | TEXT | 참여 지역 제한 (전국/서울/경기 등) | category |

### Table: bid_detail (공고 세부항목)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 공고번호 | TEXT FK→bid_announcement | 공고 번호 | identifier |
| 세부항목번호 | INTEGER | 항목 순번 | count |
| 품목명 | TEXT | 세부 품목/작업 명칭 | text |
| 수량 | INTEGER | 수량 | count |
| 단위 | TEXT | 단위 (식/대/개 등) | unit |
| 예정단가 | INTEGER | 예정 단가 (원) | financial_amount |
| 규격 | TEXT | 규격/사양 | text |
| PK: (공고번호, 세부항목번호) | | |

### Table: agency (발주기관)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 기관코드 | TEXT PK | 기관 고유 코드 | identifier |
| 기관명 | TEXT NOT NULL | 기관 이름 | text |
| 기관유형 | TEXT | 중앙부처/공기업/지방자치단체 | category |
| 상위기관코드 | TEXT | 상위 기관 (NULL이면 최상위) | identifier |
| 지역 | TEXT | 소재 지역 | category |
| 설립연도 | INTEGER | 설립 연도 | date |

### Table: contractor (입찰참여업체)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 사업자번호 | TEXT PK | 사업자등록번호 | identifier |
| 업체명 | TEXT NOT NULL | 업체 상호 | text |
| 대표자명 | TEXT | 대표자 성명 | text |
| 업종 | TEXT | 주 업종 (IT서비스/건설/통신/AI/클라우드/방산IT) | category |
| 소재지 | TEXT | 본사 소재지 | category |
| 설립연도 | INTEGER | 설립 연도 | date |
| 자본금 | INTEGER | 자본금 (원, KRW) | financial_amount |

### Table: contract (계약)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 계약번호 | TEXT PK | 계약 고유 번호 | identifier |
| 공고번호 | TEXT FK→bid_announcement | 원 공고 번호 | identifier |
| 사업자번호 | TEXT FK→contractor | 계약 업체 | identifier |
| 계약금액 | INTEGER | 계약 금액 (원, KRW) | financial_amount |
| 계약일자 | TEXT | 계약 체결일 | date |
| 납품기한 | TEXT | 납품/완료 기한 | date |
| 계약상태 | TEXT | 진행중/완료/해지 | category |

### Table: evaluation (입찰평가)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| 평가번호 | INTEGER PK AUTOINCREMENT | 평가 고유 번호 | identifier |
| 공고번호 | TEXT FK→bid_announcement | 공고 번호 | identifier |
| 사업자번호 | TEXT FK→contractor | 업체 번호 | identifier |
| 기술점수 | REAL | 기술 평가 점수 (0~100) | measurement |
| 가격점수 | REAL | 가격 평가 점수 (0~100) | measurement |
| 총점 | REAL | 가중 합산 점수 | measurement |
| 순위 | INTEGER | 평가 순위 (1=낙찰) | count |
| 적격여부 | TEXT | 적격/부적격 | category |

## Domain Glossary (공공조달 용어)
- 추정가격: 사업 예산 규모 (원 단위). 계약금액과 다름 (낙찰률 = 계약금액/추정가격 × 100).
- 낙찰: evaluation.순위=1인 업체가 낙찰자. contract에 기록됨.
- 낙찰률: 계약금액 ÷ 추정가격 × 100 (%). 보통 90~98%.
- 입찰방식: 일반경쟁(모든 업체 참여), 제한경쟁(업종/지역 제한), 지명경쟁(기관 지명).
- 공고유형: 용역(서비스/SW), 공사(건설/토목), 물품(구매).
- 기관유형: 중앙부처(정부 부처), 공기업(정부 출자), 지방자치단체(시/도/군).

## Join Rules
- bid_announcement ↔ bid_detail: ON 공고번호
- bid_announcement ↔ contract: ON 공고번호
- bid_announcement ↔ evaluation: ON 공고번호
- bid_announcement ↔ agency: ON 발주기관명 = 기관명
- contract ↔ contractor: ON 사업자번호
- evaluation ↔ contractor: ON 사업자번호

Output ONLY the SQL query, nothing else.

Question: {question}

SQL:"""

def call_vllm(prompt, max_retries=3):
    headers = {"Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
               "stop": ["Question:", "---"], "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{VLLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL).strip()
            content = re.sub(r'^```sql\s*', '', content)
            content = re.sub(r'^```\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = ' '.join(content.split()).rstrip(';').strip()
            return content
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1: time.sleep(2**attempt)
    return ""

def setup_database(db_path):
    if db_path.exists(): db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_DDL)
    conn.executescript(SEED_DATA_SQL)
    conn.commit()
    return conn

def execute_sql(conn, sql):
    try:
        cursor = conn.execute(sql)
        return True, cursor.fetchall()
    except: return False, None

def evaluate_ex(conn, pred, gold):
    po, pr = execute_sql(conn, pred)
    go, gr = execute_sql(conn, gold)
    if not po or not go: return False
    if pr is None or gr is None: return False
    if "order by" in gold.lower(): return pr == gr
    return set(map(tuple, pr)) == set(map(tuple, gr))

def evaluate_em(p, g):
    n = lambda s: re.sub(r'\s+', ' ', s.lower().strip()).rstrip(';').strip()
    return n(p) == n(g)

@dataclass
class Result:
    condition: str; run_id: int; question_id: str; difficulty: str
    nl_question: str; gold_sql: str; pred_sql: str
    ex_score: bool; em_score: bool; execution_ok: bool; latency_ms: float

def run_experiment():
    logger.info("="*60)
    logger.info("DKAP Public Procurement Experiment")
    logger.info("="*60)
    conn = setup_database(DB_PATH)
    for t in ['bid_announcement','bid_detail','agency','contractor','contract','evaluation']:
        c = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        logger.info(f"  {t}: {c}")
    test = call_vllm("SELECT 1")
    if not test: sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = []
    conditions = {"B1": lambda q: PROMPT_B1.format(question=q),
        "B2": lambda q: PROMPT_B2.format(question=q, retrieved_chunks=get_b2_chunks(q)),
        "DKAP": lambda q: PROMPT_DKAP.format(question=q)}
    total = len(BENCHMARK)*len(conditions)*NUM_RUNS
    progress = 0
    for run_id in range(1, NUM_RUNS+1):
        logger.info(f"\n--- Run {run_id}/{NUM_RUNS} ---")
        for cn, pf in conditions.items():
            logger.info(f"  {cn}")
            for item in BENCHMARK:
                progress += 1
                t0 = time.time()
                pred = call_vllm(pf(item["nl"]))
                lat = (time.time()-t0)*1000
                eo, _ = execute_sql(conn, pred)
                ex = evaluate_ex(conn, pred, item["gold_sql"]) if eo else False
                em = evaluate_em(pred, item["gold_sql"])
                all_results.append(Result(cn,run_id,item["id"],item["difficulty"],item["nl"],item["gold_sql"],pred,ex,em,eo,lat))
                st = "EX" if ex else ("EXEC" if eo else "FAIL")
                logger.info(f"    [{progress}/{total}] {item['id']} ({item['difficulty']}) -> {st} ({lat:.0f}ms)")
    conn.close()
    with open(RESULTS_DIR/"raw_results.json","w",encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results],f,ensure_ascii=False,indent=2)
    summary = {}
    for cond in ["B1","B2","DKAP"]:
        cr = [r for r in all_results if r.condition==cond]
        n=len(cr); ext=sum(1 for r in cr if r.ex_score); emt=sum(1 for r in cr if r.em_score)
        exect=sum(1 for r in cr if r.execution_ok); al=sum(r.latency_ms for r in cr)/n if n else 0
        db={}
        for d in ["E","M","H"]:
            dr=[r for r in cr if r.difficulty==d]; dn=len(dr)
            if dn: db[d]={"n":dn,"EX":sum(1 for r in dr if r.ex_score),"EX_pct":round(sum(1 for r in dr if r.ex_score)*100/dn,1),"EM":sum(1 for r in dr if r.em_score),"EM_pct":round(sum(1 for r in dr if r.em_score)*100/dn,1)}
        re2={}
        for rid in range(1,NUM_RUNS+1):
            rr=[r for r in cr if r.run_id==rid]; rn=len(rr)
            re2[f"run_{rid}"]=round(sum(1 for r in rr if r.ex_score)*100/rn,1) if rn else 0
        summary[cond]={"n":n,"EX_total":ext,"EX_pct":round(ext*100/n,1) if n else 0,"EM_total":emt,"EM_pct":round(emt*100/n,1) if n else 0,"EXEC_total":exect,"EXEC_pct":round(exect*100/n,1) if n else 0,"avg_latency_ms":round(al,1),"by_difficulty":db,"by_run":re2}
    logger.info(f"{'Cond':<8} {'EX%':<8} {'EM%':<8} {'EXEC%':<8}")
    for c in ["B1","B2","DKAP"]:
        s=summary[c]; logger.info(f"{c:<8} {s['EX_pct']:<8} {s['EM_pct']:<8} {s['EXEC_pct']:<8}")
    for c in ["B1","B2","DKAP"]:
        logger.info(f"  {c}:")
        for d in ["E","M","H"]:
            dd=summary[c]["by_difficulty"].get(d,{})
            if dd: logger.info(f"    {d}: EX={dd['EX_pct']}% ({dd['EX']}/{dd['n']})")
    with open(RESULTS_DIR/"summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    logger.info("DONE")
    return summary

if __name__=="__main__": run_experiment()
