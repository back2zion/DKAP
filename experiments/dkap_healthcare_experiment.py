#!/usr/bin/env python3
"""
DKAP Healthcare Domain Experiment (Preliminary)
================================================
Clinical QA over OMOP CDM-like schema.
Three conditions: B1 (vanilla) / B2 (flat-chunk RAG) / DKAP (structured L2).
Model: Qwen3-32B-AWQ via vLLM (localhost:8000)

Author: Dooil Kwak (automated experiment runner)
"""

import json, time, sqlite3, os, sys, re, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

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
RESULTS_DIR = Path("/home/aigen/dkap_healthcare_results")
DB_PATH = Path("/tmp/healthcare_benchmark.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dkap_healthcare_experiment.log")])
logger = logging.getLogger(__name__)

# ============================================================
# 1. OMOP CDM-inspired Clinical Data Schema
# ============================================================
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS patient (
    patient_id          INTEGER PRIMARY KEY,
    gender              TEXT NOT NULL,        -- M/F
    birth_year          INTEGER,
    age                 INTEGER,
    blood_type          TEXT,                 -- A/B/AB/O
    insurance_type      TEXT                  -- 건강보험/의료급여/산재/자동차
);

CREATE TABLE IF NOT EXISTS visit_occurrence (
    visit_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    visit_type          TEXT NOT NULL,        -- 외래/입원/응급
    visit_start_date    TEXT NOT NULL,        -- YYYY-MM-DD
    visit_end_date      TEXT,
    department          TEXT,                 -- 내과/외과/정형외과/소아과/산부인과/신경과/정신건강의학과
    attending_doctor_id INTEGER,
    FOREIGN KEY (patient_id) REFERENCES patient(patient_id)
);

CREATE TABLE IF NOT EXISTS condition_occurrence (
    condition_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    visit_id            INTEGER,
    condition_concept   TEXT NOT NULL,        -- KCD-8 code description
    icd10_code          TEXT,                 -- ICD-10 code
    condition_start_date TEXT NOT NULL,
    condition_end_date  TEXT,
    condition_status    TEXT DEFAULT '활성',  -- 활성/완치/관해
    FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
    FOREIGN KEY (visit_id) REFERENCES visit_occurrence(visit_id)
);

CREATE TABLE IF NOT EXISTS drug_exposure (
    drug_exposure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    visit_id            INTEGER,
    drug_name           TEXT NOT NULL,        -- Generic name (Korean)
    drug_concept_id     INTEGER,             -- OMOP concept ID
    dose_value          REAL,
    dose_unit           TEXT,                -- mg/ml/mcg
    route               TEXT,                -- 경구/정맥주사/피하주사/외용
    days_supply         INTEGER,
    prescribe_date      TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
    FOREIGN KEY (visit_id) REFERENCES visit_occurrence(visit_id)
);

CREATE TABLE IF NOT EXISTS measurement (
    measurement_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    visit_id            INTEGER,
    measurement_name    TEXT NOT NULL,        -- 검사명
    value_as_number     REAL,
    unit                TEXT,
    range_low           REAL,
    range_high          REAL,
    abnormal_flag       TEXT,                -- 정상/높음/낮음/위험
    measurement_date    TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patient(patient_id)
);

CREATE TABLE IF NOT EXISTS procedure_occurrence (
    procedure_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    visit_id            INTEGER,
    procedure_name      TEXT NOT NULL,
    procedure_date      TEXT NOT NULL,
    cost                INTEGER,             -- 원 (KRW)
    FOREIGN KEY (patient_id) REFERENCES patient(patient_id)
);

CREATE TABLE IF NOT EXISTS doctor (
    doctor_id           INTEGER PRIMARY KEY,
    doctor_name         TEXT,
    specialty           TEXT,                -- 전문과목
    license_year        INTEGER
);
"""

SEED_DATA_SQL = """
-- Patients
INSERT INTO patient VALUES (1,'M',1965,60,'A','건강보험');
INSERT INTO patient VALUES (2,'F',1978,47,'B','건강보험');
INSERT INTO patient VALUES (3,'M',1950,75,'O','의료급여');
INSERT INTO patient VALUES (4,'F',1988,37,'AB','건강보험');
INSERT INTO patient VALUES (5,'M',1972,53,'A','건강보험');
INSERT INTO patient VALUES (6,'F',1945,80,'O','의료급여');
INSERT INTO patient VALUES (7,'M',1995,30,'B','건강보험');
INSERT INTO patient VALUES (8,'F',1960,65,'A','건강보험');
INSERT INTO patient VALUES (9,'M',1982,43,'AB','산재');
INSERT INTO patient VALUES (10,'F',1970,55,'O','건강보험');

-- Doctors
INSERT INTO doctor VALUES (101,'김의사','내과',2005);
INSERT INTO doctor VALUES (102,'이의사','외과',2010);
INSERT INTO doctor VALUES (103,'박의사','정형외과',2008);
INSERT INTO doctor VALUES (104,'최의사','소아과',2012);
INSERT INTO doctor VALUES (105,'정의사','신경과',2003);

-- Visits
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (1,'외래','2025-01-10',NULL,'내과',101);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (1,'입원','2025-01-15','2025-01-22','내과',101);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (2,'외래','2025-01-08',NULL,'외과',102);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (3,'응급','2025-01-20','2025-01-25','신경과',105);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (3,'입원','2025-01-25','2025-02-10','신경과',105);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (4,'외래','2025-02-01',NULL,'소아과',104);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (5,'외래','2025-01-12',NULL,'내과',101);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (5,'입원','2025-02-05','2025-02-12','내과',101);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (6,'입원','2025-01-05','2025-01-30','내과',101);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (7,'외래','2025-02-10',NULL,'정형외과',103);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (8,'외래','2025-01-18',NULL,'내과',101);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (9,'응급','2025-02-15','2025-02-16','외과',102);
INSERT INTO visit_occurrence (patient_id,visit_type,visit_start_date,visit_end_date,department,attending_doctor_id) VALUES (10,'외래','2025-01-25',NULL,'내과',101);

-- Conditions
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (1,1,'제2형 당뇨병','E11','2025-01-10',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (1,2,'급성 심근경색','I21','2025-01-15','2025-01-22','완치');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (2,3,'유방암','C50','2025-01-08',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (3,4,'뇌경색','I63','2025-01-20',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (3,5,'고혈압','I10','2025-01-25',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (4,6,'천식','J45','2025-02-01',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (5,7,'만성 신장병','N18','2025-01-12',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (5,8,'제2형 당뇨병','E11','2025-02-05',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (6,9,'폐렴','J18','2025-01-05','2025-01-30','완치');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (6,9,'고혈압','I10','2025-01-05',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (7,10,'요추 추간판 탈출증','M51','2025-02-10',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (8,11,'고지혈증','E78','2025-01-18',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (8,11,'제2형 당뇨병','E11','2025-01-18',NULL,'활성');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (9,12,'복부 외상','S36','2025-02-15','2025-02-16','완치');
INSERT INTO condition_occurrence (patient_id,visit_id,condition_concept,icd10_code,condition_start_date,condition_end_date,condition_status) VALUES (10,13,'갑상선기능저하증','E03','2025-01-25',NULL,'활성');

-- Drug Exposures
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (1,1,'메트포르민',1503297,500,'mg','경구',30,'2025-01-10');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (1,2,'아스피린',1112807,100,'mg','경구',90,'2025-01-15');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (1,2,'헤파린',1367571,5000,'IU','정맥주사',7,'2025-01-15');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (2,3,'타목시펜',1436678,20,'mg','경구',180,'2025-01-08');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (3,4,'알테플라제',19054825,0.9,'mg/kg','정맥주사',1,'2025-01-20');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (3,5,'암로디핀',1332418,5,'mg','경구',30,'2025-01-25');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (3,5,'클로피도그렐',1322184,75,'mg','경구',30,'2025-01-25');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (4,6,'살부타몰',1149196,100,'mcg','흡입',30,'2025-02-01');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (5,7,'에리스로포이에틴',1301125,4000,'IU','피하주사',30,'2025-01-12');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (5,8,'인슐린 글라진',1596977,20,'IU','피하주사',30,'2025-02-05');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (6,9,'세프트리악손',1777087,2,'g','정맥주사',14,'2025-01-05');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (6,9,'발사르탄',1308216,80,'mg','경구',30,'2025-01-05');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (8,11,'아토르바스타틴',1545958,20,'mg','경구',30,'2025-01-18');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (8,11,'메트포르민',1503297,1000,'mg','경구',30,'2025-01-18');
INSERT INTO drug_exposure (patient_id,visit_id,drug_name,drug_concept_id,dose_value,dose_unit,route,days_supply,prescribe_date) VALUES (10,13,'레보티록신',19049024,50,'mcg','경구',30,'2025-01-25');

-- Measurements
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (1,1,'HbA1c',8.2,'%',4.0,5.6,'높음','2025-01-10');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (1,1,'공복혈당',156,'mg/dL',70,100,'높음','2025-01-10');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (1,2,'트로포닌I',2.5,'ng/mL',0,0.04,'위험','2025-01-15');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (1,2,'CK-MB',45,'ng/mL',0,5,'위험','2025-01-15');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (3,4,'수축기혈압',185,'mmHg',90,120,'높음','2025-01-20');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (3,4,'이완기혈압',110,'mmHg',60,80,'높음','2025-01-20');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (5,7,'크레아티닌',3.8,'mg/dL',0.7,1.3,'위험','2025-01-12');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (5,7,'eGFR',18,'mL/min',90,120,'위험','2025-01-12');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (5,8,'HbA1c',9.1,'%',4.0,5.6,'높음','2025-02-05');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (6,9,'WBC',15200,'cells/uL',4000,11000,'높음','2025-01-05');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (6,9,'CRP',85,'mg/L',0,5,'위험','2025-01-05');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (8,11,'LDL콜레스테롤',185,'mg/dL',0,130,'높음','2025-01-18');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (8,11,'총콜레스테롤',275,'mg/dL',0,200,'높음','2025-01-18');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (8,11,'HbA1c',7.5,'%',4.0,5.6,'높음','2025-01-18');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (10,13,'TSH',12.5,'mIU/L',0.4,4.0,'높음','2025-01-25');
INSERT INTO measurement (patient_id,visit_id,measurement_name,value_as_number,unit,range_low,range_high,abnormal_flag,measurement_date) VALUES (10,13,'Free T4',0.5,'ng/dL',0.8,1.8,'낮음','2025-01-25');

-- Procedures
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (1,2,'관상동맥 스텐트 삽입술','2025-01-16',8500000);
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (2,3,'유방 조직검사','2025-01-08',350000);
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (3,4,'뇌 CT','2025-01-20',280000);
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (3,4,'뇌 MRI','2025-01-21',650000);
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (5,7,'투석','2025-01-12',450000);
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (6,9,'흉부 X-ray','2025-01-05',35000);
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (7,10,'요추 MRI','2025-02-10',550000);
INSERT INTO procedure_occurrence (patient_id,visit_id,procedure_name,procedure_date,cost) VALUES (9,12,'복부 CT','2025-02-15',320000);
"""

# ============================================================
# 2. Benchmark Questions (40 Clinical NL→SQL pairs)
# ============================================================
BENCHMARK = [
    # Easy
    {"id":"H01","difficulty":"E","nl":"전체 환자 수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM patient"},
    {"id":"H02","difficulty":"E","nl":"남성 환자 수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM patient WHERE gender='M'"},
    {"id":"H03","difficulty":"E","nl":"의료급여 환자의 이름(patient_id)과 나이를 조회하시오.",
     "gold_sql":"SELECT patient_id, age FROM patient WHERE insurance_type='의료급여'"},
    {"id":"H04","difficulty":"E","nl":"입원(입원) 방문 건수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM visit_occurrence WHERE visit_type='입원'"},
    {"id":"H05","difficulty":"E","nl":"ICD-10 코드가 E11인 진단명을 조회하시오.",
     "gold_sql":"SELECT DISTINCT condition_concept FROM condition_occurrence WHERE icd10_code='E11'"},
    {"id":"H06","difficulty":"E","nl":"경구 투여 약물의 수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM drug_exposure WHERE route='경구'"},
    {"id":"H07","difficulty":"E","nl":"비정상 판정(abnormal_flag가 '위험')인 검사 결과를 모두 조회하시오.",
     "gold_sql":"SELECT * FROM measurement WHERE abnormal_flag='위험'"},
    {"id":"H08","difficulty":"E","nl":"시술 비용이 가장 높은 시술명과 비용을 조회하시오.",
     "gold_sql":"SELECT procedure_name, cost FROM procedure_occurrence ORDER BY cost DESC LIMIT 1"},
    {"id":"H09","difficulty":"E","nl":"내과 소속 의사의 이름과 면허 취득 연도를 조회하시오.",
     "gold_sql":"SELECT doctor_name, license_year FROM doctor WHERE specialty='내과'"},
    {"id":"H10","difficulty":"E","nl":"활성 상태인 진단 건수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM condition_occurrence WHERE condition_status='활성'"},

    # Medium
    {"id":"H11","difficulty":"M","nl":"환자별 방문 횟수를 구하고, 2회 이상인 환자만 조회하시오.",
     "gold_sql":"SELECT patient_id, COUNT(*) AS visit_count FROM visit_occurrence GROUP BY patient_id HAVING COUNT(*)>=2"},
    {"id":"H12","difficulty":"M","nl":"내과에서 진료받은 환자의 성별과 나이를 조회하시오.",
     "gold_sql":"SELECT DISTINCT p.patient_id, p.gender, p.age FROM patient p JOIN visit_occurrence v ON p.patient_id=v.patient_id WHERE v.department='내과'"},
    {"id":"H13","difficulty":"M","nl":"제2형 당뇨병(E11) 환자에게 처방된 모든 약물명을 조회하시오.",
     "gold_sql":"SELECT DISTINCT d.drug_name FROM drug_exposure d JOIN condition_occurrence c ON d.patient_id=c.patient_id WHERE c.icd10_code='E11'"},
    {"id":"H14","difficulty":"M","nl":"HbA1c 검사 결과가 7.0 이상인 환자의 ID와 수치를 조회하시오.",
     "gold_sql":"SELECT patient_id, value_as_number FROM measurement WHERE measurement_name='HbA1c' AND value_as_number>=7.0"},
    {"id":"H15","difficulty":"M","nl":"진단별 환자 수를 구하시오. 환자 수 내림차순 정렬.",
     "gold_sql":"SELECT condition_concept, COUNT(DISTINCT patient_id) AS patient_count FROM condition_occurrence GROUP BY condition_concept ORDER BY patient_count DESC"},
    {"id":"H16","difficulty":"M","nl":"투여 경로별 약물 처방 건수를 구하시오.",
     "gold_sql":"SELECT route, COUNT(*) AS prescription_count FROM drug_exposure GROUP BY route"},
    {"id":"H17","difficulty":"M","nl":"65세 이상 환자의 진단명을 모두 조회하시오.",
     "gold_sql":"SELECT DISTINCT c.condition_concept FROM condition_occurrence c JOIN patient p ON c.patient_id=p.patient_id WHERE p.age>=65"},
    {"id":"H18","difficulty":"M","nl":"응급 방문 환자의 진단명과 담당 의사 이름을 조회하시오.",
     "gold_sql":"SELECT c.condition_concept, d.doctor_name FROM visit_occurrence v JOIN condition_occurrence c ON v.visit_id=c.visit_id JOIN doctor d ON v.attending_doctor_id=d.doctor_id WHERE v.visit_type='응급'"},
    {"id":"H19","difficulty":"M","nl":"환자별 시술 비용 합계를 구하시오.",
     "gold_sql":"SELECT patient_id, SUM(cost) AS total_cost FROM procedure_occurrence GROUP BY patient_id"},
    {"id":"H20","difficulty":"M","nl":"메트포르민을 처방받은 환자의 HbA1c 수치를 조회하시오.",
     "gold_sql":"SELECT DISTINCT m.patient_id, m.value_as_number FROM measurement m JOIN drug_exposure d ON m.patient_id=d.patient_id WHERE d.drug_name='메트포르민' AND m.measurement_name='HbA1c'"},

    # Hard
    {"id":"H21","difficulty":"H","nl":"다제 처방(2개 이상 약물) 환자의 ID, 약물 수, 진단명을 조회하시오.",
     "gold_sql":"SELECT d.patient_id, d.drug_count, c.condition_concept FROM (SELECT patient_id, COUNT(DISTINCT drug_name) AS drug_count FROM drug_exposure GROUP BY patient_id HAVING COUNT(DISTINCT drug_name)>=2) d JOIN condition_occurrence c ON d.patient_id=c.patient_id"},
    {"id":"H22","difficulty":"H","nl":"비정상 검사 결과가 있는 환자의 진단명, 검사명, 수치를 모두 조회하시오.",
     "gold_sql":"SELECT p.patient_id, c.condition_concept, m.measurement_name, m.value_as_number, m.abnormal_flag FROM patient p JOIN condition_occurrence c ON p.patient_id=c.patient_id JOIN measurement m ON p.patient_id=m.patient_id WHERE m.abnormal_flag IN ('높음','위험','낮음')"},
    {"id":"H23","difficulty":"H","nl":"입원 기간(일수)이 가장 긴 환자의 ID, 입원일수, 진단명을 구하시오.",
     "gold_sql":"SELECT v.patient_id, JULIANDAY(v.visit_end_date)-JULIANDAY(v.visit_start_date) AS days, c.condition_concept FROM visit_occurrence v JOIN condition_occurrence c ON v.visit_id=c.visit_id WHERE v.visit_type='입원' AND v.visit_end_date IS NOT NULL ORDER BY days DESC LIMIT 1"},
    {"id":"H24","difficulty":"H","nl":"진료과별 평균 환자 나이와 총 방문 건수를 구하시오.",
     "gold_sql":"SELECT v.department, AVG(p.age) AS avg_age, COUNT(*) AS visit_count FROM visit_occurrence v JOIN patient p ON v.patient_id=p.patient_id GROUP BY v.department"},
    {"id":"H25","difficulty":"H","nl":"제2형 당뇨병 환자 중 HbA1c가 8.0 이상이면서 크레아티닌이 비정상인 환자를 조회하시오.",
     "gold_sql":"SELECT DISTINCT p.patient_id, p.age FROM patient p JOIN condition_occurrence c ON p.patient_id=c.patient_id JOIN measurement m1 ON p.patient_id=m1.patient_id JOIN measurement m2 ON p.patient_id=m2.patient_id WHERE c.icd10_code='E11' AND m1.measurement_name='HbA1c' AND m1.value_as_number>=8.0 AND m2.measurement_name='크레아티닌' AND m2.abnormal_flag IN ('높음','위험')"},
    {"id":"H26","difficulty":"H","nl":"약물별 처방 환자 수와 평균 투여일수를 구하시오. 환자 수 내림차순.",
     "gold_sql":"SELECT drug_name, COUNT(DISTINCT patient_id) AS patient_count, AVG(days_supply) AS avg_days FROM drug_exposure GROUP BY drug_name ORDER BY patient_count DESC"},
    {"id":"H27","difficulty":"H","nl":"고혈압(I10) 환자의 수축기혈압 측정값과 처방 약물을 함께 조회하시오.",
     "gold_sql":"SELECT c.patient_id, m.value_as_number AS systolic_bp, d.drug_name FROM condition_occurrence c JOIN measurement m ON c.patient_id=m.patient_id JOIN drug_exposure d ON c.patient_id=d.patient_id WHERE c.icd10_code='I10' AND m.measurement_name='수축기혈압'"},
    {"id":"H28","difficulty":"H","nl":"시술을 받은 환자 중 입원 환자의 시술명, 입원기간, 총비용을 구하시오.",
     "gold_sql":"SELECT pr.patient_id, pr.procedure_name, JULIANDAY(v.visit_end_date)-JULIANDAY(v.visit_start_date) AS days, pr.cost FROM procedure_occurrence pr JOIN visit_occurrence v ON pr.visit_id=v.visit_id WHERE v.visit_type='입원' AND v.visit_end_date IS NOT NULL"},
    {"id":"H29","difficulty":"H","nl":"성별에 따른 평균 시술 비용과 평균 나이를 비교하시오.",
     "gold_sql":"SELECT p.gender, AVG(p.age) AS avg_age, AVG(pr.cost) AS avg_cost FROM patient p JOIN procedure_occurrence pr ON p.patient_id=pr.patient_id GROUP BY p.gender"},
    {"id":"H30","difficulty":"H","nl":"2건 이상 진단받은 환자의 ID, 진단 수, 모든 ICD-10 코드를 조회하시오.",
     "gold_sql":"SELECT patient_id, COUNT(*) AS condition_count, GROUP_CONCAT(DISTINCT icd10_code) AS icd_codes FROM condition_occurrence GROUP BY patient_id HAVING COUNT(*)>=2"},
    {"id":"H31","difficulty":"M","nl":"검사명별 비정상 결과 건수를 구하시오.",
     "gold_sql":"SELECT measurement_name, COUNT(*) AS abnormal_count FROM measurement WHERE abnormal_flag!='정상' GROUP BY measurement_name"},
    {"id":"H32","difficulty":"E","nl":"전체 시술 건수를 구하시오.",
     "gold_sql":"SELECT COUNT(*) FROM procedure_occurrence"},
    {"id":"H33","difficulty":"M","nl":"환자별 처방 약물 수를 구하시오.",
     "gold_sql":"SELECT patient_id, COUNT(DISTINCT drug_name) AS drug_count FROM drug_exposure GROUP BY patient_id"},
    {"id":"H34","difficulty":"H","nl":"담당 의사별 환자 수와 입원 환자 비율을 구하시오.",
     "gold_sql":"SELECT d.doctor_name, COUNT(DISTINCT v.patient_id) AS patient_count, ROUND(SUM(CASE WHEN v.visit_type='입원' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS admission_rate_pct FROM visit_occurrence v JOIN doctor d ON v.attending_doctor_id=d.doctor_id GROUP BY d.doctor_name"},
    {"id":"H35","difficulty":"H","nl":"전체 의료비(시술비) 대비 각 환자의 비중(%)을 구하시오.",
     "gold_sql":"SELECT patient_id, SUM(cost) AS patient_cost, ROUND(SUM(cost)*100.0/(SELECT SUM(cost) FROM procedure_occurrence),2) AS cost_pct FROM procedure_occurrence GROUP BY patient_id ORDER BY cost_pct DESC"},
    {"id":"H36","difficulty":"M","nl":"방문 유형별 건수를 구하시오.",
     "gold_sql":"SELECT visit_type, COUNT(*) FROM visit_occurrence GROUP BY visit_type"},
    {"id":"H37","difficulty":"E","nl":"혈액형별 환자 수를 구하시오.",
     "gold_sql":"SELECT blood_type, COUNT(*) FROM patient GROUP BY blood_type"},
    {"id":"H38","difficulty":"H","nl":"최근 6개월 이내 2회 이상 방문한 환자의 ID와 방문 횟수를 구하시오.",
     "gold_sql":"SELECT patient_id, COUNT(*) AS visit_count FROM visit_occurrence WHERE visit_start_date>='2024-09-17' GROUP BY patient_id HAVING COUNT(*)>=2"},
    {"id":"H39","difficulty":"M","nl":"진료과별 처방된 고유 약물 수를 구하시오.",
     "gold_sql":"SELECT v.department, COUNT(DISTINCT d.drug_name) AS unique_drugs FROM drug_exposure d JOIN visit_occurrence v ON d.visit_id=v.visit_id GROUP BY v.department"},
    {"id":"H40","difficulty":"H","nl":"당뇨병(E11) 환자의 HbA1c 추이를 환자별, 날짜순으로 조회하시오.",
     "gold_sql":"SELECT c.patient_id, m.measurement_date, m.value_as_number AS HbA1c FROM condition_occurrence c JOIN measurement m ON c.patient_id=m.patient_id WHERE c.icd10_code='E11' AND m.measurement_name='HbA1c' ORDER BY c.patient_id, m.measurement_date"},
]


# ============================================================
# 3. Prompt Templates
# ============================================================
PROMPT_B1 = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.
Output ONLY the SQL query, nothing else. Do not include explanation.

Question: {question}

SQL:"""

PROMPT_B2 = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.
You have access to the following database documentation chunks:

---
{retrieved_chunks}
---

Output ONLY the SQL query, nothing else. Do not include explanation.

Question: {question}

SQL:"""

def get_b2_chunks_healthcare(question: str) -> str:
    chunks = []
    kw_map = {
        "환자": "patient 테이블: patient_id(INTEGER PK), gender(TEXT), birth_year(INTEGER), age(INTEGER), blood_type(TEXT), insurance_type(TEXT)",
        "방문": "visit_occurrence 테이블: visit_id(INTEGER PK), patient_id(INTEGER FK), visit_type(TEXT), visit_start_date(TEXT), visit_end_date(TEXT), department(TEXT), attending_doctor_id(INTEGER)",
        "진단": "condition_occurrence 테이블: condition_id(INTEGER PK), patient_id(INTEGER FK), visit_id(INTEGER FK), condition_concept(TEXT), icd10_code(TEXT), condition_start_date(TEXT), condition_end_date(TEXT), condition_status(TEXT)",
        "약물": "drug_exposure 테이블: drug_exposure_id(INTEGER PK), patient_id(INTEGER FK), visit_id(INTEGER FK), drug_name(TEXT), drug_concept_id(INTEGER), dose_value(REAL), dose_unit(TEXT), route(TEXT), days_supply(INTEGER), prescribe_date(TEXT)",
        "처방": "drug_exposure 테이블: drug_exposure_id(INTEGER PK), patient_id(INTEGER FK), visit_id(INTEGER FK), drug_name(TEXT), drug_concept_id(INTEGER), dose_value(REAL), dose_unit(TEXT), route(TEXT), days_supply(INTEGER), prescribe_date(TEXT)",
        "검사": "measurement 테이블: measurement_id(INTEGER PK), patient_id(INTEGER FK), visit_id(INTEGER FK), measurement_name(TEXT), value_as_number(REAL), unit(TEXT), range_low(REAL), range_high(REAL), abnormal_flag(TEXT), measurement_date(TEXT)",
        "수치": "measurement 테이블: measurement_id(INTEGER PK), patient_id(INTEGER FK), visit_id(INTEGER FK), measurement_name(TEXT), value_as_number(REAL), unit(TEXT), range_low(REAL), range_high(REAL), abnormal_flag(TEXT), measurement_date(TEXT)",
        "시술": "procedure_occurrence 테이블: procedure_id(INTEGER PK), patient_id(INTEGER FK), visit_id(INTEGER FK), procedure_name(TEXT), procedure_date(TEXT), cost(INTEGER)",
        "비용": "procedure_occurrence 테이블: procedure_id(INTEGER PK), patient_id(INTEGER FK), visit_id(INTEGER FK), procedure_name(TEXT), procedure_date(TEXT), cost(INTEGER)",
        "의사": "doctor 테이블: doctor_id(INTEGER PK), doctor_name(TEXT), specialty(TEXT), license_year(INTEGER)",
        "입원": "visit_type 값: 외래, 입원, 응급",
        "ICD": "icd10_code 예: E11(제2형당뇨병), I10(고혈압), I21(급성심근경색), I63(뇌경색), C50(유방암), J45(천식), J18(폐렴), N18(만성신장병), E78(고지혈증)",
        "HbA1c": "measurement_name 종류: HbA1c, 공복혈당, 트로포닌I, CK-MB, 수축기혈압, 이완기혈압, 크레아티닌, eGFR, WBC, CRP, LDL콜레스테롤, 총콜레스테롤, TSH, Free T4",
        "당뇨": "icd10_code E11 = 제2형 당뇨병. 관련 약물: 메트포르민(drug_concept_id=1503297), 인슐린 글라진",
        "혈압": "고혈압 icd10_code=I10. 검사: 수축기혈압, 이완기혈압. 약물: 암로디핀, 발사르탄",
    }
    matched = set()
    for kw, chunk in kw_map.items():
        if kw in question:
            matched.add(chunk)
    if not matched:
        matched.add(kw_map["환자"])
        matched.add(kw_map["방문"])
    return "\n".join(matched)


PROMPT_DKAP = """You are a SQL expert. Convert the following Korean natural language question into a valid SQLite SQL query.

## Database Schema (OMOP CDM-inspired Clinical Data Warehouse)

### Table: patient (환자 기본정보)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| patient_id | INTEGER PK | 환자 고유 번호 | identifier |
| gender | TEXT NOT NULL | 성별 (M:남성, F:여성) | category |
| birth_year | INTEGER | 출생 연도 | date |
| age | INTEGER | 만 나이 | count |
| blood_type | TEXT | 혈액형 (A/B/AB/O) | category |
| insurance_type | TEXT | 보험 유형 (건강보험/의료급여/산재/자동차) | category |

### Table: visit_occurrence (방문 기록)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| visit_id | INTEGER PK AUTOINCREMENT | 방문 고유 번호 | identifier |
| patient_id | INTEGER FK→patient | 환자 번호 | identifier |
| visit_type | TEXT NOT NULL | 방문 유형 (외래/입원/응급) | category |
| visit_start_date | TEXT NOT NULL | 방문 시작일 (YYYY-MM-DD) | date |
| visit_end_date | TEXT | 방문 종료일 (입원 시 퇴원일, 외래는 NULL) | date |
| department | TEXT | 진료과 (내과/외과/정형외과/소아과/산부인과/신경과/정신건강의학과) | category |
| attending_doctor_id | INTEGER FK→doctor | 담당 의사 ID | identifier |

### Table: condition_occurrence (진단 기록)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| condition_id | INTEGER PK AUTOINCREMENT | 진단 고유 번호 | identifier |
| patient_id | INTEGER FK→patient | 환자 번호 | identifier |
| visit_id | INTEGER FK→visit_occurrence | 방문 번호 | identifier |
| condition_concept | TEXT NOT NULL | 진단명 (KCD-8 한국표준질병분류 기준) | text |
| icd10_code | TEXT | ICD-10 코드 | code |
| condition_start_date | TEXT NOT NULL | 진단 시작일 | date |
| condition_end_date | TEXT | 진단 종료일 (완치 시) | date |
| condition_status | TEXT DEFAULT '활성' | 진단 상태 (활성/완치/관해) | category |

### Table: drug_exposure (약물 처방 기록)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| drug_exposure_id | INTEGER PK AUTOINCREMENT | 처방 고유 번호 | identifier |
| patient_id | INTEGER FK→patient | 환자 번호 | identifier |
| visit_id | INTEGER FK→visit_occurrence | 방문 번호 | identifier |
| drug_name | TEXT NOT NULL | 약물 일반명 (한국어) | text |
| drug_concept_id | INTEGER | OMOP concept ID | code |
| dose_value | REAL | 투여량 | measurement |
| dose_unit | TEXT | 투여 단위 (mg/ml/mcg/IU/g/mg/kg) | unit |
| route | TEXT | 투여 경로 (경구/정맥주사/피하주사/외용/흡입) | category |
| days_supply | INTEGER | 투여 일수 | count |
| prescribe_date | TEXT NOT NULL | 처방 일자 | date |

### Table: measurement (검사 결과)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| measurement_id | INTEGER PK AUTOINCREMENT | 검사 고유 번호 | identifier |
| patient_id | INTEGER FK→patient | 환자 번호 | identifier |
| visit_id | INTEGER FK→visit_occurrence | 방문 번호 | identifier |
| measurement_name | TEXT NOT NULL | 검사명 (한국어) | text |
| value_as_number | REAL | 검사 수치 | measurement |
| unit | TEXT | 단위 | unit |
| range_low | REAL | 정상 범위 하한 | measurement |
| range_high | REAL | 정상 범위 상한 | measurement |
| abnormal_flag | TEXT | 판정 (정상/높음/낮음/위험) | category |
| measurement_date | TEXT NOT NULL | 검사 일자 | date |

### Table: procedure_occurrence (시술/수술 기록)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| procedure_id | INTEGER PK AUTOINCREMENT | 시술 고유 번호 | identifier |
| patient_id | INTEGER FK→patient | 환자 번호 | identifier |
| visit_id | INTEGER FK→visit_occurrence | 방문 번호 | identifier |
| procedure_name | TEXT NOT NULL | 시술명 | text |
| procedure_date | TEXT NOT NULL | 시술 일자 | date |
| cost | INTEGER | 시술 비용 (원, KRW) | financial_amount |

### Table: doctor (의사 정보)
| Column | Type | Description | Domain |
|--------|------|-------------|--------|
| doctor_id | INTEGER PK | 의사 고유 번호 | identifier |
| doctor_name | TEXT | 의사명 | text |
| specialty | TEXT | 전문과목 | category |
| license_year | INTEGER | 면허 취득 연도 | date |

## Clinical Domain Glossary
- ICD-10 코드 매핑: E11=제2형 당뇨병, I10=고혈압, I21=급성 심근경색, I63=뇌경색, C50=유방암, J45=천식, J18=폐렴, N18=만성 신장병, E78=고지혈증, M51=요추 추간판 탈출증, S36=복부 외상, E03=갑상선기능저하증
- drug_concept_id 매핑: 메트포르민=1503297, 아스피린=1112807, 헤파린=1367571, 타목시펜=1436678, 암로디핀=1332418
- visit_type: 외래(outpatient, visit_end_date=NULL), 입원(inpatient, visit_end_date=퇴원일), 응급(emergency)
- condition_status: 활성(현재 치료 중), 완치(치료 완료), 관해(증상 소실 관찰 중)
- abnormal_flag: 정상(범위 내), 높음(상한 초과), 낮음(하한 미만), 위험(임상적 위험 수준)
- 입원 기간 계산: JULIANDAY(visit_end_date) - JULIANDAY(visit_start_date)

## Join Rules
- patient ↔ visit_occurrence: ON patient_id
- visit_occurrence ↔ condition_occurrence: ON visit_id (또는 patient_id)
- visit_occurrence ↔ drug_exposure: ON visit_id (또는 patient_id)
- visit_occurrence ↔ measurement: ON visit_id (또는 patient_id)
- visit_occurrence ↔ procedure_occurrence: ON visit_id
- visit_occurrence ↔ doctor: ON attending_doctor_id = doctor_id
- condition/drug/measurement/procedure → patient: ON patient_id (직접 조인 가능)

Output ONLY the SQL query, nothing else. Do not include explanation.

Question: {question}

SQL:"""


# ============================================================
# 4-7. Reuse same infrastructure as finance experiment
# ============================================================
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
    except:
        return False, None

def normalize_sql(sql):
    return re.sub(r'\s+', ' ', sql.lower().strip()).rstrip(';').strip()

def evaluate_ex(conn, pred_sql, gold_sql):
    pred_ok, pred_r = execute_sql(conn, pred_sql)
    gold_ok, gold_r = execute_sql(conn, gold_sql)
    if not pred_ok or not gold_ok: return False
    if pred_r is None or gold_r is None: return False
    if "order by" in gold_sql.lower(): return pred_r == gold_r
    return set(map(tuple, pred_r)) == set(map(tuple, gold_r))

def evaluate_em(pred, gold):
    return normalize_sql(pred) == normalize_sql(gold)

@dataclass
class Result:
    condition: str; run_id: int; question_id: str; difficulty: str
    nl_question: str; gold_sql: str; pred_sql: str
    ex_score: bool; em_score: bool; execution_ok: bool; latency_ms: float

def run_experiment():
    logger.info("=" * 60)
    logger.info("DKAP Healthcare Clinical QA Experiment")
    logger.info("=" * 60)
    conn = setup_database(DB_PATH)
    for t in ['patient','visit_occurrence','condition_occurrence','drug_exposure','measurement','procedure_occurrence','doctor']:
        c = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        logger.info(f"  {t}: {c} rows")
    test = call_vllm("SELECT 1")
    if not test: logger.error("vLLM unavailable"); sys.exit(1)
    logger.info(f"  vLLM OK")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = []
    conditions = {
        "B1": lambda q: PROMPT_B1.format(question=q),
        "B2": lambda q: PROMPT_B2.format(question=q, retrieved_chunks=get_b2_chunks_healthcare(q)),
        "DKAP": lambda q: PROMPT_DKAP.format(question=q),
    }
    total = len(BENCHMARK) * len(conditions) * NUM_RUNS
    progress = 0

    for run_id in range(1, NUM_RUNS+1):
        logger.info(f"\n--- Run {run_id}/{NUM_RUNS} ---")
        for cond_name, prompt_fn in conditions.items():
            logger.info(f"  Condition: {cond_name}")
            for item in BENCHMARK:
                progress += 1
                prompt = prompt_fn(item["nl"])
                t0 = time.time()
                pred = call_vllm(prompt)
                lat = (time.time()-t0)*1000
                ex_ok, _ = execute_sql(conn, pred)
                ex_s = evaluate_ex(conn, pred, item["gold_sql"]) if ex_ok else False
                em_s = evaluate_em(pred, item["gold_sql"])
                all_results.append(Result(cond_name, run_id, item["id"], item["difficulty"],
                    item["nl"], item["gold_sql"], pred, ex_s, em_s, ex_ok, lat))
                st = "EX" if ex_s else ("EXEC" if ex_ok else "FAIL")
                logger.info(f"    [{progress}/{total}] {item['id']} ({item['difficulty']}) -> {st} ({lat:.0f}ms)")
    conn.close()

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    with open(RESULTS_DIR / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    summary = {}
    for cond in ["B1","B2","DKAP"]:
        cr = [r for r in all_results if r.condition==cond]
        n = len(cr)
        ex_t = sum(1 for r in cr if r.ex_score)
        em_t = sum(1 for r in cr if r.em_score)
        exec_t = sum(1 for r in cr if r.execution_ok)
        avg_lat = sum(r.latency_ms for r in cr)/n if n else 0
        diff_b = {}
        for d in ["E","M","H"]:
            dr = [r for r in cr if r.difficulty==d]
            dn = len(dr)
            if dn:
                diff_b[d] = {"n":dn,"EX":sum(1 for r in dr if r.ex_score),
                    "EX_pct":round(sum(1 for r in dr if r.ex_score)*100/dn,1),
                    "EM":sum(1 for r in dr if r.em_score),
                    "EM_pct":round(sum(1 for r in dr if r.em_score)*100/dn,1)}
        run_ex = {}
        for rid in range(1,NUM_RUNS+1):
            rr = [r for r in cr if r.run_id==rid]
            rn = len(rr)
            run_ex[f"run_{rid}"] = round(sum(1 for r in rr if r.ex_score)*100/rn,1) if rn else 0
        summary[cond] = {"n":n,"EX_total":ex_t,"EX_pct":round(ex_t*100/n,1) if n else 0,
            "EM_total":em_t,"EM_pct":round(em_t*100/n,1) if n else 0,
            "EXEC_total":exec_t,"EXEC_pct":round(exec_t*100/n,1) if n else 0,
            "avg_latency_ms":round(avg_lat,1),"by_difficulty":diff_b,"by_run":run_ex}

    logger.info(f"{'Cond':<8} {'EX%':<8} {'EM%':<8} {'EXEC%':<8} {'Lat(ms)':<10}")
    logger.info("-"*42)
    for c in ["B1","B2","DKAP"]:
        s=summary[c]; logger.info(f"{c:<8} {s['EX_pct']:<8} {s['EM_pct']:<8} {s['EXEC_pct']:<8} {s['avg_latency_ms']:<10}")
    for c in ["B1","B2","DKAP"]:
        logger.info(f"  {c}:")
        for d in ["E","M","H"]:
            dd=summary[c]["by_difficulty"].get(d,{})
            if dd: logger.info(f"    {d}: EX={dd['EX_pct']}% ({dd['EX']}/{dd['n']})")

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSummary saved to {RESULTS_DIR / 'summary.json'}")
    logger.info("EXPERIMENT COMPLETE")
    return summary

if __name__ == "__main__":
    run_experiment()
