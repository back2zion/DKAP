"""
domain_adaptive_pipeline.py
===========================
Cross-Domain LLM Pipeline 실험 프레임워크
대상 도메인: 금융 (AIhub) / 의료 (AIhub) / 공공 (AIhub)
베이스라인: Vanilla LLM / Single RAG / FinSQL-style
제안 방법: Domain-Adaptive Multi-Stage Pipeline

논문 실험 재현 가능 구조 (IEEE Access 투고 기준)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ──────────────────────────────────────────
# 1. 도메인 설정
# ──────────────────────────────────────────

@dataclass
class DomainConfig:
    name: str                          # 'finance' | 'healthcare' | 'public'
    aihub_dataset_id: str              # AIhub 데이터셋 ID
    metadata_path: str                 # 용어사전 / 온톨로지 경로
    schema_info: dict = field(default_factory=dict)  # 도메인 스키마
    task_type: str = "text2sql"        # 'text2sql' | 'ner' | 'qa'
    reference_benchmark: str = ""      # FinSQL / EHRSQL / BiomedSQL

DOMAIN_CONFIGS = {
    "finance": DomainConfig(
        name="finance",
        aihub_dataset_id="71723",        # AIhub 금융 자연어처리 데이터
        metadata_path="data/finance/terminology.json",
        schema_info={
            "tables": ["account", "transaction", "customer", "loan"],
            "domain_terms": "금융 용어사전",
            "governance": "PBAC 기반 접근 제어",
        },
        task_type="text2sql",
        reference_benchmark="FinSQL",
    ),
    "healthcare": DomainConfig(
        name="healthcare",
        aihub_dataset_id="71777",        # AIhub 의료 질의응답 데이터
        metadata_path="data/healthcare/omop_cdm_mapping.json",
        schema_info={
            "standard": "OMOP CDM",
            "ontologies": ["SNOMED-CT", "ICD-10", "LOINC", "RxNorm"],
            "ner_labels": ["Disease", "Drug", "Procedure", "Lab"],
        },
        task_type="text2sql",            # NER도 병행
        reference_benchmark="EHRSQL",
    ),
    "public": DomainConfig(
        name="public",
        aihub_dataset_id="71401",        # AIhub 공공 행정 문서 QA
        metadata_path="data/public/regulation_keywords.json",
        schema_info={
            "document_types": ["법령", "고시", "지침", "매뉴얼"],
            "domain_terms": "행정 키워드",
        },
        task_type="qa",
        reference_benchmark="KorQuAD",
    ),
}


# ──────────────────────────────────────────
# 2. 파이프라인 구현
# ──────────────────────────────────────────

class DomainAdaptivePipeline:
    """
    제안 3단계 파이프라인
    Stage 1: 도메인 메타데이터 구조화 (핵심 독립변수)
    Stage 2: 다단계 파이프라인 주입
    Stage 3: LLM 추론
    """

    def __init__(self, llm_client, vector_store, config: DomainConfig):
        self.llm = llm_client
        self.vs  = vector_store
        self.cfg = config
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        path = Path(self.cfg.metadata_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # 메타데이터 파일 없을 때 기본 구조 반환
        return {"terms": {}, "schema": self.cfg.schema_info}

    # ── Stage 1 ──
    def stage1_metadata_grounding(self, query: str) -> dict:
        """
        도메인 메타데이터를 쿼리에 결합
        - 용어 정규화 (도메인 용어 → 스키마 컬럼명 매핑)
        - 스키마 컨텍스트 추가
        - 도메인 제약 조건 명시
        """
        grounded = {
            "original_query": query,
            "normalized_terms": self._normalize_terms(query),
            "schema_context": self._build_schema_context(),
            "domain_constraints": self._get_domain_constraints(),
        }
        return grounded

    # ── Stage 2 ──
    def stage2_retrieval_injection(self, grounded: dict) -> dict:
        """
        다단계 검색 및 컨텍스트 주입
        - 벡터 검색 (의미 유사도)
        - 키워드 검색 (BM25)
        - 도메인 규칙 기반 필터링
        """
        query = grounded["original_query"]
        retrieved = self.vs.search(query, top_k=5, domain=self.cfg.name)
        enriched = {
            **grounded,
            "retrieved_docs": retrieved,
            "injection_prompt": self._build_injection_prompt(grounded, retrieved),
        }
        return enriched

    # ── Stage 3 ──
    def stage3_llm_inference(self, enriched: dict) -> dict:
        """
        LLM 추론 실행
        """
        prompt = enriched["injection_prompt"]
        response = self.llm.generate(prompt)
        return {
            "query": enriched["original_query"],
            "response": response,
            "domain": self.cfg.name,
            "stage_trace": enriched,
        }

    def run(self, query: str) -> dict:
        g = self.stage1_metadata_grounding(query)
        e = self.stage2_retrieval_injection(g)
        r = self.stage3_llm_inference(e)
        return r

    # ── 헬퍼 ──
    def _normalize_terms(self, query: str) -> dict:
        terms = self.metadata.get("terms", {})
        matched = {t: terms[t] for t in terms if t in query}
        return matched

    def _build_schema_context(self) -> str:
        schema = self.cfg.schema_info
        return json.dumps(schema, ensure_ascii=False, indent=2)

    def _get_domain_constraints(self) -> list:
        constraints_map = {
            "finance":    ["PII 마스킹 필수", "PBAC 권한 확인", "금액 단위: 원"],
            "healthcare": ["개인정보 비식별화", "ICD 코드 정규화", "단위: mg/dL"],
            "public":     ["법령 출처 명시", "개정 이력 확인"],
        }
        return constraints_map.get(self.cfg.name, [])

    def _build_injection_prompt(self, grounded: dict, retrieved: list) -> str:
        domain_instruction = {
            "finance":    "당신은 금융 데이터 전문 SQL 생성 AI입니다.",
            "healthcare": "당신은 임상 데이터 분석 전문 AI입니다.",
            "public":     "당신은 행정 문서 질의응답 전문 AI입니다.",
        }
        base = domain_instruction.get(self.cfg.name, "당신은 전문 AI입니다.")
        schema_ctx = grounded.get("schema_context", "")
        terms_ctx = json.dumps(grounded.get("normalized_terms", {}), ensure_ascii=False)
        docs_ctx = "\n".join([d.get("text", "") for d in retrieved[:3]])
        constraints = "\n".join(grounded.get("domain_constraints", []))
        query = grounded["original_query"]

        prompt = f"""{base}

[도메인 스키마]
{schema_ctx}

[도메인 용어 매핑]
{terms_ctx}

[관련 문서]
{docs_ctx}

[도메인 제약 조건]
{constraints}

[질문]
{query}

[답변]"""
        return prompt


# ──────────────────────────────────────────
# 3. Baseline 구현
# ──────────────────────────────────────────

class VanillaLLMBaseline:
    """Baseline 1: 도메인 지식 없이 LLM 직접 호출"""
    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, query: str, domain: str) -> dict:
        response = self.llm.generate(query)
        return {"query": query, "response": response, "domain": domain}


class SingleRAGBaseline:
    """Baseline 2: 단순 벡터 검색 + LLM (도메인 메타데이터 없음)"""
    def __init__(self, llm_client, vector_store):
        self.llm = llm_client
        self.vs  = vector_store

    def run(self, query: str, domain: str) -> dict:
        docs = self.vs.search(query, top_k=3, domain=domain)
        context = "\n".join([d.get("text", "") for d in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = self.llm.generate(prompt)
        return {"query": query, "response": response, "domain": domain}


# ──────────────────────────────────────────
# 4. 평가 지표
# ──────────────────────────────────────────

class Evaluator:
    """
    도메인별 평가 지표
    - text2sql: Execution Accuracy (EX), Exact Match (EM)
    - ner:      F1 Score (Precision, Recall)
    - qa:       Answer Accuracy, F1
    """

    @staticmethod
    def execution_accuracy(predictions: list, gold_sqls: list, db_conn) -> float:
        """SQL 실행 결과가 정답과 일치하는 비율"""
        correct = 0
        for pred, gold in zip(predictions, gold_sqls):
            try:
                pred_result = db_conn.execute(pred)
                gold_result = db_conn.execute(gold)
                if set(pred_result) == set(gold_result):
                    correct += 1
            except Exception:
                pass
        return correct / len(predictions) if predictions else 0.0

    @staticmethod
    def exact_match(predictions: list, gold: list) -> float:
        """정규화 후 문자열 완전 일치"""
        def normalize(s): return " ".join(s.lower().split())
        matches = sum(1 for p, g in zip(predictions, gold) if normalize(p) == normalize(g))
        return matches / len(predictions) if predictions else 0.0

    @staticmethod
    def ner_f1(pred_entities: list, gold_entities: list) -> dict:
        """NER F1 (micro average)"""
        tp = sum(len(set(p) & set(g)) for p, g in zip(pred_entities, gold_entities))
        fp = sum(len(set(p) - set(g)) for p, g in zip(pred_entities, gold_entities))
        fn = sum(len(set(g) - set(p)) for p, g in zip(pred_entities, gold_entities))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}


# ──────────────────────────────────────────
# 5. 실험 실행기
# ──────────────────────────────────────────

class ExperimentRunner:
    """
    IEEE Access 논문 실험 전체 실행
    - 5회 반복 → 평균 ± 표준편차
    - 도메인 2~3개 병렬 비교
    - 결과 JSON 저장
    """

    def __init__(self, llm_client, vector_store, n_runs: int = 5):
        self.llm     = llm_client
        self.vs      = vector_store
        self.n_runs  = n_runs
        self.results = {}

    def run_domain(self, domain_name: str, test_data: list) -> dict:
        """단일 도메인 실험: Baseline 3종 vs 제안 파이프라인"""
        config   = DOMAIN_CONFIGS[domain_name]
        pipeline = DomainAdaptivePipeline(self.llm, self.vs, config)
        b1       = VanillaLLMBaseline(self.llm)
        b2       = SingleRAGBaseline(self.llm, self.vs)

        run_results = {"proposed": [], "vanilla": [], "rag": []}

        for run_idx in range(self.n_runs):
            print(f"  [{domain_name}] Run {run_idx+1}/{self.n_runs}")
            proposed_preds, vanilla_preds, rag_preds = [], [], []

            for sample in test_data:
                query = sample["question"]
                proposed_preds.append(pipeline.run(query)["response"])
                vanilla_preds.append(b1.run(query, domain_name)["response"])
                rag_preds.append(b2.run(query, domain_name)["response"])

            gold = [s["answer"] for s in test_data]
            run_results["proposed"].append(Evaluator.exact_match(proposed_preds, gold))
            run_results["vanilla"].append(Evaluator.exact_match(vanilla_preds, gold))
            run_results["rag"].append(Evaluator.exact_match(rag_preds, gold))

        return self._aggregate(run_results)

    def run_all_domains(self, datasets: dict) -> dict:
        """금융·의료·공공 전체 실험 실행"""
        for domain, data in datasets.items():
            print(f"\n[{domain.upper()} 도메인 실험 시작]")
            self.results[domain] = self.run_domain(domain, data)
        return self.results

    def _aggregate(self, run_results: dict) -> dict:
        import statistics
        aggregated = {}
        for method, scores in run_results.items():
            aggregated[method] = {
                "mean":  round(statistics.mean(scores), 4),
                "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
                "runs":  scores,
            }
        return aggregated

    def save_results(self, output_path: str = "results/experiment_results.json"):
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장 완료: {output_path}")

    def print_summary(self):
        """IEEE Access 논문 Table 형식으로 결과 출력"""
        print("\n" + "="*65)
        print("실험 결과 요약 (Mean ± Std)")
        print("="*65)
        header = f"{'도메인':<12} {'제안 방법':<18} {'Vanilla LLM':<18} {'Single RAG':<18}"
        print(header)
        print("-"*65)
        for domain, metrics in self.results.items():
            p = metrics.get("proposed", {})
            v = metrics.get("vanilla",  {})
            r = metrics.get("rag",      {})
            row = (
                f"{domain:<12} "
                f"{p.get('mean',0):.4f}±{p.get('stdev',0):.4f}   "
                f"{v.get('mean',0):.4f}±{v.get('stdev',0):.4f}   "
                f"{r.get('mean',0):.4f}±{r.get('stdev',0):.4f}"
            )
            print(row)
        print("="*65)


# ──────────────────────────────────────────
# 6. 진입점 (테스트용 더미 실행)
# ──────────────────────────────────────────

class DummyLLM:
    """LLM 클라이언트 플레이스홀더 — 실제 Qwen3/SOLAR로 교체"""
    def generate(self, prompt: str) -> str:
        return "SELECT * FROM account WHERE id = 1"

class DummyVectorStore:
    """벡터 스토어 플레이스홀더 — Qdrant/Milvus로 교체"""
    def search(self, query: str, top_k: int = 5, domain: str = "") -> list:
        return [{"text": f"[{domain}] 관련 문서 {i+1}" } for i in range(top_k)]


if __name__ == "__main__":
    llm = DummyLLM()
    vs  = DummyVectorStore()

    # 더미 테스트 데이터
    test_datasets = {
        "finance":    [{"question": "지난달 이체 금액이 100만원 이상인 계좌 목록을 보여줘", "answer": "SELECT ..."}],
        "healthcare": [{"question": "당뇨 환자 중 메트포르민을 처방받은 환자 수는?", "answer": "SELECT ..."}],
        "public":     [{"question": "도로법 제45조의 내용은 무엇입니까?", "answer": "도로법 제45조..."}],
    }

    runner = ExperimentRunner(llm, vs, n_runs=3)
    runner.run_all_domains(test_datasets)
    runner.print_summary()
    runner.save_results()
