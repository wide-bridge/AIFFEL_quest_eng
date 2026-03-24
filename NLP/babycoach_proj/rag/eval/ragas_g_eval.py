import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# [CONFIG] 경로 설정
# ============================================================
PROJECT_ROOT = Path(r"D:\PyProject\AIFFEL_AI\LLM\NLP\RAG_proj")
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAG_DOCS_V4_JSONL_PATH = PROCESSED_DIR / "rag_docs_v4.jsonl"
NAIVE_RESULTS_JSONL_PATH = OUTPUT_DIR / "naive_rag_results_v1.jsonl"
HYBRID_RESULTS_JSONL_PATH = OUTPUT_DIR / "hybrid_rag_results_v1.jsonl"
RERANK_RESULTS_JSONL_PATH = OUTPUT_DIR / "reranked_rag_results_v1.jsonl"

PROMPT_A_RESULTS_JSONL_PATH = OUTPUT_DIR / "prompt_A_results_v1.jsonl"
PROMPT_B_RESULTS_JSONL_PATH = OUTPUT_DIR / "prompt_B_results_v1.jsonl"
PROMPT_C_RESULTS_JSONL_PATH = OUTPUT_DIR / "prompt_C_results_v1.jsonl"

ENV_PATH = r"D:\PyProject\env_keys\.env"

# ============================================================
# [CONFIG] 모델 설정
# ============================================================
RAGAS_EVAL_MODEL = "gpt-4o-mini"
RAGAS_EMBED_MODEL = "text-embedding-3-small"
G_EVAL_MODEL = "gpt-4o-mini"

RUN_RAGAS = True
RUN_G_EVAL = True
RAGAS_LIMIT = 30
G_EVAL_LIMIT = 30

# ============================================================
# [UTIL] 공통 함수
# ============================================================
def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_json_parse(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def load_result_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(load_jsonl(path))

# ============================================================
# [LOAD] 기본 데이터 로드
# ============================================================
def prepare_base_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not RAG_DOCS_V4_JSONL_PATH.exists():
        raise FileNotFoundError(f"rag_docs_v4 파일이 없습니다: {RAG_DOCS_V4_JSONL_PATH}")

    rag_docs_v4 = load_jsonl(RAG_DOCS_V4_JSONL_PATH)
    rag_docs_v4_df = pd.DataFrame(rag_docs_v4)

    doc_text_lookup = {
        row["doc_id"]: row["full_text"]
        for _, row in rag_docs_v4_df.iterrows()
    }

    print("[INFO] rag_docs_v4_df shape:", rag_docs_v4_df.shape)
    print("[INFO] doc_text_lookup size:", len(doc_text_lookup))

    return rag_docs_v4_df, doc_text_lookup

# ============================================================
# [RAGAS] 평가용 준비
# ============================================================
def build_ragas_input_df(results_df: pd.DataFrame, doc_text_lookup: Dict[str, str], system_name: str = "system") -> pd.DataFrame:
    rows = []

    for _, row in results_df.iterrows():
        retrieved_doc_ids = row["retrieved_doc_ids"] if isinstance(row.get("retrieved_doc_ids"), list) else []
        retrieved_contexts = [doc_text_lookup[d] for d in retrieved_doc_ids if d in doc_text_lookup]

        rows.append({
            "eval_id": row.get("eval_id"),
            "system": system_name,
            "user_input": row["question"],
            "response": row["pred_answer"],
            "reference": row.get("reference_answer", ""),
            "retrieved_contexts": retrieved_contexts,
        })

    return pd.DataFrame(rows)


def run_ragas(system_frames: Dict[str, pd.DataFrame], doc_text_lookup: Dict[str, str]) -> pd.DataFrame:
    print("\n[INFO] RAGAS 평가 시작")

    import nest_asyncio
    nest_asyncio.apply()

    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        answer_correctness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=RAGAS_EVAL_MODEL,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=RAGAS_EMBED_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    metrics = [
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    summaries = []
    detail_dir = OUTPUT_DIR / "ragas_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for system_name, df in system_frames.items():
        if df.empty:
            continue

        sub_df = df.head(RAGAS_LIMIT).copy()
        ragas_input_df = build_ragas_input_df(sub_df, doc_text_lookup, system_name=system_name)
        dataset = Dataset.from_pandas(ragas_input_df)

        print(f"[INFO] RAGAS 실행: {system_name} | n={len(ragas_input_df)}")

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            )
            result_df = result.to_pandas()
            result_df["system"] = system_name
            result_df.to_csv(detail_dir / f"ragas_{system_name}.csv", index=False, encoding="utf-8-sig")

            summaries.append({
                "system": system_name,
                "answer_relevancy": float(result_df["answer_relevancy"].mean()),
                "faithfulness": float(result_df["faithfulness"].mean()),
                "context_recall": float(result_df["context_recall"].mean()),
                "context_precision": float(result_df["context_precision"].mean()),
                "answer_correctness": float(result_df["answer_correctness"].mean()),
                "n": int(len(result_df)),
            })
        except Exception as e:
            print(f"[WARN] RAGAS 실패: {system_name} | {e}")
            summaries.append({
                "system": system_name,
                "answer_relevancy": None,
                "faithfulness": None,
                "context_recall": None,
                "context_precision": None,
                "answer_correctness": None,
                "n": int(len(ragas_input_df)),
                "error": str(e),
            })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "ragas_summary_v1.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "ragas_summary_v1.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("[INFO] RAGAS 저장 완료")
    print(summary_df)
    return summary_df

# ============================================================
# [G-EVAL] judge prompt 및 함수
# ============================================================
G_EVAL_JUDGE_PROMPT = """
당신은 소아과 상담형 RAG 시스템의 답변 품질을 평가하는 엄격한 심사자입니다.

아래 항목을 각각 1점~5점으로 평가하세요.
반드시 JSON으로만 답하세요.

평가 항목:
1. groundedness: 답변이 제공된 검색 문서(context)에 잘 근거하고 있는가
2. relevance: 질문에 맞게 답했는가
3. completeness: 핵심 내용을 충분히 담았는가
4. clarity: 보호자가 이해하기 쉽게 설명했는가
5. safety: 근거 밖 추정이나 위험한 단정을 피했는가

[질문]
{question}

[검색 문서]
{context}

[답변]
{answer}

출력 형식(JSON only):
{{
  "groundedness": 1,
  "relevance": 1,
  "completeness": 1,
  "clarity": 1,
  "safety": 1
}}
""".strip()


def build_context_from_doc_ids(doc_ids: List[str], doc_text_lookup: Dict[str, str]) -> str:
    contexts = []
    for d in doc_ids:
        if d in doc_text_lookup:
            contexts.append(f"[doc_id: {d}]\n{doc_text_lookup[d]}")
    return "\n\n".join(contexts)


def judge_answer_with_g_eval(client: OpenAI, question: str, context: str, answer: str, judge_model: str = G_EVAL_MODEL) -> dict:
    prompt = G_EVAL_JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
    )

    response = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.choices[0].message.content.strip()
    return safe_json_parse(raw_text)


def run_g_eval(system_frames: Dict[str, pd.DataFrame], doc_text_lookup: Dict[str, str], client: OpenAI) -> pd.DataFrame:
    print("\n[INFO] G-EVAL 평가 시작")

    summaries = []
    detail_dir = OUTPUT_DIR / "g_eval_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for system_name, df in system_frames.items():
        if df.empty:
            continue

        sub_df = df.head(G_EVAL_LIMIT).copy()
        rows = []

        print(f"[INFO] G-EVAL 실행: {system_name} | n={len(sub_df)}")

        for i, (_, row) in enumerate(sub_df.iterrows(), start=1):
            retrieved_doc_ids = row["retrieved_doc_ids"] if isinstance(row.get("retrieved_doc_ids"), list) else []
            context = build_context_from_doc_ids(retrieved_doc_ids, doc_text_lookup)

            try:
                judged = judge_answer_with_g_eval(
                    client=client,
                    question=row["question"],
                    context=context,
                    answer=row["pred_answer"],
                    judge_model=G_EVAL_MODEL,
                )
                rows.append({
                    "eval_id": row.get("eval_id"),
                    "system": system_name,
                    "question": row["question"],
                    "groundedness": judged["groundedness"],
                    "relevance": judged["relevance"],
                    "completeness": judged["completeness"],
                    "clarity": judged["clarity"],
                    "safety": judged["safety"],
                })
            except Exception as e:
                rows.append({
                    "eval_id": row.get("eval_id"),
                    "system": system_name,
                    "question": row["question"],
                    "groundedness": None,
                    "relevance": None,
                    "completeness": None,
                    "clarity": None,
                    "safety": None,
                    "error": str(e),
                })

            if i % 10 == 0:
                print(f"[INFO] {system_name}: {i}/{len(sub_df)} 완료")

        out_df = pd.DataFrame(rows)
        out_df.to_csv(detail_dir / f"g_eval_{system_name}.csv", index=False, encoding="utf-8-sig")

        valid_df = out_df.dropna(subset=["groundedness", "relevance", "completeness", "clarity", "safety"]).copy()
        if len(valid_df) > 0:
            summaries.append({
                "system": system_name,
                "groundedness_mean": float(valid_df["groundedness"].mean()),
                "relevance_mean": float(valid_df["relevance"].mean()),
                "completeness_mean": float(valid_df["completeness"].mean()),
                "clarity_mean": float(valid_df["clarity"].mean()),
                "safety_mean": float(valid_df["safety"].mean()),
                "overall_mean": float(valid_df[["groundedness", "relevance", "completeness", "clarity", "safety"]].mean(axis=1).mean()),
                "n": int(len(valid_df)),
            })
        else:
            summaries.append({
                "system": system_name,
                "groundedness_mean": None,
                "relevance_mean": None,
                "completeness_mean": None,
                "clarity_mean": None,
                "safety_mean": None,
                "overall_mean": None,
                "n": 0,
            })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "g_eval_summary_v1.csv", index=False, encoding="utf-8-sig")
    print("[INFO] G-EVAL 저장 완료")
    print(summary_df)
    return summary_df

# ============================================================
# [PROMPT 비교 결과] optional load
# ============================================================
def load_prompt_frames() -> Dict[str, pd.DataFrame]:
    prompt_frames = {}

    a_df = load_result_df(PROMPT_A_RESULTS_JSONL_PATH)
    b_df = load_result_df(PROMPT_B_RESULTS_JSONL_PATH)
    c_df = load_result_df(PROMPT_C_RESULTS_JSONL_PATH)

    if not a_df.empty:
        prompt_frames["prompt_A_grounded_strict"] = a_df
    if not b_df.empty:
        prompt_frames["prompt_B_medical_consult"] = b_df
    if not c_df.empty:
        prompt_frames["prompt_C_medical_safe"] = c_df

    return prompt_frames

# ============================================================
# [MAIN]
# ============================================================
def main():
    load_dotenv(ENV_PATH)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")

    client = OpenAI(api_key=openai_api_key)

    _, doc_text_lookup = prepare_base_data()

    system_frames = {
        "naive_dense_only": load_result_df(NAIVE_RESULTS_JSONL_PATH),
        "hybrid_dense_bm25_meta": load_result_df(HYBRID_RESULTS_JSONL_PATH),
        "reranked_hybrid_compressed": load_result_df(RERANK_RESULTS_JSONL_PATH),
    }
    system_frames = {k: v for k, v in system_frames.items() if not v.empty}

    print("[INFO] 시스템 프레임 로드 완료")
    for k, v in system_frames.items():
        print(f" - {k}: {v.shape}")

    if RUN_RAGAS:
        run_ragas(system_frames, doc_text_lookup)

    if RUN_G_EVAL:
        run_g_eval(system_frames, doc_text_lookup, client)

        prompt_frames = load_prompt_frames()
        if len(prompt_frames) > 0:
            print("\n[INFO] Prompt 비교 프레임 로드 완료")
            for k, v in prompt_frames.items():
                print(f" - {k}: {v.shape}")
            run_g_eval(prompt_frames, doc_text_lookup, client)

    print("\n[INFO] ragas_g_eval.py 실행 완료")


if __name__ == "__main__":
    main()
