from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import (
    BABYDOC_DENSE_MODEL_NAME,
    BABYDOC_EMBEDDINGS_CACHE_PATH,
    BABYDOC_RAG_DOCS_PATH,
    BABYDOC_RERANKER_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Hybrid fusion weights (spec).
_WEIGHT_DENSE = 0.45
_WEIGHT_BM25 = 0.40
_WEIGHT_META = 0.15

# OOD: hybrid only (not cross-encoder). If question names no known disease_kor, BM25+minmax
# can still spike; gate with raw max cosine (same dense model, no minmax).
THRESHOLD = 0.45
_MAX_COS_WHEN_NO_DISEASE = 0.47  # tuned: OOD fake ~0.445; generic "아기 열이 나요" ~0.496

_TOP_N_BEFORE_RERANK = 8
_TOP_K_AFTER_RERANK = 3
_MAX_RERANK_CHARS = 2000
_MAX_COMPRESSED_TOTAL_CHARS = 4500
_MAX_COMPRESSED_PER_DOC = 1800


def _tokenize_for_bm25(text: str) -> List[str]:
    """Lightweight tokenization for BM25 (Korean word chunks + alnum tokens)."""

    if not text:
        return []
    lowered = text.lower()
    parts = re.findall(r"[가-힣]+|[a-z0-9]+", lowered)
    return [p for p in parts if len(p) > 1 or (len(p) == 1 and p.isalpha())]


def _tokenize_overlap(text: str) -> set[str]:
    """Token set for overlap scoring (slightly looser than BM25)."""

    return set(_tokenize_for_bm25(text))


def extract_disease_kor(question: str, disease_candidates: Sequence[str]) -> Optional[str]:
    """
    Return disease_kor if the question explicitly contains that label.

    Longer names are checked first to reduce partial matches.
    """

    q = (question or "").strip()
    if not q:
        return None
    uniq = sorted({d for d in disease_candidates if d}, key=len, reverse=True)
    for name in uniq:
        if name in q:
            return name
    return None


def extract_intention(question: str) -> Optional[str]:
    """
    Rule-based intention tag aligned with corpus `intention` values.

    Order matters: longer / more specific keywords first.
    """

    q = question or ""
    # Spec order: 약물 → 예방 → 원인 → 진단 → 증상 → 치료 (longer keywords first within each group).
    rules: List[Tuple[str, Tuple[str, ...]]] = [
        ("약물", ("약물", "복용", "처방", "약")),
        ("예방", ("예방법", "예방", "위생")),
        ("원인", ("원인", "이유", "왜")),
        ("진단", ("진단", "검사")),
        ("증상", ("증상", "발진", "열")),
        ("치료", ("치료법", "치료")),
    ]
    for label, kws in rules:
        for kw in kws:
            if kw and kw in q:
                return label
    return None


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo < 1e-12:
        return np.ones_like(scores) * 0.5
    return (scores - lo) / (hi - lo)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。])\s+", text.replace("\n", " "))
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if len(s) >= 8:
            out.append(s)
    if not out and text.strip():
        return [text.strip()[:_MAX_COMPRESSED_PER_DOC]]
    return out


def compress_context(question: str, full_text: str, max_chars: int = _MAX_COMPRESSED_PER_DOC) -> str:
    """
    Sentence-level compression: pick sentences with token overlap vs. question,
    include neighboring sentences for local coherence.
    """

    sentences = _split_sentences(full_text)
    if not sentences:
        return ""
    q_tok = _tokenize_overlap(question)
    if not q_tok:
        joined = " ".join(sentences[:5])
        return joined[:max_chars]

    scored: List[Tuple[int, int, str]] = []
    for i, s in enumerate(sentences):
        st = _tokenize_overlap(s)
        overlap = len(q_tok & st)
        scored.append((overlap, i, s))
    scored.sort(key=lambda x: (-x[0], x[1]))

    seed_indices = [i for (ov, i, _) in scored[:15] if ov > 0]
    if not seed_indices:
        seed_indices = [i for (_, i, _) in scored[:5]]

    picked: set[int] = set()
    for i in seed_indices:
        for j in range(max(0, i - 1), min(len(sentences), i + 2)):
            picked.add(j)

    ordered = sorted(picked)
    buf: List[str] = []
    total = 0
    for i in ordered:
        chunk = sentences[i].strip()
        if total + len(chunk) + 1 > max_chars:
            break
        buf.append(chunk)
        total += len(chunk) + 1
    text_out = " ".join(buf).strip()
    if not text_out:
        text_out = " ".join(sentences[:3])[:max_chars]
    return text_out[:max_chars]


def _fail_retrieval(question: str, top_score: float) -> Dict[str, Any]:
    return {
        "success": False,
        "question": question,
        "retrieved_doc_ids": [],
        "retrieved_docs": [],
        "compressed_context": "",
        "top_score": float(top_score),
    }


class BabyDocRAGService:
    """
    BabyDoc retrieval: hybrid dense + BM25, metadata boost, cross-encoder rerank,
    and sentence overlap compression.
    """

    def __init__(
        self,
        *,
        jsonl_path: Optional[Union[str, Path]] = None,
        dense_model_name: str = BABYDOC_DENSE_MODEL_NAME,
        reranker_model_name: str = BABYDOC_RERANKER_MODEL_NAME,
        embeddings_cache_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._jsonl_path = Path(jsonl_path) if jsonl_path is not None else Path(BABYDOC_RAG_DOCS_PATH)
        self._dense_model_name = dense_model_name
        self._reranker_model_name = reranker_model_name
        self._embeddings_cache_path = Path(embeddings_cache_path) if embeddings_cache_path is not None else Path(
            BABYDOC_EMBEDDINGS_CACHE_PATH
        )

        self._docs: List[Dict[str, Any]] = []
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []
        self._doc_embeddings: Optional[np.ndarray] = None
        self._bi_encoder = None
        self._cross_encoder = None
        self._disease_universe: List[str] = []

        try:
            self._load_corpus()
            self._init_retriever_models()
        except Exception as exc:
            logger.exception("BabyDocRAGService init failed: %s", exc)
            raise

    def _load_corpus(self) -> None:
        path = self._jsonl_path
        if not path.is_file():
            raise FileNotFoundError(f"RAG jsonl not found: {path}")

        docs: List[Dict[str, Any]] = []
        diseases: set[str] = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = row.get("doc_id")
                dk = (row.get("disease_kor") or "").strip()
                intention = (row.get("intention") or "").strip()
                full_text = (row.get("full_text") or "").strip()
                if not doc_id or not full_text:
                    continue
                docs.append(
                    {
                        "doc_id": doc_id,
                        "disease_kor": dk,
                        "intention": intention,
                        "full_text": full_text,
                    }
                )
                if dk:
                    diseases.add(dk)

        if not docs:
            raise RuntimeError(f"No valid documents loaded from {path}")

        self._docs = docs
        self._disease_universe = sorted(diseases)
        self._tokenized_corpus = [_tokenize_for_bm25(d["full_text"]) for d in docs]
        logger.info("Loaded %d BabyDoc chunks from %s", len(docs), path)

    def _init_retriever_models(self) -> None:
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.debug("BM25 index ready (%d docs)", len(self._docs))

        self._bi_encoder = SentenceTransformer(self._dense_model_name)
        texts = [d["full_text"] for d in self._docs]
        cache_path = self._embeddings_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        loaded: Optional[np.ndarray] = None
        if cache_path.is_file():
            try:
                with cache_path.open("rb") as f:
                    loaded = pickle.load(f)
                if not isinstance(loaded, np.ndarray) or loaded.shape[0] != len(self._docs):
                    logger.warning("Embedding cache ignored (shape mismatch with corpus).")
                    loaded = None
                else:
                    logger.info("Loaded doc embeddings from cache %s", cache_path)
            except Exception as exc:
                logger.warning("Embedding cache load failed (%s), recomputing.", exc)
                loaded = None

        if loaded is None:
            self._doc_embeddings = self._bi_encoder.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            try:
                with cache_path.open("wb") as f:
                    pickle.dump(self._doc_embeddings, f)
                logger.info("Saved doc embeddings to %s", cache_path)
            except Exception as exc:
                logger.warning("Could not write embedding cache: %s", exc)
        else:
            self._doc_embeddings = loaded

        logger.debug("Dense embeddings ready shape=%s", getattr(self._doc_embeddings, "shape", None))

        self._cross_encoder = CrossEncoder(self._reranker_model_name)
        logger.debug("Cross-encoder loaded: %s", self._reranker_model_name)

    def _metadata_boost(self, doc: Dict[str, Any], disease_hit: Optional[str], intention_hit: Optional[str]) -> int:
        boost = 0
        if disease_hit and (doc.get("disease_kor") or "") == disease_hit:
            boost += 1
        if intention_hit and (doc.get("intention") or "") == intention_hit:
            boost += 1
        return boost

    def retrieve(self, question: str) -> Dict[str, Any]:
        """
        Run hybrid retrieval + rerank + compression.

        Uses hybrid score max for OOD threshold only (not cross-encoder).
        """

        q = (question or "").strip()

        def _ok(
            retrieved_doc_ids: List[str],
            retrieved_docs: List[Dict[str, Any]],
            compressed_context: str,
            top_score: float,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "question": q,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_docs": retrieved_docs,
                "compressed_context": compressed_context,
                "top_score": float(top_score),
            }

        try:
            if not q or not self._docs:
                return _fail_retrieval(q, 0.0)

            disease_hit = extract_disease_kor(q, self._disease_universe)
            intention_hit = extract_intention(q)
            logger.debug("extracted disease=%r intention=%r", disease_hit, intention_hit)

            q_tokens = _tokenize_for_bm25(q)
            bm25_scores = np.array(self._bm25.get_scores(q_tokens), dtype=np.float64)
            bm25_norm = _minmax_norm(bm25_scores)

            q_emb = self._bi_encoder.encode(
                [q],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            doc_emb = self._doc_embeddings
            dense_sims = (doc_emb @ q_emb[0].T).reshape(-1)
            dense_norm = _minmax_norm(dense_sims)

            hybrid = np.zeros(len(self._docs), dtype=np.float64)
            hybrid_nometa = np.zeros(len(self._docs), dtype=np.float64)
            for i, doc in enumerate(self._docs):
                meta = self._metadata_boost(doc, disease_hit, intention_hit)
                dn = float(dense_norm[i])
                bn = float(bm25_norm[i])
                hybrid_nometa[i] = _WEIGHT_DENSE * dn + _WEIGHT_BM25 * bn
                hybrid[i] = hybrid_nometa[i] + _WEIGHT_META * float(meta)

            top_score = float(np.max(hybrid))
            max_cos = float(np.max(dense_sims))

            if disease_hit is None:
                if max_cos < _MAX_COS_WHEN_NO_DISEASE:
                    logger.debug(
                        "OOD gate (no disease_kor in question): max_cos=%s < %s",
                        max_cos,
                        _MAX_COS_WHEN_NO_DISEASE,
                    )
                    return _fail_retrieval(q, top_score)
            elif top_score < THRESHOLD:
                logger.debug(
                    "OOD / low hybrid: top_score=%s < %s disease_hit=%r",
                    top_score,
                    THRESHOLD,
                    disease_hit,
                )
                return _fail_retrieval(q, top_score)

            top_n = min(_TOP_N_BEFORE_RERANK, len(self._docs))
            top_idx = np.argsort(-hybrid)[:top_n]
            idx_list = [int(i) for i in top_idx]

            pairs: List[List[str]] = []
            for i in idx_list:
                doc = self._docs[i]
                passage = (doc.get("full_text") or "")[:_MAX_RERANK_CHARS]
                pairs.append([q, passage])

            ce_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
            ce_scores = np.asarray(ce_scores, dtype=np.float64).reshape(-1)
            order = np.argsort(-ce_scores)
            top_k = min(_TOP_K_AFTER_RERANK, len(idx_list))
            final_local = [idx_list[int(j)] for j in order[:top_k]]

            retrieved_docs: List[Dict[str, Any]] = []
            compressed_parts: List[str] = []
            total_comp = 0

            for rank, i in enumerate(final_local, start=1):
                doc = self._docs[i]
                comp = compress_context(q, doc["full_text"], max_chars=_MAX_COMPRESSED_PER_DOC)
                header = (
                    f"[{rank}] doc_id={doc['doc_id']} disease_kor={doc.get('disease_kor', '')} "
                    f"intention={doc.get('intention', '')}"
                )
                block = f"{header}\n{comp}"
                if total_comp + len(block) + 2 > _MAX_COMPRESSED_TOTAL_CHARS:
                    comp = comp[: max(200, _MAX_COMPRESSED_TOTAL_CHARS - total_comp - len(header) - 10)]
                    block = f"{header}\n{comp}"
                compressed_parts.append(block)
                total_comp += len(block) + 2

                retrieved_docs.append(
                    {
                        "doc_id": doc["doc_id"],
                        "disease_kor": doc.get("disease_kor"),
                        "intention": doc.get("intention"),
                        "compressed_excerpt": comp,
                    }
                )

            return _ok(
                [d["doc_id"] for d in retrieved_docs],
                retrieved_docs,
                "\n\n".join(compressed_parts).strip(),
                top_score,
            )

        except Exception as exc:
            logger.exception("retrieve() failed: %s", exc)
            return _fail_retrieval(q, 0.0)
