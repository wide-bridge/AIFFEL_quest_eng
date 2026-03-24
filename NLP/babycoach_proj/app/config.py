from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Requirement: load_dotenv uses the fixed path below.
_DOTENV_PATH = r"D:\PyProject\env_keys\.env"
load_dotenv(_DOTENV_PATH)


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


OPENAI_MODEL = "gpt-5-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# PoC requirement: allow end-to-end checks without any .env configuration.
# If BABYCOACH_LLM_MOCK is not explicitly set, default to mock mode when OPENAI_API_KEY is missing.
_mock_env = os.getenv("BABYCOACH_LLM_MOCK")
if _mock_env is None:
    BABYCOACH_LLM_MOCK = not bool(OPENAI_API_KEY)
else:
    BABYCOACH_LLM_MOCK = _env_flag("BABYCOACH_LLM_MOCK", default=_mock_env)


# BabyDoc RAG: paths resolved from this package (no absolute project paths in service code).
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent

BABYDOC_RAG_DOCS_PATH = _PROJECT_ROOT / "rag" / "data" / "rag_docs_v4.jsonl"
BABYDOC_EMBEDDINGS_CACHE_PATH = _PROJECT_ROOT / "rag" / "cache" / "doc_embeddings.pkl"
BABYDOC_DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BABYDOC_RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BABYDOC_LLM_MODEL = "gpt-4o-mini"


def require_openai_api_key() -> str:
    """
    Return OPENAI_API_KEY or raise a clear error.

    For smoke tests and local dev, you can set BABYCOACH_LLM_MOCK=1.
    """

    if BABYCOACH_LLM_MOCK:
        return "mock-api-key"
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create an .env file at D:\\PyProject\\env_keys\\.env "
            "with OPENAI_API_KEY set, or run with BABYCOACH_LLM_MOCK=1."
        )
    return OPENAI_API_KEY

