from __future__ import annotations

import os
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

