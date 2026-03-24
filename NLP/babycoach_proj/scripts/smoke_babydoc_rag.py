"""
Smoke test for BabyDoc RAG + agent.

환경: conda 환경 `aiffel` 활성화 후 실행하세요.

  conda activate aiffel

실행 (프로젝트 루트 babycoach_proj 기준):

  set PYTHONPATH=%CD%   # Windows cmd
  set BABYCOACH_LLM_MOCK=1
  python scripts/smoke_babydoc_rag.py

PowerShell:

  $env:PYTHONPATH = (Get-Location).Path
  $env:BABYCOACH_LLM_MOCK = "1"
  python scripts/smoke_babydoc_rag.py
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("BABYCOACH_LLM_MOCK", "1")

from app.agents.babydoc_agent import BabyDocAgent
from app.services.rag_service import BabyDocRAGService


def main() -> None:
    svc = BabyDocRAGService()
    agent = BabyDocAgent(rag_service=svc)

    cases = [
        "가와사키병 증상 알려줘",
        "가와사키병 예방법 알려줘",
        "완전히없는질환XYZ123 원인이 뭐야",
    ]

    for q in cases:
        r = svc.retrieve(q)
        o = agent.answer(q)
        print("---")
        print("Q:", q)
        print("retrieve success:", r.get("success"), "top_score:", r.get("top_score"))
        print("agent status:", o.get("status"))
        print("doc_ids:", r.get("retrieved_doc_ids"))


if __name__ == "__main__":
    main()
