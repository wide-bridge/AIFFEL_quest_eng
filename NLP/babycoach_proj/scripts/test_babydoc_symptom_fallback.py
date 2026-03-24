"""BabyDocAgent: RAG failure + symptom vs insufficient (no full RAG load)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.babydoc_agent import BabyDocAgent  # noqa: E402


def main() -> None:
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = {"success": False}
    agent = BabyDocAgent(rag_service=mock_rag)

    with patch("app.agents.babydoc_agent.load_latest_baby_context", return_value={}):
        for q in ("열이 나요", "기침을 해요", "설사를 해요"):
            out = agent.answer(q)
            assert out["status"] == "symptom_guidance", (q, out)
            assert "정확한 진단과 치료는 의료진 상담이 필요합니다" in out["answer"], q
            assert "소아과 진료를 권장드립니다" in out["answer"], q
            assert "제공된 정보만으로는" not in out["answer"], q

        q_bad = "완전히없는질환XYZ123"
        out = agent.answer(q_bad)
        assert out["status"] == "insufficient_evidence", out
        assert "제공된 정보만으로는 정확히 답변하기 어렵습니다" in out["answer"], out

    ctx = {
        "baby_name": "전서연",
        "age_months": 12,
        "baby_state": "예민함",
        "concerns": "낮잠 불규칙",
        "parent_goals": "수면 안정",
        "growth_direction": "",
        "free_text": "",
    }
    with patch("app.agents.babydoc_agent.load_latest_baby_context", return_value=ctx):
        out = agent.answer("열이 나요")
        assert out["status"] == "symptom_guidance"
        assert "서연이" in out["answer"]
        assert "예민한 편" in out["answer"]

    print("test_babydoc_symptom_fallback: ok")


if __name__ == "__main__":
    main()
