from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..agents.babydoc_agent import BabyDocAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/babydoc", tags=["babydoc"])

_babydoc_agent: Optional[BabyDocAgent] = None


def get_babydoc_agent() -> BabyDocAgent:
    global _babydoc_agent
    if _babydoc_agent is None:
        _babydoc_agent = BabyDocAgent()
    return _babydoc_agent


class BabyDocChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


@router.post("/chat")
def babydoc_chat(body: BabyDocChatRequest) -> Dict[str, Any]:
    """
    BabyDoc RAG + LLM answer. Used by the 베이비닥 web tab.
    """

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    try:
        out = get_babydoc_agent().answer(q)
        return out
    except Exception as exc:
        logger.exception("babydoc_chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="BabyDoc 처리 중 오류가 발생했습니다.") from exc
