from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..graph import run_recommendation
from ..schemas import RecommendResponse
from ..state import BabyCoachState, build_state_from_input


router = APIRouter(prefix="", tags=["recommend"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: Dict[str, Any]) -> RecommendResponse:
    """
    Run BabyCoach LangGraph and return `final_output`.

    PoC 2차 요구:
    - UI가 아래 중첩 payload를 보낼 수 있음
      { child_profile, spoon_input, play_input }
    - 기존 flat input도 호환하도록 병합 처리합니다.
    """

    try:
        merged: Dict[str, Any] = {}
        if isinstance(payload, dict) and "child_profile" in payload:
            merged.update(payload.get("child_profile") or {})
            merged.update(payload.get("spoon_input") or {})
            merged.update(payload.get("play_input") or {})
            if payload.get("parent_query"):
                merged["parent_query"] = payload.get("parent_query")
        else:
            merged = payload

        input_state: BabyCoachState = build_state_from_input(merged)
        final_state = run_recommendation(input_state)
        final_output = final_state.get("final_output", {})
        if not isinstance(final_output, dict):
            raise TypeError("final_output must be a dict")
        return RecommendResponse(final_output=final_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

