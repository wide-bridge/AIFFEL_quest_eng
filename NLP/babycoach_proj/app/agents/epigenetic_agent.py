from __future__ import annotations

from ..state import BabyCoachState


def epigenetic_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Epigenetic node (state -> state).

    PoC에서는 '실제 분자/의학적 주장'이 아니라,
    생활 루틴/스트레스 완화/감각 환경을 설명하는 요약을 만듭니다.
    """

    meal_refusal = bool(state.get("meal_refusal", False))
    refusal = bool(state.get("refusal", False))
    parent_note = state.get("parent_note", "") or ""

    epigenetic_summary = "오늘은 '자극을 크게 늘리기'보다 '예측 가능하게'를 우선하는 흐름이에요."
    if meal_refusal or refusal:
        epigenetic_summary = (
            "식사/놀이에서 거부 신호가 있을 수 있어서, 오늘은 자극을 줄이고 속도를 낮춰 "
            "예측 가능한 루틴(짧게-반복-끝내기)을 더 강조했어요."
        )

    if parent_note.strip():
        epigenetic_summary += f" 부모 메모도 참고해서: {parent_note.strip()}"

    return {**state, "epigenetic_summary": epigenetic_summary}

