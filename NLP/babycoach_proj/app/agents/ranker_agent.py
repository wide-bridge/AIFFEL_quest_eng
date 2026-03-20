from __future__ import annotations

from typing import List

from ..state import BabyCoachState


def ranker_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Ranker node (state -> state).

    For PoC: decides `rank_tags`, limits suggestions, and creates a lightweight `ranker_reason`.
    """

    rank_tags: List[str] = []
    protein_stage = int(state.get("protein_count_3d", 0))
    focus = int(state.get("focus_minutes", 0))
    responsive = int(state.get("responsive_turns", 0))
    meal_refusal = bool(state.get("meal_refusal", False))
    reaction_flags = state.get("reaction_flags", []) or []

    if protein_stage <= 1:
        rank_tags.append("영양")
    if focus <= 5 or state.get("repeat_count", 0) <= 2:
        rank_tags.append("놀이")
    if responsive <= 2 or state.get("flat_response", False):
        rank_tags.append("상호작용")
    if meal_refusal or any(f.strip() not in {"", "없음"} for f in reaction_flags):
        rank_tags.append("주의")

    if not rank_tags:
        rank_tags = ["일상"]

    ranker_reason = (
        f"입력 신호 중 우선순위를 '상태 체크' 관점으로 잡았어요: "
        f"protein_count_3d={protein_stage}, focus_minutes={focus}, responsive_turns={responsive}, "
        f"meal_refusal={meal_refusal}, reaction_flags={reaction_flags}."
    )

    spoon = state.get("spoon_suggestions", []) or []
    play = state.get("play_suggestions", []) or []

    # Keep UI compact: 1~2 suggestions each.
    spoon_suggestions = spoon[:2] if spoon else ["오늘은 한 가지를 아주 편안하게 시도해보세요."]
    play_suggestions = play[:2] if play else ["오늘은 2~3회 주고받는 짧은 놀이로 시작해보세요."]

    return {
        **state,
        "spoon_suggestions": spoon_suggestions,
        "play_suggestions": play_suggestions,
        "rank_tags": rank_tags[:4],
        "ranker_reason": ranker_reason,
    }

