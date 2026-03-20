from __future__ import annotations

from typing import Any, Dict

from .state import BabyCoachState


def format_final_output(state: BabyCoachState) -> Dict[str, Any]:
    """
    Build the final JSON payload returned by `/recommend`.
    """

    spoon = {
        "title": "Spoon",
        "suggestions": state.get("spoon_suggestions", []),
        "notes": state.get("nutrition_summary", ""),
    }
    play = {
        "title": "Play",
        "suggestions": state.get("play_suggestions", []),
        "notes": state.get("interaction_summary", ""),
    }
    growth = {
        "title": "Growth",
        "observation_points": state.get("growth_points", []),
    }
    nudge = {
        "title": "오늘의 한 문장 코칭",
        "nudge_message": state.get("nudge_message", ""),
        "tags": state.get("rank_tags", []),
    }
    explanation = {
        "title": "설명",
        "explanation": state.get("explanation", ""),
    }

    chat_context_summary = (
        f"월령 {state.get('age_months')}개월 / 체중 {state.get('weight_kg')}kg. "
        f"식사: 단백질 단계 {state.get('protein_count_3d')}, 채소 단계 {state.get('vegetable_count_3d')}, "
        f"다양성 {state.get('food_diversity_3d')}. "
        f"놀이: focus {state.get('focus_minutes')}분, 반복 {state.get('repeat_count')}, "
        f"아이 주도 비율 {state.get('child_led_ratio')}. "
        f"오늘 추천 한 줄 코칭: {state.get('nudge_message', '')}"
    )

    return {
        "spoon": spoon,
        "play": play,
        "growth": growth,
        "nudge": nudge,
        "explanation": explanation,
        "chat_context_summary": chat_context_summary,
    }

