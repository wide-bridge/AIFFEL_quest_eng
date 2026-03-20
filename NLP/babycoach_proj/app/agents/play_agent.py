from __future__ import annotations

from typing import List

from ..state import BabyCoachState


def _ratio_label(r: float) -> str:
    if r <= 0.35:
        return "부모 주도"
    if r >= 0.7:
        return "아이 주도"
    return "중간"


def play_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Play node (state -> state).

    Creates `interaction_summary` and `play_suggestions`.
    """

    play_types = state.get("play_types", []) or []
    focus = int(state.get("focus_minutes", 0))
    repeat_count = int(state.get("repeat_count", 0))
    child_led_ratio = float(state.get("child_led_ratio", 0.5))
    refusal = bool(state.get("refusal", False))

    ratio_label = _ratio_label(child_led_ratio)

    # Suggest based on focus and repetition.
    suggestions: List[str] = []
    if refusal:
        suggestions.append("놀이를 강요하지 말고, 1~2분 안에 끝나는 '짧은 루틴'부터 다시 시작해보세요.")
    if focus <= 5:
        suggestions.append("짧게 주고받는 놀이(예: 다시 흔들기/다시 넣기)를 2~3회 반복해보세요. (목표: 집중 시작)")
    else:
        suggestions.append("지금의 집중 흐름을 이어서, 같은 테마로 '변형을 한 가지'만 추가해보세요.")

    if repeat_count >= 5:
        suggestions.append("아이의 반복 선호를 살려, 같은 동작을 하되 소리/색/재질 같은 감각 요소를 아주 조금 바꿔주세요.")

    if not play_types:
        suggestions.insert(0, f"오늘은 {ratio_label} 느낌에 맞춰 촉감/쌓기 중 하나를 골라 시작해보세요.")
    else:
        suggestions.insert(0, f"이미 하는 놀이({', '.join(play_types[:2])})를 유지하고, '다음 단계'만 아주 작게 제안해요.")

    # interaction_summary is used as "play notes" in UI to avoid separate bot nodes.
    interaction_summary = (
        f"놀이 상태를 보면 focus {focus}분, 반복 {repeat_count}회, 아이 주도 비율은 {child_led_ratio}({ratio_label})예요. "
        "그래서 오늘은 '짧게-반복-작은 변형' 흐름을 우선했어요."
    )

    return {
        **state,
        "interaction_summary": interaction_summary,
        "play_suggestions": suggestions[:2],
    }

