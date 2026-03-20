from __future__ import annotations

from typing import List

from ..state import BabyCoachState


def growth_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Growth node (state -> state).

    Creates `growth_points` as "today's observation points".
    """

    focus = int(state.get("focus_minutes", 0))
    meal_refusal = bool(state.get("meal_refusal", False))
    responsive = int(state.get("responsive_turns", 0))
    flat = bool(state.get("flat_response", False))
    reaction_flags = state.get("reaction_flags", []) or []

    points: List[str] = []
    if meal_refusal:
        points.append("식사는 '한 입 크기 + 짧게'로 끝나는 흐름을 관찰해보세요.")
    else:
        points.append("오늘 추천 음식/텍스처를 1~2번만 시도했을 때의 표정/반응을 기록해보세요.")

    if focus <= 5:
        points.append("집중이 짧게 시작된다면, 2~3회 주고받고 바로 마무리되는지 확인해요.")
    else:
        points.append("지금 집중이 이어진다면 같은 테마에서 '변형 1개'만 추가해도 괜찮은지 확인해요.")

    if flat or responsive <= 2:
        points.append("오늘은 '반응에 맞춰 속도 낮추기'가 도움이 될 수 있어요: 멈추고 기다리는 순간을 늘려보세요.")
    else:
        points.append("주고받기가 잘 이어진다면, 아이의 반응이 나온 직후에 짧게 이름 붙여주는 걸 더 해보세요.")

    if reaction_flags and "없음" not in [f.strip() for f in reaction_flags]:
        points.append(f"반응 플래그({', '.join(reaction_flags)})가 있었다면 '조심스럽게' 관찰하는 관점을 유지해요.")

    return {**state, "growth_points": points[:3]}

