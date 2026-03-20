from __future__ import annotations

from ..state import BabyCoachState


def interaction_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Interaction node (state -> state).

    Creates `interaction_summary` (already used by play notes).
    """

    touch = int(state.get("touch_count", 0))
    labeling = int(state.get("labeling_count", 0))
    joint = int(state.get("joint_attention_count", 0))
    responsive = int(state.get("responsive_turns", 0))
    flat = bool(state.get("flat_response", False))

    # This node keeps it light: no judgement, only "today's tendency".
    tendency = "전반적으로 반응이 괜찮은 편" if not flat else "오늘은 전반적으로 반응이 적게 느껴질 수 있어요"

    summary = (
        f"스킨십/터치 {touch}회, 말 걸기 {labeling}회, 같이 보기 {joint}회, 주고받기 {responsive}회. "
        f"오늘의 경향: {tendency}. 그래서 상호작용은 '부담 낮게-짧게-반응에 맞춰' 조절하는 방향을 잡았어요."
    )

    return {**state, "interaction_summary": state.get("interaction_summary", summary)}

