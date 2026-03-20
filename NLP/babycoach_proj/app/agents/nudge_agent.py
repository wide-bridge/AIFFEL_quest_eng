from __future__ import annotations

from ..llm import generate_nudge_message
from ..state import BabyCoachState


def nudge_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Nudge node (state -> state).

    Uses GPT-5 mini to generate `nudge_message`.
    """

    nudge_message = generate_nudge_message(state)
    return {**state, "nudge_message": nudge_message}

