from __future__ import annotations

from ..llm import generate_explanation
from ..state import BabyCoachState


def explanation_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Explanation node (state -> state).

    Uses GPT-5 mini to generate `explanation`.
    """

    explanation = generate_explanation(state)
    return {**state, "explanation": explanation}

