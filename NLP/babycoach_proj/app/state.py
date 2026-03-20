from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class BabyCoachState(TypedDict, total=False):
    # --- Input fields ---
    age_months: int
    weight_kg: float
    allergies: List[str]
    notes: str

    protein_count_3d: int
    vegetable_count_3d: int
    food_diversity_3d: int
    meal_refusal: bool
    reaction_flags: List[str]

    play_types: List[str]
    focus_minutes: int
    repeat_count: int
    child_led_ratio: float
    refusal: bool
    parent_note: str

    touch_count: int
    labeling_count: int
    joint_attention_count: int
    responsive_turns: int
    flat_response: bool

    parent_query: str

    # --- Input expansion (PoC 2차) ---
    food_tag: str
    meal_reaction: str
    play_focus_level: str

    # --- Intermediate summaries ---
    nutrition_summary: str
    interaction_summary: str
    epigenetic_summary: str
    growth_points: List[str]

    spoon_suggestions: List[str]
    play_suggestions: List[str]
    rank_tags: List[str]
    ranker_reason: str

    nudge_message: str
    explanation: str

    # --- Final output for API/UI ---
    final_output: Dict[str, Any]
    chat_context_summary: str


def build_state_from_input(raw: Dict[str, Any]) -> BabyCoachState:
    """
    Convert raw UI/API input into a normalized `BabyCoachState`.

    - Keeps DB-free end-to-end execution.
    - Applies reasonable defaults for missing expanded fields.
    """

    state: Dict[str, Any] = dict(raw)

    # Normalize list-like fields.
    allergies = state.get("allergies", []) or []
    if allergies == ["없음"]:
        allergies = []
    state["allergies"] = allergies

    reaction_flags = state.get("reaction_flags", []) or []
    if "없음" in reaction_flags:
        reaction_flags = []
    state["reaction_flags"] = reaction_flags

    # Defaults for expansion fields.
    state.setdefault("food_tag", "")
    state.setdefault("meal_reaction", "")
    state.setdefault("play_focus_level", "")

    # If play_focus_level is missing, derive from focus_minutes.
    if not state.get("play_focus_level"):
        focus_minutes = state.get("focus_minutes")
        try:
            focus_minutes_i = int(focus_minutes) if focus_minutes is not None else 0
        except (TypeError, ValueError):
            focus_minutes_i = 0
        if focus_minutes_i <= 5:
            state["play_focus_level"] = "낮음"
        elif focus_minutes_i <= 15:
            state["play_focus_level"] = "중간"
        else:
            state["play_focus_level"] = "높음"

    # If meal_reaction is missing, derive from meal_refusal + reaction_flags.
    if not state.get("meal_reaction"):
        meal_refusal = bool(state.get("meal_refusal", False))
        reaction_flags = state.get("reaction_flags", [])
        if meal_refusal:
            state["meal_reaction"] = "거부 신호"
        elif reaction_flags:
            state["meal_reaction"] = "조심"
        else:
            state["meal_reaction"] = "괜찮아요"

    # Ensure mandatory numeric defaults exist for downstream agents.
    state.setdefault("age_months", 0)
    state.setdefault("weight_kg", 0.0)

    state.setdefault("protein_count_3d", 0)
    state.setdefault("vegetable_count_3d", 0)
    state.setdefault("food_diversity_3d", 1)
    state.setdefault("meal_refusal", False)
    state.setdefault("reaction_flags", [])

    state.setdefault("play_types", [])
    state.setdefault("focus_minutes", 0)
    state.setdefault("repeat_count", 0)
    state.setdefault("child_led_ratio", 0.0)
    state.setdefault("refusal", False)
    state.setdefault("parent_note", "")

    state.setdefault("touch_count", 0)
    state.setdefault("labeling_count", 0)
    state.setdefault("joint_attention_count", 0)
    state.setdefault("responsive_turns", 0)
    state.setdefault("flat_response", False)

    state.setdefault("parent_query", "")

    return state  # type: ignore[return-value]

