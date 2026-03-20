from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BabyCoachInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Note: PoC 2차 expanded fields. They are used for ranking/LLM context.
    age_months: int = Field(ge=0, le=36)
    weight_kg: float = Field(ge=0, le=50)

    allergies: List[str] = Field(default_factory=list)
    notes: str = ""

    protein_count_3d: int = Field(ge=0, le=3)
    vegetable_count_3d: int = Field(ge=0, le=3)
    food_diversity_3d: int = Field(ge=1, le=10)
    meal_refusal: bool = False
    reaction_flags: List[str] = Field(default_factory=list)

    play_types: List[str] = Field(default_factory=list)
    focus_minutes: int = Field(ge=0, le=30)
    repeat_count: int = Field(ge=0, le=10)
    child_led_ratio: float = Field(ge=0.0, le=1.0)
    refusal: bool = False
    parent_note: str = ""

    touch_count: int = Field(ge=0, le=10)
    labeling_count: int = Field(ge=0, le=10)
    joint_attention_count: int = Field(ge=0, le=10)
    responsive_turns: int = Field(ge=0, le=10)
    flat_response: bool = False

    parent_query: str = ""

    # --- PoC 2차 input expansion ---
    food_tag: str = ""
    meal_reaction: str = ""
    play_focus_level: str = ""


class RecommendResponse(BaseModel):
    final_output: Dict[str, Any]


class ChatRequest(BaseModel):
    final_output: Dict[str, Any]
    state_summary: Optional[str] = None
    user_message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    assistant_message: str

