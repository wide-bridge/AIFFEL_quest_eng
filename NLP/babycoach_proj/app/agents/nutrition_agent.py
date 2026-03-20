from __future__ import annotations

from typing import List

from ..state import BabyCoachState


def _has_allergy(allergies: List[str], name: str) -> bool:
    return any(a.strip().lower() == name.strip().lower() for a in allergies)


def nutrition_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Nutrition node (state -> state).

    Creates `nutrition_summary` and `spoon_suggestions`.
    """

    allergies = state.get("allergies", []) or []
    reaction_flags = state.get("reaction_flags", []) or []

    protein_stage = int(state.get("protein_count_3d", 0))
    veg_stage = int(state.get("vegetable_count_3d", 0))
    diversity = int(state.get("food_diversity_3d", 5))
    meal_refusal = bool(state.get("meal_refusal", False))

    # Basic ingredient pools with allergy-aware filtering.
    protein_candidates = ["부드러운 두부", "렌틸(잘 익혀 으깨기)", "달걀흰자(가능 시)", "살코기(잘게 다지기)"]
    egg_blocked = _has_allergy(allergies, "달걀흰자") or _has_allergy(allergies, "달걀")
    milk_blocked = _has_allergy(allergies, "우유")
    if egg_blocked:
        protein_candidates = [p for p in protein_candidates if "달걀" not in p]
    if milk_blocked:
        protein_candidates = [p for p in protein_candidates if "우유" not in p]

    if protein_stage <= 1:
        protein_pick = protein_candidates[0]
    elif protein_stage == 2:
        protein_pick = protein_candidates[1] if len(protein_candidates) > 1 else protein_candidates[0]
    else:
        protein_pick = protein_candidates[min(2, len(protein_candidates) - 1)]

    veg_pick = "채소는 한 가지를 더 부드럽게(삶기/갈기)"
    if veg_stage <= 1:
        veg_pick = "채소는 익힌 한 가지를 더 부드럽게(삶기/갈기)"
    elif veg_stage == 2:
        veg_pick = "채소는 두 가지 조합을 아주 소량부터"
    else:
        veg_pick = "채소는 다양성을 유지하되 텍스처를 고르게"

    calm_suffix = ""
    if meal_refusal or (reaction_flags and "없음" not in [f.strip() for f in reaction_flags]):
        calm_suffix = " (거부/반응이 있다면 속도는 더 천천히, 한 입 크기로 시작해요)"

    spoon_suggestions: List[str] = []
    spoon_suggestions.append(f"단백질: {protein_pick} 처럼 '부드러운 텍스처'부터 시작{calm_suffix}")
    if diversity <= 4 or veg_stage <= 1:
        spoon_suggestions.append(f"채소: {veg_pick} (오늘은 '추가'보다 '편안함'을 우선)")
    else:
        spoon_suggestions.append(f"다양성: 기존 음식에 작은 변형(온도/농도/모양)만 더해보세요")

    # Nutrition summary is shown as notes in UI.
    nutrition_summary = (
        f"식사 단계(단백질 {protein_stage}, 채소 {veg_stage})와 다양성({diversity})을 기준으로 "
        f"오늘은 '부드럽게/작게/속도 천천히' 흐름을 추천했어요."
    )
    if reaction_flags:
        nutrition_summary += f" 반응 플래그가 있어 조심스럽게 접근하는 방향을 포함했어요: {reaction_flags}"

    return {
        **state,
        "nutrition_summary": nutrition_summary,
        "spoon_suggestions": spoon_suggestions[:2],
    }

