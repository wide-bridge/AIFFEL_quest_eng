from __future__ import annotations

import json
import difflib
import hashlib
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import BABYCOACH_LLM_MOCK, OPENAI_API_KEY, OPENAI_MODEL, require_openai_api_key
from .state import BabyCoachState


def _extract_output_text(response: Any) -> str:
    """
    Extract plain text from OpenAI Responses API response in a robust way.
    """

    if response is None:
        return ""

    # Newer SDKs often expose `output_text`.
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Fallback: try to locate text parts in `output`.
    output = getattr(response, "output", None)
    if isinstance(output, list):
        texts: List[str] = []
        for block in output:
            if isinstance(block, dict):
                content = block.get("content")
                if isinstance(content, list):
                    for part in content:
                        text = part.get("text") if isinstance(part, dict) else None
                        if isinstance(text, str) and text.strip():
                            texts.append(text)
        joined = "\n".join(texts).strip()
        if joined:
            return joined

    return ""


def _pick_primary_domain(state: BabyCoachState) -> str:
    """
    Choose one primary domain for the single-action nudge.

    Priority rule:
    - meal_refusal: 식사(spoon) 우선
    - refusal(놀이 거부): 놀이(play) 우선
    - rank_tags: 영양 > 놀이 > 상호작용
    - otherwise: growth
    """

    if bool(state.get("meal_refusal", False)):
        return "spoon"
    if bool(state.get("refusal", False)):
        return "play"

    rank_tags = state.get("rank_tags", []) or []
    if "영양" in rank_tags:
        return "spoon"
    if "놀이" in rank_tags:
        return "play"
    if "상호작용" in rank_tags:
        return "play"
    return "growth"


def _make_short_nudge(state: BabyCoachState) -> str:
    """
    Create a one-sentence nudge (target: 30~60 chars).
    Only one actionable item is allowed.
    """

    domain = _pick_primary_domain(state)
    meal_refusal = bool(state.get("meal_refusal", False))
    reaction_flags = state.get("reaction_flags", []) or []
    refusal = bool(state.get("refusal", False))

    if domain == "spoon":
        if meal_refusal or (reaction_flags and reaction_flags != ["없음"]):
            return "오늘은 단백질만 한 입 크기로 천천히 해보세요."
        return "오늘은 단백질 텍스처를 한 숟갈만 천천히 해보세요."

    if domain == "play":
        if refusal:
            return "오늘은 놀이를 1~2분만 짧게 다시 해보세요."
        return "오늘은 같은 놀이를 2번만 짧게 반복해 보세요."

    return "오늘은 관찰 포인트 1개만 체크해봐요."


def _mock_nudge_and_explanation(state: BabyCoachState) -> tuple[str, str]:
    """
    Deterministic mock outputs for smoke tests / local UI.

    - nudge_message: 1 sentence, single actionable item
    - explanation: 3~5 sentences, with (현재 상태/왜 중요한지/오늘 제안) clarity
    """

    nudge_message = _make_short_nudge(state)

    protein_stage = int(state.get("protein_count_3d", 0))
    veg_stage = int(state.get("vegetable_count_3d", 0))
    diversity = int(state.get("food_diversity_3d", 5))
    focus = int(state.get("focus_minutes", 0))
    repeat_count = int(state.get("repeat_count", 0))
    child_led_ratio = float(state.get("child_led_ratio", 0.5))
    meal_refusal = bool(state.get("meal_refusal", False))
    refusal = bool(state.get("refusal", False))

    domain = _pick_primary_domain(state)
    if domain == "spoon":
        current = f"현재 식사는 단백질 단계 {protein_stage}, 채소 단계 {veg_stage}, 다양성 {diversity}예요."
        why = "부드러운 텍스처로 시작하면 거부/불편을 줄이고 반복이 쉬워져요."
        today = f"오늘 제안: {nudge_message}"
    elif domain == "play":
        current = f"지금 놀이는 집중 {focus}분, 반복 {repeat_count}회, 아이 주도 비율 {child_led_ratio:.2f}예요."
        why = "짧게 반복하면 성공 경험이 쌓이고 아이 리듬에 맞추기가 쉬워져요."
        today = f"오늘 제안: {nudge_message}"
    else:
        current = "지금은 작은 관찰로 오늘의 흐름을 확인하기 좋은 타이밍이에요."
        why = "관찰 1개만 쌓이면 다음 시도에서 속도와 텍스처 조절이 더 쉬워져요."
        today = f"오늘 제안: {nudge_message}"

    # Keep first sentence natural: `current` already starts with "현재/지금 ..." per domain.
    # So we don't prefix with an extra "지금 상태:" label to avoid redundancy.
    explanation = f"{current} {why} {today}"
    return nudge_message, explanation


def _responses_create_text(prompt: str, *, system: str) -> str:
    """
    Call OpenAI Responses API and return extracted output text.
    """

    if BABYCOACH_LLM_MOCK:
        # This should be handled by higher-level mock wrappers, but keep it safe.
        return prompt[:200]

    api_key = require_openai_api_key()
    client = OpenAI(api_key=api_key)

    # Responses API supports `input` as plain text.
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        # Use system-like instruction by prepending for maximum compatibility.
        # (Responses API currently uses `input` roles; we keep it simple.)
        # If the SDK supports `instructions`, switch later.
    )

    text = _extract_output_text(resp)
    if text:
        return text
    # If extraction fails, return raw best-effort string.
    return str(resp).strip()[:2000]


def generate_nudge_message(state: BabyCoachState) -> str:
    """
    Generate `nudge_message` using GPT-5 mini (Responses API).
    """

    if BABYCOACH_LLM_MOCK:
        nudge, _ = _mock_nudge_and_explanation(state)
        return nudge

    domain = _pick_primary_domain(state)
    candidate_nudge = _make_short_nudge(state)

    # Keep prompt short but explicit about style constraints.
    prompt = (
        "너는 BabyCoach. 의료 진단처럼 말하지 말고, 환경/행동 제안 중심으로 답해.\n"
        "요구: 출력은 한국어 '한 문장'만. 길이 30~60자 수준.\n"
        "Spoon/Play/Growth를 섞지 말고, ranker 기준 최우선 행동 1개만 반영해.\n\n"
        f"입력 요약:\n"
        f"- age_months: {state.get('age_months')}\n"
        f"- weight_kg: {state.get('weight_kg')}\n"
        f"- meal_refusal: {state.get('meal_refusal')}\n"
        f"- reaction_flags: {state.get('reaction_flags')}\n"
        f"- play_types: {state.get('play_types')}\n"
        f"- child_led_ratio: {state.get('child_led_ratio')}\n"
        f"- refusal(놀이): {state.get('refusal')}\n"
        f"- parent_query: {state.get('parent_query')}\n\n"
        f"ranker 최우선 도메인: {domain}\n"
        f"후보(참조): {candidate_nudge}\n\n"
        "출력: 한 문장 nudge_message만 그대로."
    )
    return _responses_create_text(prompt, system="BabyCoach")


def generate_explanation(state: BabyCoachState) -> str:
    """
    Generate `explanation` using GPT-5 mini (Responses API).
    """

    if BABYCOACH_LLM_MOCK:
        _, explanation = _mock_nudge_and_explanation(state)
        return explanation

    domain = _pick_primary_domain(state)
    candidate_nudge = _make_short_nudge(state)

    prompt = (
        "너는 BabyCoach 설명 담당. 의료 진단/병명/확정적 표현은 금지하고, 부모가 안심할 수 있는 톤으로 작성해.\n"
        "요구: 한국어 3~5문장. 아래 3파트를 모두 포함해.\n"
        "- 상태(1문장)\n"
        "- 왜 중요한지(1문장)\n"
        "- 오늘 제안(1문장; nudge_message와 같은 행동 1개만)\n\n"
        f"ranker 최우선 도메인: {domain}\n"
        f"nudge_message(오늘 제안으로 사용할 행동): {candidate_nudge}\n\n"
        "출력은 3파트를 문장으로만 작성하고 줄바꿈은 자유롭게 허용."
    )
    return _responses_create_text(prompt, system="BabyCoach")


def generate_chat_reply(
    *,
    final_output: Dict[str, Any],
    user_message: str,
    state_summary: Optional[str] = None,
) -> str:
    """
    Generate BabyCoach chatbot reply.

    Requirements implemented:
    - LLM input includes: user_message, spoon_context, play_context, interaction_context, growth_context, final_coaching
    - Reply follows 4-step sentence structure (해석 → 맥락 → 발달 관점 → 오늘 행동 1개)
    - Regenerate if too similar to previous reply
    - No banned phrases
    - Honorific speech only
    """

    def strip_numbers(s: str) -> str:
        s = re.sub(r"\d+(?:[.,]\d+)?", "", s or "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def digits_to_korean(s: str) -> str:
        # Replace simple Arabic numerals to avoid digit-like 숫자 노출 in chat.
        mapping = {
            "0": "영",
            "1": "한",
            "2": "두",
            "3": "세",
            "4": "네",
            "5": "다섯",
            "6": "여섯",
            "7": "일곱",
            "8": "여덟",
            "9": "아홉",
            "10": "열",
        }

        # Longest keys first to keep "10" from being partially replaced.
        s = s or ""
        for k, v in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            s = s.replace(k, v)
        return s
    msg = user_message or ""

    # ---- Build contexts from `final_output` ----
    spoon = final_output.get("spoon") or {}
    play = final_output.get("play") or {}
    growth = final_output.get("growth") or {}
    nudge = final_output.get("nudge") or {}
    explanation = final_output.get("explanation") or {}

    spoon_context = {
        "suggestions": spoon.get("suggestions") or [],
        "notes": spoon.get("notes") or "",
    }
    play_context = {
        "suggestions": play.get("suggestions") or [],
    }
    interaction_context = {
        # In this PoC, play.notes is produced by the interaction agent summary.
        "notes": play.get("notes") or "",
    }
    growth_context = {
        "observation_points": growth.get("observation_points") or [],
    }
    final_coaching = {
        "nudge_message": nudge.get("nudge_message") or "",
        "explanation": explanation.get("explanation") or "",
    }

    # ---- Similarity / repetition guard ----
    banned_phrases = ["오늘 추천은", "오늘 제안은", "흐름을 바탕으로", "좋은 질문이야"]

    # Server-side memory (single-process PoC).
    global _CHAT_LAST_REPLY_CACHE  # created lazily below
    try:
        _CHAT_LAST_REPLY_CACHE  # type: ignore[name-defined]
    except NameError:
        _CHAT_LAST_REPLY_CACHE = {}  # type: ignore[misc]

    def _state_fingerprint() -> str:
        key_obj = {
            "state_summary": state_summary or "",
            "spoon": spoon_context,
            "play": play_context,
            "interaction": interaction_context,
            "growth": growth_context,
            "final_coaching": final_coaching,
        }
        raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a or "", b or "").ratio()

    def _contains_banned(s: str) -> bool:
        return any(p in (s or "") for p in banned_phrases)

    def _normalize_candidate(s: str) -> str:
        s = str(s or "").strip()
        # normalize punctuation for splitting/comparison
        s = s.replace("。", ".")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _extract_last_sentence(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", s)
        if len(parts) >= 2:
            return parts[-1].strip()
        # fallback: last ~30 chars
        return s[-30:].strip()

    def _extract_sentences(s: str) -> List[str]:
        # Split by common sentence-ending punctuation.
        raw = (s or "").strip().replace("。", ".")
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
        return parts

    def _mock_rewrite_nudge_sentence(nudge_message_in: str, variant: int) -> str:
        nm = digits_to_korean(nudge_message_in or "").strip()
        nm = re.sub(r"\s+", " ", nm)
        # If the nudge already starts with "오늘", vary the opening phrase.
        # Keep the actionable tail intact.
        rest = nm
        rest = re.sub(r"^오늘(은|도)\s*", "", rest)
        rest = re.sub(r"^\s*오늘\s*", "", rest)
        prefixes = ["오늘도 ", "이번엔 ", "지금은 ", "다음 시도엔 ", "우선 ", "오늘 한번은 "]
        prefix = prefixes[(variant + len(msg)) % len(prefixes)]
        # Ensure it ends politely.
        if rest.endswith(".") or rest.endswith("!"):
            return (prefix + rest).strip()
        return (prefix + rest).strip()

    def _mock_chat_reply(variant: int) -> str:
        # Light keyword steering; LLM-free mock still uses provided contexts.
        m = msg
        lower = m.lower()

        spoon_notes = strip_numbers((spoon_context.get("notes") or "").strip())
        interaction_notes = strip_numbers((interaction_context.get("notes") or "").strip())
        growth_points = growth_context.get("observation_points") or []
        growth_first = ""
        if isinstance(growth_points, list) and growth_points:
            growth_first = strip_numbers(str(growth_points[0] or ""))
        nudge_message = final_coaching.get("nudge_message") or ""

        # Keep snippets short to avoid template-like repetition.
        def _clip(s: str, n: int = 48) -> str:
            s = (s or "").strip()
            return s if len(s) <= n else s[: n - 1] + "…"

        q_snip = _clip(strip_numbers(m), 24)
        spoon_snip = _clip(spoon_notes, 60)
        interaction_snip = _clip(interaction_notes, 60)
        growth_snip = _clip(growth_first, 60)

        # Decide emphasis.
        intent = "general"
        if any(k in lower for k in ["잠", "수면", "자", "깨어", "밤", "뒤척"]):
            intent = "sleep"
        elif any(k in lower for k in ["밥", "먹", "식사", "이유식", "안 먹", "싫어", "잘 안"]):
            intent = "meal"
        elif any(k in lower for k in ["놀이", "장난감", "금방", "싫어", "안 해", "재미"]):
            intent = "play"
        elif any(k in lower for k in ["상호", "교감", "터치", "스킨십", "말 걸", "대화", "반응"]):
            intent = "interaction"

        # Sentence 1: interpret question with state context
        if intent == "meal":
            s1 = f"말씀하신 '{q_snip}' 고민은, 현재 식사 맥락({spoon_snip or '영양 단계'})과 함께 보면 이해가 쉬우세요."
            s2 = f"그래서 Spoon 쪽에서는 {spoon_snip or '부드럽게/천천히 접근'}을 기준으로 조절 방향을 잡아보는 걸 권해요."
        elif intent == "play":
            s1 = f"말씀하신 '{q_snip}' 고민은, 지금의 놀이 리듬과 아이 반응({interaction_snip or '관찰 포인트'})을 같이 보시면 좋아요."
            s2 = f"Play에서는 {interaction_snip or '짧게 반복하며 성공 경험을 쌓는 방식'}에 초점을 두는 편이예요."
        elif intent == "interaction":
            s1 = f"말씀하신 '{q_snip}' 고민은, 지금 아이가 보이는 신호({interaction_snip or '반응 패턴'})에 맞춰 풀어보면 좋아요."
            s2 = f"교감 맥락에서는 {interaction_snip or '신호를 확인하며 주고받는 속도 조절'}이 핵심이에요."
        elif intent == "sleep":
            s1 = f"말씀하신 '{q_snip}' 고민은, 아이에게 {growth_snip or '예측 가능한 흐름'}이 충분히 전달되는지 함께 확인해보면 좋아요."
            s2 = f"Growth에서는 {growth_snip or '안정감 있는 리듬'}을 중심으로 전환 부담을 낮추는 쪽으로 잡아요."
        else:
            s1 = f"말씀하신 '{q_snip}' 고민은 현재 상태 신호를 기준으로, 한 번에 크게 바꾸기보다 편안하게 맞춰보는 방식이 잘 맞아요."
            s2 = f"Spoon/Play/Growth 중에서 오늘은 {growth_snip or '반응을 우선으로 보는 관점'}을 더 강조해보는 게 좋아요."

        # Sentence 3: developmental viewpoint (short)
        s3_variants = [
            "아이에게는 예측 가능성과 편안함이 쌓일수록 다음 반응으로 이어지기 쉬우세요.",
            "발달 관점에서는 '성공 경험'과 '편안한 속도'가 학습 부담을 줄여줘요.",
            "지금의 소소한 조절이 관계의 안정과 다음 행동으로 이어지는 기반이 돼요.",
        ]
        s3 = s3_variants[(variant + len(msg)) % len(s3_variants)]

        # Sentence 4: one actionable suggestion (derived from nudge)
        s4 = _mock_rewrite_nudge_sentence(nudge_message, variant)
        s4 = s4.strip()
        if not s4:
            s4 = "오늘은 아이 신호에 맞춰 짧게 다시 시도해보세요."
        s4 = digits_to_korean(s4)

        # Ensure banned phrases are not included.
        candidate = _normalize_candidate(" ".join([s1, s2, s3, s4]))
        candidate = candidate.replace("오늘 추천은", "").replace("오늘 제안은", "")
        return candidate

    def _llm_chat_reply(payload: Dict[str, Any], last_reply: str, variant: int) -> str:
        # Build prompt with required input structure.
        prompt = (
            "너는 BabyCoach 코칭 봇입니다. 의료 진단/확정적 표현은 금지합니다.\n"
            "아래 출력 규칙을 반드시 지켜주세요.\n"
            "\n"
            "입력(JSON):\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "\n"
            "출력 규칙:\n"
            "1) 첫 문장: 사용자 질문을 현재 상태와 연결해서 해석하세요.\n"
            "2) 둘째 문장: Spoon / Play / 교감 / Growth 중 관련 맥락을 짧게 설명하세요.\n"
            "3) 셋째 문장: 발달 관점에서 왜 중요한지 짧게 설명하세요.\n"
            "4) 넷째 문장: 오늘 행동 1개만, 실제로 해볼 수 있게 제안하세요.\n"
            "\n"
            "말투: 반드시 존댓말만 사용하세요.\n"
            "형식: 정확히 4문장으로만 작성하세요. 각 문장은 마침표(.)로 끝내세요.\n"
            "금지 문구: '오늘 추천은', '오늘 제안은', '흐름을 바탕으로', '좋은 질문이야' 는 절대 포함하지 마세요.\n"
            "반복 금지: 이전 답변과 유사한 문장/표현을 최대한 피하세요.\n"
            f"이전 답변(비교용): {last_reply}\n"
            f"변형 힌트: {variant}\n"
        )
        out = _responses_create_text(prompt, system="BabyCoach")
        return _normalize_candidate(out)

    # ---- Prepare payload for LLM / mock ----
    state_key = _state_fingerprint()
    last_reply = _CHAT_LAST_REPLY_CACHE.get(state_key, "")

    llm_payload = {
        "user_message": msg,
        "spoon_context": spoon_context,
        "play_context": play_context,
        "interaction_context": interaction_context,
        "growth_context": growth_context,
        "final_coaching": final_coaching,
    }

    # Try multiple candidates to satisfy similarity/repetition constraints.
    last_candidate = ""
    for attempt in range(3):
        candidate = ""
        if BABYCOACH_LLM_MOCK:
            candidate = _mock_chat_reply(attempt)
        else:
            candidate = _llm_chat_reply(llm_payload, last_reply, attempt)

        candidate = _normalize_candidate(candidate)
        last_candidate = candidate

        if _contains_banned(candidate):
            continue

        if last_reply:
            if _similarity(candidate, last_reply) >= 0.7:
                continue

            # Disallow identical last sentence (direct repetition)
            if _extract_last_sentence(candidate) == _extract_last_sentence(last_reply):
                continue

            # Disallow direct sentence overlap (stronger repetition guard)
            prev_sents = set(_extract_sentences(last_reply))
            cand_sents = set(_extract_sentences(candidate))
            if prev_sents and cand_sents and (prev_sents & cand_sents):
                continue

        # If candidate doesn't look like 4 sentences, still accept but lightly check.
        # We enforce "4 sentences" via prompt; mock always does 4.
        _CHAT_LAST_REPLY_CACHE[state_key] = candidate
        return candidate

    result = last_candidate or "말씀하신 고민은 현재 상태 신호를 기준으로 편안하게 한 번만 조절해보시면 좋아요."
    if _contains_banned(result):
        for p in banned_phrases:
            result = result.replace(p, "")
        result = _normalize_candidate(result)
    if result:
        _CHAT_LAST_REPLY_CACHE[state_key] = result
    return result

