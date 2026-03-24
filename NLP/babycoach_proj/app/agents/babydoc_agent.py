from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from openai import OpenAI

from ..baby_display_name import baby_call_name_for_coaching
from ..config import (
    BABYCOACH_LLM_MOCK,
    BABYDOC_LLM_MODEL,
    require_openai_api_key,
)
from ..services.rag_service import BabyDocRAGService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """당신은 BabyDoc 안내 도우미입니다.
역할: 아래에 제공된 '검색 문서 요약'만 근거로 보호자에게 쉬운 한국어 존댓말로 설명합니다.

필수 규칙:
1) 첫 문장에 질문에 대한 핵심 답을 먼저 말합니다.
2) 제공된 검색 문서에 근거가 없는 내용은 말하지 않습니다. 추측·확진 표현을 하지 않습니다.
3) 근거가 부족하면 반드시 이 문구를 포함합니다: "제공된 정보만으로는 정확히 답변하기 어렵습니다"
4) 응급·중증 가능성, 지속 고열, 호흡곤란, 의식 변화, 경련, 탈수 등 위험 신호가 질문에 해당하면
   답변에 반드시 포함합니다: "소아과 진료 권장"
5) 개별 처방·용량 결정은 하지 않고, 반드시 의료진 상담을 안내합니다.
"""

_INSUFFICIENT_ANSWER = (
    "제공된 정보만으로는 정확히 답변하기 어렵습니다. 아이 상태가 걱정되시면 소아과 진료를 권장드립니다."
)

# RAG에 걸리지 않는 일반 증상 질문용 (단순 포함 판별).
_SYMPTOM_KEYWORDS = (
    "열",
    "기침",
    "발진",
    "설사",
    "구토",
    "콧물",
    "복통",
    "변비",
)


def load_latest_baby_context() -> dict:
    """
    SQLite에서 최신 아기 프로필·맥락을 읽습니다.
    실패 시 빈 dict.
    """
    try:
        from ..db import get_connection, init_db

        init_db()
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    p.name,
                    p.age_months,
                    c.happiness
                FROM baby_profile p
                LEFT JOIN baby_context c ON c.baby_id = p.id
                ORDER BY p.id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return {}

        happy_raw = row["happiness"]
        happy: dict = {}
        if happy_raw:
            try:
                happy = (
                    json.loads(happy_raw)
                    if isinstance(happy_raw, str)
                    else (happy_raw if isinstance(happy_raw, dict) else {})
                )
            except Exception:
                happy = {}
        if not isinstance(happy, dict):
            happy = {}

        def _join_list(key: str) -> str:
            v = happy.get(key) or []
            if isinstance(v, list):
                return ", ".join(str(x) for x in v if x is not None and str(x).strip())
            return str(v).strip() if v else ""

        return {
            "baby_name": (row["name"] or "").strip(),
            "age_months": int(row["age_months"] or 0),
            "baby_state": _join_list("baby_status"),
            "concerns": _join_list("current_worries"),
            "parent_goals": _join_list("parent_hopes"),
            "growth_direction": _join_list("growth_direction"),
            "free_text": (happy.get("free_text") or "").strip(),
        }
    except Exception as exc:
        logger.warning("load_latest_baby_context failed: %s", exc)
        return {}


def is_symptom_question(question: str) -> bool:
    """질문에 일반 증상 키워드가 포함되는지 여부."""

    q = question or ""
    return any(kw in q for kw in _SYMPTOM_KEYWORDS)


def _symptom_response_footer() -> str:
    return (
        "정확한 진단과 치료는 의료진 상담이 필요합니다.\n"
        "아이 상태가 걱정되시면 소아과 진료를 권장드립니다."
    )


def _build_symptom_personal_sentence(baby_context: dict) -> str:
    """이름은 최대 1문장에서만 사용. 상태·고민·목표는 가볍게 반영."""
    ctx = baby_context or {}
    full_name = (ctx.get("baby_name") or "").strip()
    call = baby_call_name_for_coaching(full_name) if full_name else ""
    st = (ctx.get("baby_state") or "").strip()
    concerns = (ctx.get("concerns") or "").strip()
    goals = f"{ctx.get('parent_goals') or ''} {ctx.get('growth_direction') or ''}".strip()

    sentences: list[str] = []

    if call:
        if "예민함" in st:
            sentences.append(
                f"{call}가 예민한 편이라면 자극을 줄이고 편안한 환경에서 충분히 쉬게 해 주세요."
            )
        elif "활동적" in st:
            sentences.append(
                f"{call}가 활동적인 편이라면 당분간 활동을 조금 줄이고 충분히 쉬게 해 주세요."
            )
        elif "피곤해 보임" in st:
            sentences.append(
                f"{call}가 평소보다 피곤해 보인다면 무리하지 않게 쉼을 우선해 주세요."
            )
        elif "안정적" in st:
            sentences.append(
                f"{call}의 평소 리듬을 최대한 유지하시되, 무리만 없게 지켜봐 주시면 좋습니다."
            )

    if "낮잠 불규칙" in concerns:
        if sentences:
            sentences.append("특히 충분한 휴식이 도움이 될 수 있습니다.")
        else:
            sentences.append(
                "낮잠이 불규칙하다는 점을 감안하면, 충분한 휴식을 챙겨 주시면 좋습니다."
            )
    elif "식사 거부" in concerns and not sentences:
        sentences.append(
            "식사가 잘 안 될 때는 무리하게 먹이기보다 소량·자주 나눠 보시는 편이 좋습니다."
        )
    elif "자주 보챔" in concerns and not sentences:
        sentences.append(
            "보챔이 잦을 때는 안정된 환경에서 천천히 달래 주시면 도움이 될 수 있습니다."
        )

    if not sentences and (
        "튼튼" in goals
        or "건강한 생활" in goals
        or "잘 먹고 잘 자는" in goals
        or "수면 안정" in goals
    ):
        sentences.append(
            "회복과 컨디션을 위해 수분과 편한 휴식을 함께 챙겨 주시면 좋습니다."
        )

    return " ".join(sentences[:2]).strip()


def _symptom_blocks_for_keyword(kw: str) -> tuple[str, str] | None:
    """(도입 문단, 권장·실천 문단) — 소아과 권장은 이 블록에만 넣고 footer는 공통."""
    if kw == "열":
        return (
            "열이 나는 경우 아이의 상태를 먼저 확인하는 것이 중요합니다.",
            "수분 섭취를 충분히 하고, 아이가 많이 처져 보이거나 열이 높게(예: 약 38.5도 이상) "
            "느껴지거나 2~3일 이상 지속되면 소아과 진료를 권장드립니다.",
        )
    if kw == "기침":
        return (
            "기침은 감기나 호흡기 자극으로 인해 나타날 수 있습니다.",
            "실내 습도를 가볍게 유지하고 아이가 충분히 쉴 수 있게 도와주세요. "
            "기침이 오래 지속되거나 숨이 차 보이면 소아과 진료를 권장드립니다.",
        )
    if kw == "설사":
        return (
            "설사는 장의 일시적인 반응이나 음식 영향 등 여러 경우에 나타날 수 있습니다.",
            "탈수를 막기 위해 수분 섭취를 충분히 해 주세요. "
            "설사가 지속되거나 아이가 많이 처져 보이면 소아과 진료를 권장드립니다.",
        )
    if kw == "구토":
        return (
            "구토가 있을 때는 당분간 소량씩 수분을 나눠 드리고, 무리하게 먹이지 않도록 해 주세요.",
            "구토가 반복되거나 물도 잘 못 삼키면 탈수가 걱정되므로 소아과 진료를 권장드립니다.",
        )
    if kw == "발진":
        return (
            "발진은 피부 자극이나 감염 등 여러 원인이 있을 수 있어 원인을 단정하기 어렵습니다.",
            "긁지 않게 해 주시고, 급격히 퍼지거나 열·호흡이 불편해 보이면 소아과 진료를 권장드립니다.",
        )
    if kw == "콧물":
        return (
            "콧물은 감기나 비염 등으로 흔히 나타날 수 있습니다.",
            "아이가 숨 쉬기 편하도록 해 주시고, 증상이 심해지거나 오래가면 소아과 진료를 권장드립니다.",
        )
    if kw == "복통":
        return (
            "배가 아프다고 말할 때는 원인이 다양하므로 상황을 차분히 살펴봐 주시는 것이 좋습니다.",
            "통증이 계속되거나 구토·열이 함께 있거나 아이가 심하게 괴로워하면 소아과 진료를 권장드립니다.",
        )
    if kw == "변비":
        return (
            "변비가 있을 때는 수분과 규칙적인 배변 습관을 도와주시면 좋습니다.",
            "배가 많이 아프거나 구토·혈변이 보이면 소아과 진료를 권장드립니다.",
        )
    return None


def generate_symptom_response(question: str, baby_context: dict | None = None) -> str:
    """
    RAG 실패 시 보호자 안내(진단 아님). SQLite 맥락이 있으면 한두 문장만 개인화합니다.
    """
    ctx = baby_context if baby_context is not None else {}
    q = question or ""
    order = (
        "열",
        "기침",
        "설사",
        "구토",
        "발진",
        "콧물",
        "복통",
        "변비",
    )
    matched: str | None = None
    for kw in order:
        if kw in q:
            matched = kw
            break

    pers = _build_symptom_personal_sentence(ctx)
    blocks: list[str] = []

    if matched:
        pair = _symptom_blocks_for_keyword(matched)
        if pair:
            intro, bridge = pair
            blocks.append(intro)
            if pers:
                blocks.append(pers)
            blocks.append(bridge)
    else:
        blocks.append(
            "일상에서는 아이의 컨디션과 수분 섭취, 호흡이 편한지 먼저 살펴봐 주시면 좋습니다."
        )
        if pers:
            blocks.append(pers)
        blocks.append("증상이 심해지거나 걱정되시면 소아과 진료를 권장드립니다.")

    blocks.append(_symptom_response_footer())
    return "\n\n".join(blocks)


def _mock_answer(question: str, ctx: str) -> str:
    snippet = (ctx or "").replace("\n", " ").strip()
    if len(snippet) > 400:
        snippet = snippet[:400] + "…"
    if not snippet:
        return _INSUFFICIENT_ANSWER
    return (
        f"검색된 자료를 바탕으로 말씀드리면, 관련 설명의 일부는 다음과 같습니다: {snippet} "
        "자세한 판단은 소아과 진료에서 확인하시는 것이 좋습니다."
    )


def _looks_high_risk(text: str) -> bool:
    t = text or ""
    keys = (
        "호흡곤란",
        "숨이",
        "경련",
        "의식",
        "탈수",
        "지속 열",
        "고열",
        "5일 이상",
        "응급",
        "쇼크",
        "청색",
        "입술",
    )
    return any(k in t for k in keys)


class BabyDocAgent:
    """
    BabyDoc QA agent: calls `BabyDocRAGService.retrieve` then generates an answer with gpt-4o-mini.
    """

    def __init__(self, rag_service: BabyDocRAGService | None = None) -> None:
        self.rag_service = rag_service or BabyDocRAGService()

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Returns:
            {"answer", "retrieved_doc_ids", "sources", "status"}
        """

        q = (question or "").strip()
        empty_out: Dict[str, Any] = {
            "answer": "",
            "retrieved_doc_ids": [],
            "sources": [],
            "status": "empty",
        }

        if not q:
            empty_out["answer"] = "질문을 입력해 주세요."
            return empty_out

        baby_context = load_latest_baby_context()

        try:
            result = self.rag_service.retrieve(q)
        except Exception as exc:
            logger.exception("retrieve failed in BabyDocAgent: %s", exc)
            if is_symptom_question(q):
                return {
                    "answer": generate_symptom_response(q, baby_context),
                    "retrieved_doc_ids": [],
                    "sources": [],
                    "status": "symptom_guidance",
                }
            return {
                "answer": _INSUFFICIENT_ANSWER,
                "retrieved_doc_ids": [],
                "sources": [],
                "status": "insufficient_evidence",
            }

        if not result.get("success"):
            if is_symptom_question(q):
                return {
                    "answer": generate_symptom_response(q, baby_context),
                    "retrieved_doc_ids": [],
                    "sources": [],
                    "status": "symptom_guidance",
                }
            return {
                "answer": _INSUFFICIENT_ANSWER,
                "retrieved_doc_ids": [],
                "sources": [],
                "status": "insufficient_evidence",
            }

        doc_ids: List[str] = list(result.get("retrieved_doc_ids") or [])
        ctx = (result.get("compressed_context") or "").strip()
        retrieved_docs = result.get("retrieved_docs") or []

        sources: List[Dict[str, Any]] = []
        for d in retrieved_docs:
            if isinstance(d, dict):
                sources.append(
                    {
                        "doc_id": d.get("doc_id"),
                        "disease_kor": d.get("disease_kor"),
                        "intention": d.get("intention"),
                    }
                )

        risk_hint = ""
        if _looks_high_risk(q):
            risk_hint = (
                "\n(주의) 질문에 위험 신호가 포함될 수 있어 답변에 '소아과 진료 권장'을 반드시 넣으세요.\n"
            )

        user_block = f"보호자 질문:\n{q}\n{risk_hint}\n검색 문서 요약:\n{ctx or '(없음)'}\n"

        try:
            if BABYCOACH_LLM_MOCK:
                ans = _mock_answer(q, ctx)
            else:
                api_key = require_openai_api_key()
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=BABYDOC_LLM_MODEL,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_block},
                    ],
                )
                choice = resp.choices[0].message
                ans = (choice.content or "").strip()
                if not ans:
                    ans = "제공된 정보만으로는 정확히 답변하기 어렵습니다."

            if _looks_high_risk(q) and "소아과 진료 권장" not in ans:
                ans = ans.rstrip() + " 소아과 진료 권장드립니다."

            ans = re.sub(r"\s+", " ", ans).strip()
            return {
                "answer": ans,
                "retrieved_doc_ids": doc_ids,
                "sources": sources,
                "status": "ok",
            }

        except Exception as exc:
            logger.exception("LLM answer failed: %s", exc)
            fallback = _mock_answer(q, ctx) if ctx else _INSUFFICIENT_ANSWER
            return {
                "answer": fallback,
                "retrieved_doc_ids": doc_ids,
                "sources": sources,
                "status": "ok",
            }
