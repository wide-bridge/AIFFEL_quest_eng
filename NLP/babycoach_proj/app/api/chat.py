from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..llm import generate_chat_reply
from ..schemas import ChatRequest, ChatResponse


router = APIRouter(prefix="", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """
    Chat endpoint: uses `final_output` and `user_message` only (no DB dependency).
    """

    try:
        assistant_message = generate_chat_reply(
            final_output=payload.final_output,
            user_message=payload.user_message,
            state_summary=payload.state_summary,
        )
        return ChatResponse(assistant_message=assistant_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

