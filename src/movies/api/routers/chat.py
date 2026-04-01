from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from movies.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionDeleteResponse,
    SessionHistory,
    SessionSummary,
)

from movies.utils.file_storage import (
    list_session_summaries,
    read_session_history_payload,
    save_session_history,
    session_history_path,
)

router = APIRouter(prefix="/chat", tags=["chat"])


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


def _get_agent(request: Request) -> Any:
    return request.app.state.agent


def _get_base_dir(request: Request) -> Path:
    return request.app.state.base_dir


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    request: ChatRequest,
    http_request: Request,
) -> ChatResponse:
    try:
        agent = _get_agent(http_request)
        base_dir = _get_base_dir(http_request)

        session_id = request.session_id or "default"
        inputs: Dict[str, Any] = {"messages": [HumanMessage(content=request.user_input)]}
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        result = agent.invoke(inputs, config)  # type: ignore[arg-type]
        full_messages = result.get("messages", inputs["messages"])
        save_session_history(session_id, full_messages, base_dir=base_dir)

        ai_reply = next(
            (_content_to_text(msg.content) for msg in reversed(full_messages) if isinstance(msg, AIMessage)),
            "I am a Movie Recommendation Robot. How can I help you with movies today?",
        )
        return ChatResponse(response=ai_reply, intent_data=result.get("intent_data", {}))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sessions", response_model=List[SessionSummary], status_code=status.HTTP_200_OK)
async def list_sessions(http_request: Request) -> List[SessionSummary]:
    base_dir = _get_base_dir(http_request)
    items = list_session_summaries(base_dir=base_dir)
    return [SessionSummary(**item) for item in items]


@router.get("/history/{session_id}", response_model=SessionHistory, status_code=status.HTTP_200_OK)
async def get_session_history(session_id: str, http_request: Request) -> SessionHistory:
    base_dir = _get_base_dir(http_request)
    payload = read_session_history_payload(session_id, base_dir=base_dir)
    return SessionHistory(**payload)


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse, status_code=status.HTTP_200_OK)
async def delete_session(session_id: str, http_request: Request) -> SessionDeleteResponse:
    base_dir = _get_base_dir(http_request)
    target = session_history_path(session_id, base_dir=base_dir)
    removed = False
    if target.exists():
        target.unlink()
        removed = True
    return SessionDeleteResponse(session_id=session_id, removed=removed)
