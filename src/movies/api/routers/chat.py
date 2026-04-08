from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
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


def _infer_stage_trace(state: Dict[str, Any]) -> List[str]:
    trace = ["intent_analyzer"]
    intent_data = state.get("intent_data", {}) or {}

    if intent_data.get("intent") == "unrelated":
        trace.append("summarizer")
        return trace

    hard_filters = intent_data.get("hard_filters") or {}
    if hard_filters:
        trace.append("sql_filter")

    trace.extend(["collaborative_filter", "diversity_filter", "summarizer"])
    return trace


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 24) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    request: ChatRequest,
    http_request: Request,
) -> ChatResponse:
    try:
        started_at = time.perf_counter()
        created_at = datetime.now(timezone.utc).isoformat()
        request_id = str(uuid.uuid4())

        agent = _get_agent(http_request)
        base_dir = _get_base_dir(http_request)

        session_id = request.session_id or "default"
        inputs: Dict[str, Any] = {"messages": [HumanMessage(content=request.user_input)]}
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        result = agent.invoke(inputs, config)  # type: ignore[arg-type]
        full_messages = result.get("messages", inputs["messages"])
        save_session_history(session_id, full_messages, base_dir=base_dir)
        stage_trace = _infer_stage_trace(result)

        recommendations = result.get("final_recommendations", []) or []
        if not isinstance(recommendations, list):
            recommendations = []

        # Fetch extra movie details (poster, overview, link) using tmdb_api
        movie_ids = []
        for rec in recommendations:
            mid = rec.get("movieId") or rec.get("movie_id") or rec.get("id")
            if mid is not None:
                try:
                    movie_ids.append(int(mid))
                except ValueError:
                    pass

        if movie_ids:
            try:
                import asyncio
                from scripts.tmdb_api import get_and_store_movie_details
                details_map = await asyncio.to_thread(get_and_store_movie_details, movie_ids)
                for rec in recommendations:
                    mid = rec.get("movieId") or rec.get("movie_id") or rec.get("id")
                    if mid is not None:
                        try:
                            mid_int = int(mid)
                            if mid_int in details_map:
                                details = details_map[mid_int]
                                rec["poster_url"] = details.get("poster_url")
                                rec["overview"] = details.get("overview")
                                rec["link"] = details.get("link")
                                rec["title"] = details.get("title") or rec.get("title")
                                rec["release_date"] = details.get("release_date")
                                rec["popularity"] = details.get("popularity")
                        except ValueError:
                            pass
            except Exception as e:
                print(f"Error fetching movie details from tmdb: {e}")

        ai_reply = next(
            (_content_to_text(msg.content) for msg in reversed(full_messages) if isinstance(msg, AIMessage)),
            "I am a Movie Recommendation Robot. How can I help you with movies today?",
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        return ChatResponse(
            session_id=session_id,
            request_id=request_id,
            created_at=created_at,
            response=ai_reply,
            latency_ms=latency_ms,
            message_count=len(full_messages),
            stage_trace=stage_trace,
            recommendation_count=len(recommendations),
            recommendation_cards=recommendations,
            intent_data=result.get("intent_data", {}),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stream", status_code=status.HTTP_200_OK)
async def chat_stream_endpoint(request: ChatRequest, http_request: Request) -> StreamingResponse:
    agent = _get_agent(http_request)
    base_dir = _get_base_dir(http_request)
    session_id = request.session_id or "default"

    async def event_generator():
        started_at = time.perf_counter()
        created_at = datetime.now(timezone.utc).isoformat()
        request_id = str(uuid.uuid4())

        yield _sse(
            "start",
            {
                "request_id": request_id,
                "session_id": session_id,
                "created_at": created_at,
            },
        )

        try:
            inputs: Dict[str, Any] = {"messages": [HumanMessage(content=request.user_input)]}
            config: RunnableConfig = {"configurable": {"thread_id": session_id}}

            # Stream node-level progress and custom token events from graph nodes.
            observed_stages: List[str] = ["intent_analyzer"]
            streamed_state: Dict[str, Any] = {}
            saw_token_events = False
            
            # initial stage is always intent_analyzer
            yield _sse(
                "stage",
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "stage": "intent_analyzer",
                },
            )

            for mode, chunk in agent.stream(inputs, config, stream_mode=["updates", "custom"]):  # type: ignore[arg-type]
                if mode == "updates" and isinstance(chunk, dict):
                    for stage_name, payload in chunk.items():
                        if isinstance(payload, dict):
                            streamed_state.update(payload)
                        
                        # Infer next stage dynamically to keep frontend in sync
                        next_stage = None
                        if stage_name == "intent_analyzer":
                            intent_data = streamed_state.get("intent_data", {})
                            if intent_data.get("intent") == "unrelated":
                                next_stage = "summarizer"
                            elif not intent_data.get("hard_filters"):
                                next_stage = "collaborative_filter"
                            else:
                                next_stage = "sql_filter"
                        elif stage_name == "sql_filter":
                            next_stage = "collaborative_filter"
                        elif stage_name == "collaborative_filter":
                            next_stage = "diversity_filter"
                        elif stage_name == "diversity_filter":
                            next_stage = "summarizer"
                            
                        if next_stage and next_stage not in observed_stages:
                            observed_stages.append(next_stage)
                            yield _sse(
                                "stage",
                                {
                                    "request_id": request_id,
                                    "session_id": session_id,
                                    "stage": next_stage,
                                },
                            )
                elif mode == "custom" and isinstance(chunk, dict):
                    if chunk.get("type") == "token":
                        saw_token_events = True
                        yield _sse(
                            "token",
                            {
                                "request_id": request_id,
                                "session_id": session_id,
                                "text": chunk.get("text", ""),
                            },
                        )

            # Always prefer canonical thread state from checkpointer after stream run.
            result: Dict[str, Any] = {}
            try:
                snapshot = agent.get_state(config)  # type: ignore[attr-defined]
                values = getattr(snapshot, "values", None)
                if isinstance(values, dict):
                    result = values
            except Exception:
                result = {}

            # Fallback hierarchy for robustness.
            if "messages" not in result and "messages" in streamed_state:
                result = streamed_state
            if "messages" not in result:
                result = agent.invoke(inputs, config)  # type: ignore[arg-type]
            full_messages = result.get("messages", streamed_state.get("messages", inputs["messages"]))
            save_session_history(session_id, full_messages, base_dir=base_dir)

            ai_reply = next(
                (_content_to_text(msg.content) for msg in reversed(full_messages) if isinstance(msg, AIMessage)),
                "I am a Movie Recommendation Robot. How can I help you with movies today?",
            )

            # Fallback chunking if model token events were not available.
            if not saw_token_events:
                for chunk in _chunk_text(ai_reply):
                    yield _sse(
                        "token",
                        {
                            "request_id": request_id,
                            "session_id": session_id,
                            "text": chunk,
                        },
                    )

            recommendations = result.get("final_recommendations", []) or []
            if not isinstance(recommendations, list):
                recommendations = []

            # Fetch extra movie details (poster, overview, link) using tmdb_api
            movie_ids = []
            for rec in recommendations:
                mid = rec.get("movieId") or rec.get("movie_id") or rec.get("id")
                if mid is not None:
                    try:
                        movie_ids.append(int(mid))
                    except ValueError:
                        pass

            if movie_ids:
                try:
                    import asyncio
                    from scripts.tmdb_api import get_and_store_movie_details
                    details_map = await asyncio.to_thread(get_and_store_movie_details, movie_ids)
                    for rec in recommendations:
                        mid = rec.get("movieId") or rec.get("movie_id") or rec.get("id")
                        if mid is not None:
                            try:
                                mid_int = int(mid)
                                if mid_int in details_map:
                                    details = details_map[mid_int]
                                    rec["poster_url"] = details.get("poster_url")
                                    rec["overview"] = details.get("overview")
                                    rec["link"] = details.get("link")
                                    rec["title"] = details.get("title") or rec.get("title")
                                    rec["release_date"] = details.get("release_date")
                                    rec["popularity"] = details.get("popularity")
                            except ValueError:
                                pass
                except Exception as e:
                    print(f"Error fetching movie details from tmdb: {e}")

            stage_trace = observed_stages or _infer_stage_trace(result)
            latency_ms = int((time.perf_counter() - started_at) * 1000)

            yield _sse(
                "final",
                {
                    "session_id": session_id,
                    "request_id": request_id,
                    "created_at": created_at,
                    "response": ai_reply,
                    "latency_ms": latency_ms,
                    "message_count": len(full_messages),
                    "stage_trace": stage_trace,
                    "recommendation_count": len(recommendations),
                    "recommendation_cards": recommendations,
                    "intent_data": result.get("intent_data", {}),
                },
            )

            yield _sse("done", {"request_id": request_id, "session_id": session_id})
        except Exception as exc:
            yield _sse(
                "error",
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "detail": str(exc),
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Content-Encoding": "identity",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
