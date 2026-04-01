from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, cast
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

import uvicorn

import sys
from pathlib import Path
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from movies.graph import build_graph
from movies.utils.file_storage import (
    list_session_summaries,
    read_session_history_payload,
    save_session_history,
)

BASE_DIR = Path(__file__).resolve().parent
checkpoint_memory = MemorySaver()
movies_agent = build_graph(checkpointer=checkpoint_memory)

app = FastAPI(
    title="Movies Recommendation Agent API",
    description="Agentic backend for Collaborative Filtering-based Dialog Movie Recommendations",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    intent_data: Optional[Dict[str, Any]] = None


class SessionSummary(BaseModel):
    session_id: str
    updated_at: Optional[str] = None
    message_count: int
    file: str


class SessionHistory(BaseModel):
    session_id: str
    updated_at: Optional[str] = None
    message_count: int
    messages: List[Dict[str, Any]]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        inputs =  {"messages": [HumanMessage(content=request.user_input)]}

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        
        result = movies_agent.invoke(inputs, config) # type: ignore

        full_messages = result.get("messages", inputs["messages"])
        save_session_history(session_id, full_messages, base_dir=BASE_DIR)

        # Find the latest AI reply as API response.
        ai_reply = next(
            (_content_to_text(msg.content) for msg in reversed(full_messages) if isinstance(msg, AIMessage)),
            "I am a Movie Recommendation Robot. How can I help you with movies today?",
        )
        intent_info = result.get("intent_data", {})

        return ChatResponse(
            response=ai_reply,
            intent_data=intent_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions", response_model=List[SessionSummary])
async def list_sessions() -> List[SessionSummary]:
    return [SessionSummary(**item) for item in list_session_summaries(base_dir=BASE_DIR)]


@app.get("/chat/history/{session_id}", response_model=SessionHistory)
async def get_session_history(session_id: str) -> SessionHistory:
    payload = read_session_history_payload(session_id, base_dir=BASE_DIR)
    return SessionHistory(**payload)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
