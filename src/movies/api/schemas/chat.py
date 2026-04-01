from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="The user input for this turn")
    session_id: str = Field("default", min_length=1, description="Conversation thread id")


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


class SessionDeleteResponse(BaseModel):
    session_id: str
    removed: bool
