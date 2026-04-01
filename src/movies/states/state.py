from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class IntentResult(TypedDict):
    intent: str  # "greeting", "unrelated", "recommendation"
    hard_filters: Dict[str, Any]  # e.g., {"genre": "Action", "year": 2020}
    response: Optional[str]  # Free chat response if not recommendation

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent_data: IntentResult
    filtered_movies: List[Dict[str, Any]]
    cf_recommendations: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]
