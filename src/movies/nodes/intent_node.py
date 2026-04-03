import json
from langchain_core.messages import AIMessage
from movies.states.state import AgentState
from movies.agents.intent_analyzer import intent_analyzer

def analyze_intent(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"intent_data": {}}

    try:
        # Directly pass the messages to our structured output model
        response = intent_analyzer.invoke({"messages": messages})
        intent_data = response.get("intent_data", {})
        return {"intent_data": intent_data}
    except Exception as e:
        print(f"DEBUG: Exception in analyze_intent structured output: {e}")
        fallback = {"intent": "unrelated", "response": "Sorry, I didn't quite catch that. I am a Movie Recommendation Robot!"}
        return {"intent_data": fallback}

def route_intent(state: AgentState):
    intent_data = state.get("intent_data", {})
    if intent_data.get("intent") == "unrelated":
        return "summarize"
    hard_filters = intent_data.get("hard_filters") or {}
    if not hard_filters:
        return "collaborative_filter"
    return "sql_filter"
