import json
from langchain_core.messages import AIMessage
from movies.states.state import AgentState
from movies.agents.intent_analyzer import intent_analyzer

def analyze_intent(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"intent_data": {}}

    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    # We pass the conversation context explicitly in a single user message
    # so the intent analyzer's json-formatting behavior isn't corrupted by non-JSON AIMessages.
    prompt = f"Conversation History:\n{history_text}\n\nTask: Please analyze the LAST user's intent according to your system prompt instructions."

    response = intent_analyzer.invoke({"messages": [{"role": "user", "content": prompt}]})

    try:
        if "messages" in response and response["messages"]:
            content = response["messages"][-1].content
        else:
            content = "{}"

        # Clean up markdown JSON formatting if present
        content = content.replace("```json", "").replace("```", "").strip()

        intent_data = json.loads(content)
        return {"intent_data": intent_data}
    except Exception as e:
        print(f"DEBUG: Exception in analyze_intent parsing: {e}")
        print(f"DEBUG: content was: {content if 'content' in locals() else 'N/A'}")
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
