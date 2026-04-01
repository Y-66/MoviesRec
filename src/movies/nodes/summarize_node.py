from langchain_core.messages import AIMessage, SystemMessage
from langgraph.config import get_stream_writer
from movies.states.state import AgentState
from movies.agents.summarizer import summarizer
from movies.prompts import SUMMARIZER_PROMPT

def summarize(state: AgentState) -> dict:
    writer = None
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    intent_data = state.get("intent_data", {})
    if intent_data.get("intent") == "unrelated":
        # Just return the intent response for free chat
        msg = intent_data.get("response", "I'm a Movie Recommendation Robot!")
        if writer:
            writer({"type": "token", "text": msg})
        return {"messages": [AIMessage(content=msg)]}

    final_recommendations = state.get("final_recommendations", [])

    # We provide the system prompt with the variable format for recommendations 
    formatted_prompt = SUMMARIZER_PROMPT.format(
        recommendations=final_recommendations
    )

    # We pass the formatted system prompt followed by the actual chat history   
    system_msg = SystemMessage(content=formatted_prompt)
    messages_to_send = [system_msg] + state["messages"]

    # Stream summary generation so the API can forward token-level updates.
    streamed_parts = []
    try:
        for piece in summarizer.stream({"messages": messages_to_send}):
            streamed_parts.append(piece)
            if writer:
                writer({"type": "token", "text": piece})
        content = "".join(streamed_parts).strip() or "Here are your recommendations!"
    except Exception:
        content = "Here are your recommendations!"
        
    return {"messages": [AIMessage(content=content)]}
