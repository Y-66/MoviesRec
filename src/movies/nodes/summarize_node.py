from langchain_core.messages import AIMessage, SystemMessage
from movies.states.state import AgentState
from movies.agents.summarizer import summarizer
from movies.prompts import SUMMARIZER_PROMPT

def summarize(state: AgentState) -> dict:
    intent_data = state.get("intent_data", {})
    if intent_data.get("intent") == "unrelated":
        # Just return the intent response for free chat
        msg = intent_data.get("response", "I'm a Movie Recommendation Robot!")
        return {"messages": [AIMessage(content=msg)]}

    final_recommendations = state.get("final_recommendations", [])

    # We provide the system prompt with the variable format for recommendations 
    formatted_prompt = SUMMARIZER_PROMPT.format(
        recommendations=final_recommendations
    )

    # We pass the formatted system prompt followed by the actual chat history   
    system_msg = SystemMessage(content=formatted_prompt)
    messages_to_send = [system_msg] + state["messages"]

    # Call the summarizer agent
    res = summarizer.invoke({"messages": messages_to_send})

    try:
        if "messages" in res and res["messages"]:
            content = res["messages"][-1].content
        else:
            content = "Here are your recommendations!"
    except Exception:
        content = "Here are your recommendations!"
        
    return {"messages": [AIMessage(content=content)]}
