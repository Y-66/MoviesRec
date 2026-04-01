from langgraph.graph import StateGraph, START, END
from movies.states.state import AgentState
from movies.nodes.intent_node import analyze_intent, route_intent
from movies.nodes.sql_filter_node import sql_filter
from movies.nodes.cf_node import collaborative_filter
from movies.nodes.diversity_node import diversity_filter
from movies.nodes.summarize_node import summarize

def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("intent_analyzer", analyze_intent)
    workflow.add_node("sql_filter", sql_filter)
    workflow.add_node("collaborative_filter", collaborative_filter)
    workflow.add_node("diversity_filter", diversity_filter)
    workflow.add_node("summarizer", summarize)
    
    workflow.add_edge(START, "intent_analyzer")
    
    workflow.add_conditional_edges(
        "intent_analyzer",
        route_intent,
        {
            "sql_filter": "sql_filter",
            "collaborative_filter": "collaborative_filter",
            "summarize": "summarizer"
        }
    )
    
    workflow.add_edge("sql_filter", "collaborative_filter")
    workflow.add_edge("collaborative_filter", "diversity_filter")
    workflow.add_edge("diversity_filter", "summarizer")
    workflow.add_edge("summarizer", END)
    
    return workflow.compile(checkpointer=checkpointer)

app = build_graph()
