from movies.states.state import AgentState

def sql_filter(state: AgentState) -> dict:
    intent_data = state.get("intent_data", {})
    hard_filters = intent_data.get("hard_filters", {})
    
    # Placeholder: Implement actual SQLite filtering logic here.
    # We would connect to db and execute filtering on genre, rating, time.
    print(f"Executing SQL filter with conditions: {hard_filters}")
    
    # Fake filtered recommendations for demonstration
    filtered = [{"id": 1, "title": "Example Filtered Movie"}]
    
    return {"filtered_movies": filtered}
