from movies.states.state import AgentState
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "db" / "movies_data.db"

def sql_filter(state: AgentState) -> dict:
    intent_data = state.get("intent_data", {})
    hard_filters = intent_data.get("hard_filters", {})
    
    print(f"Executing SQL filter with conditions: {hard_filters}")
    
    # Base query
    query = "SELECT * FROM movies WHERE 1=1"
    params = []
    
    # Build conditions based on hard filters
    if "genre" in hard_filters and hard_filters["genre"]:
        query += " AND genres LIKE ?"
        params.append(f"%{hard_filters['genre']}%")
        
    if "year" in hard_filters and hard_filters["year"]:
        query += " AND year = ?"
        params.append(str(hard_filters["year"]))
        
    if "rating" in hard_filters and hard_filters["rating"]:
        query += " AND avg_rating >= ?"
        params.append(float(hard_filters["rating"]))
        
    # Limit results to avoid massive payloads
    query += " ORDER BY rating_count DESC LIMIT 20"
    
    filtered_movies = []
    try:
        # Connect to SQLite DB
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        for row in rows:
            # Convert sqlite3.Row to dict
            filtered_movies.append(dict(row))
            
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            
    if not filtered_movies:
        print("No movies matched the SQL filters.")
    
    return {"filtered_movies": filtered_movies}
