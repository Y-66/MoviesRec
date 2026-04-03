import time

from algos.svd_model_predictor import SVDRecommenderPredictor, get_collaborative_candidates
from movies.states.state import AgentState

def collaborative_filter(state: AgentState) -> dict:
    filtered_movies = state.get("filtered_movies", [])
    print(f"Applying CF on top of filtered movies: {filtered_movies}")

    filtered_movie_ids = [movies.get("movieId", None) for movies in filtered_movies]

    print(f"Extracted movie IDs for CF: {filtered_movie_ids}")
    

    predictor = SVDRecommenderPredictor()
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(e)
        exit(1)  
    
    cf_result = get_collaborative_candidates(
        algo=predictor.algo,
        user_id=1,
        candidate_movie_ids=filtered_movie_ids, # type: ignore
        top_k=5  # 取 Top 5
    )
    
    
    return {"cf_recommendations": cf_result}
