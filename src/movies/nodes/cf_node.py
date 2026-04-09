import time
from concurrent.futures import ThreadPoolExecutor

from algos.feature_extraction import rank_by_content_similarity
from algos.svd_model_predictor import SVDRecommenderPredictor, get_collaborative_candidates
from movies.states.state import AgentState

def collaborative_filter(state: AgentState) -> dict:
    filtered_movies = state.get("filtered_movies", [])
    print(f"Applying CF on top of filtered movies: {filtered_movies}")

    # Filter out None to prevent exceptions
    filtered_movie_ids = [movies.get("movieId") for movies in filtered_movies if movies.get("movieId") is not None]

    print(f"Extracted movie IDs for CF: {filtered_movie_ids}")
    

    predictor = SVDRecommenderPredictor()
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(e)
        exit(1)  
    
    target_user_id = 1
    
    # Concurrently compute SVD Collaborative Filtering and Content Similarity
    with ThreadPoolExecutor(max_workers=2) as executor:
        # To fuse scores, CF step takes scores for all movies (top_k=len)
        future_cf = executor.submit(
            get_collaborative_candidates,
            predictor.algo,
            target_user_id,
            filtered_movie_ids, # type: ignore
            len(filtered_movie_ids) if filtered_movie_ids else 5
        )
        future_sim = executor.submit(
            rank_by_content_similarity,
            target_user_id,
            filtered_movie_ids
        )
        
        cf_result_list = future_cf.result()
        movies_similarity_list = future_sim.result()
        
    # Build mapping to look up by movie_id
    cf_map = {item['movie_id']: item['svd_score'] for item in cf_result_list}
    sim_map = {item['movie_id']: item['similarity'] for item in movies_similarity_list}
    
    combined_results = []
    
    # Weight configuration: can be fine-tuned according to business needs
    WEIGHT_CF = 0.6
    WEIGHT_SIM = 0.4
    
    for movie_id in filtered_movie_ids:
        svd_score = cf_map.get(movie_id, 0.0)
        sim_score = sim_map.get(movie_id, 0.0)
        
        # Normalize svd_score(0.5-5) to 0-1 range by dividing by 5
        norm_svd = svd_score / 5.0
        # Similarity is usually <= 1, negative can be truncated or shifted, assuming content similarity isn't severely negative here
        norm_sim = max(0.0, sim_score)
        
        final_score = (norm_svd * WEIGHT_CF) + (norm_sim * WEIGHT_SIM)
        
        combined_results.append({
            "movie_id": movie_id,
            "svd_score": svd_score,
            "similarity": sim_score,
            "final_score": round(final_score, 4)
        })
        
    # Sort by weighted final score and appropriately keep more candidates for diversity node (e.g. top 20 items)
    combined_results.sort(key=lambda x: x["final_score"], reverse=True)
    top_20_recommendations = combined_results[:20]
    
    return {"cf_recommendations": top_20_recommendations}
