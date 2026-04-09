import math

from movies.states.state import AgentState

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def diversity_filter(state: AgentState) -> dict:
    cf_recommendations = state.get("cf_recommendations", [])
    filtered_movies = state.get("filtered_movies", [])
    
    if not cf_recommendations:
        return {"final_recommendations": []}
        
    print(f"Applying diversity and novelty to {len(cf_recommendations)} recommendations.")
    
    # Build a lookup dictionary to find full attributes by movie_id
    movie_features = {}
    for movie in filtered_movies:
        movie_id = movie.get("movieId")
        if movie_id is not None:
            movie_features[movie_id] = movie
            
    # [Strategy 1] Novelty
    # We use rating_count (number of ratings) to measure popularity. We want to give a score compensation to niche movies (long-tail recommendation).
    max_rating_count = 1
    for item in cf_recommendations:
        m_id = item.get("movie_id")
        rc = movie_features.get(m_id, {}).get("rating_count") or 1
        if rc > max_rating_count:
            max_rating_count = rc

    NOVELTY_WEIGHT = 0.2
    novelty_adjusted_cands = []
    
    for item in cf_recommendations:
        movie_id = item["movie_id"]
        features = movie_features.get(movie_id, {})
        rating_count = features.get("rating_count", 0) or 0
        original_score = item.get("final_score", 0.0)
        
        # Novelty score: The lower the popularity, the higher the novelty. Smooth with logarithmic decay
        novelty_score = 1.0 - (math.log1p(rating_count) / math.log1p(max_rating_count))
        
        new_score = original_score + (novelty_score * NOVELTY_WEIGHT)
        
        # Extract genres for diversity calculation
        genres_str = features.get("genres", "")
        genres_set = set(genres_str.split("|")) if genres_str else set()
        
        # Create a new dictionary to prevent modifying original state data
        cand_dict = dict(item)
        cand_dict["novelty_score"] = round(novelty_score, 4)
        cand_dict["adjusted_score"] = round(new_score, 4)
        cand_dict["_genres"] = genres_set # For internal use
        
        novelty_adjusted_cands.append(cand_dict)
        
    # Preliminary sorting based on total score after adding novelty
    novelty_adjusted_cands.sort(key=lambda x: x["adjusted_score"], reverse=True)
    
    # [Strategy 2] Diversity (MMR)
    # Based on Jaccard similarity of genres, penalize candidate movies that are too similar to already selected movies.
    DIVERSITY_WEIGHT = 0.5  # Diversity weight lambda
    TOP_K = min(5, len(novelty_adjusted_cands))
    
    selected_movies = []
    candidates = novelty_adjusted_cands.copy()
    
    while len(selected_movies) < TOP_K and candidates:
        if not selected_movies:
            # The first movie is always the one with the highest current score
            best_cand = candidates.pop(0)
            best_cand["mmr_score"] = best_cand["adjusted_score"]
            selected_movies.append(best_cand)
            continue
            
        best_mmr_score = -float('inf')
        best_cand_index = -1
        
        for idx, cand in enumerate(candidates):
            # Calculate the maximum similarity between the current candidate and globally selected movies
            max_sim = 0.0
            for selected in selected_movies:
                sim = jaccard_similarity(cand["_genres"], selected["_genres"])
                if sim > max_sim:
                    max_sim = sim
            
            # MMR Score = (1 - lambda) * relevance - lambda * max_sim
            relevance = cand["adjusted_score"]
            mmr_score = (1.0 - DIVERSITY_WEIGHT) * relevance - DIVERSITY_WEIGHT * max_sim
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_cand_index = idx
                
        best_cand = candidates.pop(best_cand_index)
        best_cand["mmr_score"] = round(best_mmr_score, 4)
        selected_movies.append(best_cand)
        
    # Clean up internal fields
    for item in selected_movies:
        if "_genres" in item:
            del item["_genres"]
            
    # Organize output results for deduplication and consistent formatting
    print(f"Produced {len(selected_movies)} final diverse recommendations.")
    
    return {"final_recommendations": selected_movies}
