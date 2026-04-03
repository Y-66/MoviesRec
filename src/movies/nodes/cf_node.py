import time
from concurrent.futures import ThreadPoolExecutor

from algos.feature_extraction import rank_by_content_similarity
from algos.svd_model_predictor import SVDRecommenderPredictor, get_collaborative_candidates
from movies.states.state import AgentState

def collaborative_filter(state: AgentState) -> dict:
    filtered_movies = state.get("filtered_movies", [])
    print(f"Applying CF on top of filtered movies: {filtered_movies}")

    # 过滤掉 None 以防异常
    filtered_movie_ids = [movies.get("movieId") for movies in filtered_movies if movies.get("movieId") is not None]

    print(f"Extracted movie IDs for CF: {filtered_movie_ids}")
    

    predictor = SVDRecommenderPredictor()
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(e)
        exit(1)  
    
    target_user_id = 1
    
    # 并发计算 SVD协同过滤 和 内容相似度
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 为了融合打分，CF这步取全部电影的得分 (top_k=len)
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
        
    # 建立映射以利用 movie_id 查找
    cf_map = {item['movie_id']: item['svd_score'] for item in cf_result_list}
    sim_map = {item['movie_id']: item['similarity'] for item in movies_similarity_list}
    
    combined_results = []
    
    # 权重配置：可根据业务微调
    WEIGHT_CF = 0.6
    WEIGHT_SIM = 0.4
    
    for movie_id in filtered_movie_ids:
        svd_score = cf_map.get(movie_id, 0.0)
        sim_score = sim_map.get(movie_id, 0.0)
        
        # 将 svd_score(0.5-5) 归一化到 0-1 区间，除以5即可
        norm_svd = svd_score / 5.0
        # 相似度通常也是 <= 1，负数可作截断或平移，这里假设内容相近度不会负得严重
        norm_sim = max(0.0, sim_score)
        
        final_score = (norm_svd * WEIGHT_CF) + (norm_sim * WEIGHT_SIM)
        
        combined_results.append({
            "movie_id": movie_id,
            "svd_score": svd_score,
            "similarity": sim_score,
            "final_score": round(final_score, 4)
        })
        
    # 根据加权最终得分排序并取 Top 5
    combined_results.sort(key=lambda x: x["final_score"], reverse=True)
    top_5_recommendations = combined_results[:5]
    
    return {"cf_recommendations": top_5_recommendations}
