import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, vstack, csr_matrix

from algos.svd_model_predictor import SVDRecommenderPredictor, get_collaborative_candidates

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "db" / "movies_data.db"
MOVIES_PATH = BASE_DIR / "datasets" / "movies_aggregated.csv"

def load_movie_features():
    """
    统一的特征工程加载函数。
    为后续性能考虑，生产环境中可以将此函数的返回值建立缓存。
    """
    # 读取数据
    movies = pd.read_csv(MOVIES_PATH)

    # Genres → One-hot（sparse）
    movies['genres'] = movies['genres'].fillna('').str.split('|')
    genres_df = movies.explode('genres')

    genres_onehot = pd.get_dummies(genres_df['genres'])
    genres_matrix = genres_onehot.groupby(genres_df['movieId']).sum()

    movie_ids = movies['movieId'].values
    genres_aligned = genres_matrix.reindex(movie_ids, fill_value=0)
    genres_sparse = csr_matrix(genres_aligned.values)

    # Tags → TF-IDF（sparse）
    movies['tags'] = movies['tags'].fillna('')
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(movies['tags'])

    # 合并特征
    movie_vectors = hstack([genres_sparse, tfidf_matrix])
    movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    return movie_vectors, movie_id_to_index

def calculate_single_user_profile(uid, user_ratings, movie_vectors, movie_id_to_index):
    """
    单独抽离出的用户画像计算核心函数，可供离线脚本使用。
    """
    if user_ratings.empty:
        return None

    # 计算该用户的真实历史平均分，用作相对基准
    user_mean = float(user_ratings['rating'].mean())
    # 获取最新时间戳作为衰减基准计算
    max_time = float(user_ratings['timestamp'].max()) if 'timestamp' in user_ratings.columns else 0.0

    vectors = []
    weights = []
    
    for _, row in user_ratings.iterrows():
        mid = int(row['movieId'])
        rating = float(row['rating'])
        
        if mid in movie_id_to_index:
            vectors.append(movie_vectors[movie_id_to_index[mid]].toarray())
            
            # 使用相对基准线
            base_weight = rating - user_mean
            
            # 引入半衰期时间截断机制 (假设半衰期为365天，使早期的口味权重降低)
            if max_time > 0 and 'timestamp' in row and not pd.isna(row['timestamp']):
                days_diff = (max_time - float(row['timestamp'])) / (24 * 3600)
                decay = np.exp(-np.log(2) * days_diff / 365.0)
            else:
                decay = 1.0
            
            weights.append(base_weight * decay) 

    if not vectors:
        return None

    vectors_stacked = np.vstack(vectors)
    weights_array = np.array(weights).reshape(-1, 1)
    
    weighted_vectors = vectors_stacked * weights_array
    
    weight_sum = np.sum(np.abs(weights_array))
    if weight_sum == 0:
        user_vector = np.mean(vectors_stacked, axis=0)
    else:
        user_vector = np.sum(weighted_vectors, axis=0) / weight_sum
        
    return user_vector

def rank_by_content_similarity(user_id, candidate_movie_ids):
    """
    提取候选电影的内容特征（类似度），并在线实时计算该用户的加权特征偏好画像。
    """
    # 1. 提取电影特征空间
    movie_vectors, movie_id_to_index = load_movie_features()
    
    # 2. 读取评分数据以供在线计算用户画像 (每次实时拉取该用户的所有评分历史)
    ratings_path = BASE_DIR / "datasets" / "ratings.csv"
    ratings = pd.read_csv(ratings_path)
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # 3. 实时计算当前用户的特征画像
    user_vector = calculate_single_user_profile(user_id, user_ratings, movie_vectors, movie_id_to_index)
    
    results = []
    for item in candidate_movie_ids:
        movie_id = item['movie_id']
        svd_score = item['svd_score']

        if movie_id in movie_id_to_index and user_vector is not None:
            movie_vector = movie_vectors[movie_id_to_index[movie_id]].toarray()
            # user_vector 放平(reshape)便于跟 movie_vector 计算余弦相似度
            similarity = cosine_similarity(user_vector.reshape(1, -1), movie_vector)[0][0]
        else:
            similarity = 0.0

        results.append({
            "movie_id": movie_id,
            "svd_score": float(svd_score),
            "similarity": float(similarity)
        })

    return results

if __name__ == "__main__":
    user_id = 1

    print("🚀 Loading SVD model...")
    # predictor = SVDRecommenderPredictor()
    # predictor.load_model()

    # 测试候选集
    candidate_movies = [
        {'movie_id': 1, 'svd_score': 4.5},
        {'movie_id': 2, 'svd_score': 3.8},
        {'movie_id': 3, 'svd_score': 4.1}
    ]

    print(f"🧠 Refining candidates for User {user_id} with DB content-based similarity...")
    results = rank_by_content_similarity(user_id, candidate_movies)

    print("\n🎬 Raw Features Extracted:")
    for r in results:
        print(r)
