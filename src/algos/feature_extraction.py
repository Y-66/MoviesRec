import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, vstack, csr_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "db" / "movies_data.db"
MOVIES_PATH = BASE_DIR / "datasets" / "movies_aggregated.csv"

@lru_cache(maxsize=1)
def load_movie_features():
    """
    统一的特征工程加载函数。
    为后续性能考虑，生产环境中可以将此函数的返回值建立缓存。
    """
    # 读取数据
    movies = pd.read_csv(MOVIES_PATH)

    # Genres → One-hot（sparse）
    movies['genres'] = movies['genres'].fillna('')
    cv = CountVectorizer(token_pattern=r'[^|]+')
    genres_sparse = cv.fit_transform(movies['genres'])

    movie_ids = movies['movieId'].values

    # Tags → TF-IDF（sparse）
    movies['tags'] = movies['tags'].fillna('')
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(movies['tags'])

    # 合并特征
    movie_vectors = hstack([genres_sparse, tfidf_matrix]).tocsr()
    movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    return movie_vectors, movie_id_to_index

from sklearn.cluster import KMeans

def calculate_single_user_profile(uid, user_ratings, movie_vectors, movie_id_to_index):
    """
    使用聚类算法（KMeans）对用户历史评分行为提取出的特征图谱进行聚类，返回用户的多个兴趣中心。
    """
    if user_ratings.empty:
        return None

    # 找出在特征矩阵中存在对应的movieId
    valid_mask = user_ratings['movieId'].isin(movie_id_to_index.keys())
    valid_ratings = user_ratings[valid_mask]
    
    if valid_ratings.empty:
        return None

    # 计算该用户的真实历史平均分，用作相对基准
    user_mean = float(user_ratings['rating'].mean())
    # 获取最新时间戳作为衰减基准计算
    max_time = float(user_ratings['timestamp'].max()) if 'timestamp' in user_ratings.columns else 0.0

    indices = [movie_id_to_index[mid] for mid in valid_ratings['movieId']]
    vectors_stacked = movie_vectors[indices].toarray()
    
    # 使用相对基准线
    base_weights = valid_ratings['rating'].values - user_mean
    
    # 引入半衰期时间截断机制
    if max_time > 0 and 'timestamp' in valid_ratings.columns:
        timestamps = valid_ratings['timestamp'].fillna(0.0).values
        days_diff = (max_time - timestamps) / (24 * 3600)
        decays = np.exp(-np.log(2) * days_diff / 365.0)
        decays[timestamps == 0] = 1.0  # 对缺失时间戳的回退处理
    else:
        decays = np.ones_like(base_weights)
        
    weights_array = (base_weights * decays).reshape(-1, 1)
    
    weighted_vectors = vectors_stacked * weights_array

    # 聚类的簇数量不能超过样本数
    n_samples = weighted_vectors.shape[0]
    n_clusters = min(3, n_samples)
    
    if n_clusters < 1:
        return np.mean(vectors_stacked, axis=0)
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(weighted_vectors)
    
    # 返回用户的多个兴趣聚类中心，作为新的用户画像
    user_vector = kmeans.cluster_centers_
        
    return user_vector

def rank_by_content_similarity(user_id, candidate_movie_ids):
    """
    提取候选电影的内容特征（类似度），并在线实时计算该用户的加权特征偏好画像。
    """
    # 1. 提取电影特征空间
    movie_vectors, movie_id_to_index = load_movie_features()
    
    # 2. 读取评分数据以供在线计算用户画像 (每次实时拉取该用户的所有评分历史)
    ratings_path = BASE_DIR / "datasets" / "ratings.csv"
    
    # 优化点: 仅读取需要的列以节省内存
    ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'timestamp'], dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # 3. 实时计算当前用户的特征画像 (此时 user_vector 包含多个聚类中心)
    user_vector = calculate_single_user_profile(user_id, user_ratings, movie_vectors, movie_id_to_index)
    
    results = []
    for movie_id in candidate_movie_ids:
        if movie_id in movie_id_to_index and user_vector is not None:
            movie_vector = movie_vectors[movie_id_to_index[movie_id]].toarray()
            # 计算候选电影图谱与用户的多个聚类中心的相似度，并取最大值
            if user_vector.ndim == 2:
                similarities = cosine_similarity(user_vector, movie_vector)
                similarity = np.max(similarities)
            else:
                # 兼容聚类降级为1维的情况
                similarity = cosine_similarity(user_vector.reshape(1, -1), movie_vector)[0][0]
        else:
            similarity = 0.0

        results.append({
            "movie_id": movie_id,
            "similarity": float(similarity)
        })

    return results

if __name__ == "__main__":
    user_id = 1

    print("🚀 Loading SVD model...")
    # predictor = SVDRecommenderPredictor()
    # predictor.load_model()

    # 测试候选集
    candidate_movies = [1, 2, 3]

    print(f"🧠 Refining candidates for User {user_id} with DB content-based similarity...")
    results = rank_by_content_similarity(user_id, candidate_movies)

    print("\n🎬 Raw Features Extracted:")
    for r in results:
        print(r)
