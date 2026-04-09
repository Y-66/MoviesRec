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
    Unified feature engineering loading function.
    For future performance consideration, this function's return value can be cached in a production environment.
    """
    # Read data
    movies = pd.read_csv(MOVIES_PATH)

    # Genres → One-hot (sparse)
    movies['genres'] = movies['genres'].fillna('')
    cv = CountVectorizer(token_pattern=r'[^|]+')
    genres_sparse = cv.fit_transform(movies['genres'])

    movie_ids = movies['movieId'].values

    # Tags → TF-IDF (sparse)
    movies['tags'] = movies['tags'].fillna('')
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(movies['tags'])

    # Merge features
    movie_vectors = hstack([genres_sparse, tfidf_matrix]).tocsr()
    movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    return movie_vectors, movie_id_to_index

from sklearn.cluster import KMeans

def calculate_single_user_profile(uid, user_ratings, movie_vectors, movie_id_to_index):
    """
    Use clustering algorithm (KMeans) to cluster the feature graph extracted from user's historical rating behavior,
    returning multiple interest centers for the user.
    """
    if user_ratings.empty:
        return None

    # Find movieIds that have a corresponding entry in the feature matrix
    valid_mask = user_ratings['movieId'].isin(movie_id_to_index.keys())
    valid_ratings = user_ratings[valid_mask]
    
    if valid_ratings.empty:
        return None

    # Calculate the user's true historical average rating, used as a relative baseline
    user_mean = float(user_ratings['rating'].mean())
    # Get the latest timestamp as a baseline for decay calc
    max_time = float(user_ratings['timestamp'].max()) if 'timestamp' in user_ratings.columns else 0.0

    indices = [movie_id_to_index[mid] for mid in valid_ratings['movieId']]
    vectors_stacked = movie_vectors[indices].toarray()
    
    # Use relative baseline
    base_weights = valid_ratings['rating'].values - user_mean
    
    # Introduce half-life time truncation mechanism
    if max_time > 0 and 'timestamp' in valid_ratings.columns:
        timestamps = valid_ratings['timestamp'].fillna(0.0).values
        days_diff = (max_time - timestamps) / (24 * 3600)
        decays = np.exp(-np.log(2) * days_diff / 365.0)
        decays[timestamps == 0] = 1.0  # Fallback handling for missing timestamps
    else:
        decays = np.ones_like(base_weights)
        
    weights_array = (base_weights * decays).reshape(-1, 1)
    
    weighted_vectors = vectors_stacked * weights_array

    # Number of clusters cannot exceed number of samples
    n_samples = weighted_vectors.shape[0]
    n_clusters = min(3, n_samples)
    
    if n_clusters < 1:
        return np.mean(vectors_stacked, axis=0)
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(weighted_vectors)
    
    # Return multiple interest cluster centers as the new user profile
    user_vector = kmeans.cluster_centers_
        
    return user_vector

def rank_by_content_similarity(user_id, candidate_movie_ids):
    """
    Extract content features (similarity) of candidate movies, and compute the user's weighted feature preference profile online in real-time.
    """
    # 1. Extract movie feature space
    movie_vectors, movie_id_to_index = load_movie_features()
    
    # 2. Read rating data for online calc of user profile (pulling all historical ratings for this user in real time)
    ratings_path = BASE_DIR / "datasets" / "ratings.csv"
    
    # Optimization: Read only needed columns to save memory
    ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'timestamp'], dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # 3. Real-time compute current user's feature profile (now user_vector contains multiple cluster centers)
    user_vector = calculate_single_user_profile(user_id, user_ratings, movie_vectors, movie_id_to_index)
    
    results = []
    for movie_id in candidate_movie_ids:
        if movie_id in movie_id_to_index and user_vector is not None:
            movie_vector = movie_vectors[movie_id_to_index[movie_id]].toarray()
            # Calculate similarity between candidate movie graph and user's multiple cluster centers, then take max
            if user_vector.ndim == 2:
                similarities = cosine_similarity(user_vector, movie_vector)
                similarity = np.max(similarities)
            else:
                # Handle fallback where clustering degrades to 1 dimension
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

    # Test candidate set
    candidate_movies = [1, 2, 3]

    print(f"🧠 Refining candidates for User {user_id} with DB content-based similarity...")
    results = rank_by_content_similarity(user_id, candidate_movies)

    print("\n🎬 Raw Features Extracted:")
    for r in results:
        print(r)
