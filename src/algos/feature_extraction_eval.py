import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
from collections import defaultdict

# =========================
# 1. Load Data (Keep original path)
# =========================
movies = pd.read_csv(r"C:\Users\21977\Desktop\MoviesRec-main\MoviesRec-main\datasets\datasets\movies_aggregated.csv")
ratings = pd.read_csv(r"C:\Users\21977\Desktop\MoviesRec-main\MoviesRec-main\datasets\datasets\ratings.csv")

# =========================
# 2. Three Feature Engineering Methods
# =========================
def build_genre_only():
    cv = CountVectorizer(token_pattern=r'[^|]+')
    return cv.fit_transform(movies['genres'].fillna(''))

def build_tfidf_only(max_features=5000):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    return tfidf.fit_transform(movies['tags'].fillna(''))

def build_hybrid(max_features=5000):
    genre = build_genre_only()
    tfidf = build_tfidf_only(max_features)
    return hstack([genre, tfidf])

# =========================
# 3. User Profile (Fix np.matrix🔥)
# =========================
def get_user_profile(user_id, movie_vectors):
    user_data = ratings[ratings['userId'] == user_id]
    liked = user_data[user_data['rating'] >= 4]['movieId'].values

    indices = [i for i, mid in enumerate(movies['movieId']) if mid in liked]

    if len(indices) == 0:
        return None

    user_vec = movie_vectors[indices].mean(axis=0)

    return np.array(user_vec)   # ✅ Critical fix

# =========================
# 4. Recommendation Function (Fix sparse🔥)
# =========================
def recommend(user_id, movie_vectors, top_k=10):
    user_vec = get_user_profile(user_id, movie_vectors)

    if user_vec is None:
        return []

    scores = []

    for i, mid in enumerate(movies['movieId']):
        movie_vec = movie_vectors[i].toarray()   # ✅ Critical fix
        sim = cosine_similarity(user_vec, movie_vec)[0][0]
        scores.append((mid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scores[:top_k]]

# =========================
# 5. Evaluation Metrics
# =========================
def precision_at_k(recommended, relevant):
    if len(recommended) == 0:
        return 0
    return len(set(recommended) & set(relevant)) / len(recommended)

def recall_at_k(recommended, relevant):
    if len(relevant) == 0:
        return 0
    return len(set(recommended) & set(relevant)) / len(relevant)

# =========================
# 6. Experiments
# =========================
def run_experiment():
    users = ratings['userId'].unique()[:50]

    results = defaultdict(list)

    methods = {
        "Genre": build_genre_only(),
        "TF-IDF": build_tfidf_only(),
        "Hybrid": build_hybrid()
    }

    for name, vectors in methods.items():
        print(f"Running {name}...")

        precisions = []
        recalls = []

        for user in users:
            user_data = ratings[ratings['userId'] == user]
            relevant = user_data[user_data['rating'] >= 4]['movieId'].values

            recs = recommend(user, vectors)

            precisions.append(precision_at_k(recs, relevant))
            recalls.append(recall_at_k(recs, relevant))

        results["method"].append(name)
        results["precision"].append(np.mean(precisions))
        results["recall"].append(np.mean(recalls))

    return pd.DataFrame(results)

# =========================
# 7. Visualization (Enhanced🔥)
# =========================
def plot_results(df):
    plt.figure()
    plt.bar(df['method'], df['precision'])
    plt.title("Precision@10 Comparison")
    plt.xlabel("Method")
    plt.ylabel("Precision")
    plt.show()

    plt.figure()
    plt.bar(df['method'], df['recall'])
    plt.title("Recall@10 Comparison")
    plt.xlabel("Method")
    plt.ylabel("Recall")
    plt.show()

    # 🔥 Added: Comparison Chart (Bonus)
    plt.figure()
    x = np.arange(len(df['method']))
    width = 0.35

    plt.bar(x - width/2, df['precision'], width, label='Precision')
    plt.bar(x + width/2, df['recall'], width, label='Recall')

    plt.xticks(x, df['method'])
    plt.title("Precision vs Recall Comparison")
    plt.legend()
    plt.show()

# =========================
# 8. Main Program
# =========================
if __name__ == "__main__":
    df = run_experiment()
    print(df)
    plot_results(df)