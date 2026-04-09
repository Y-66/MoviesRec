import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

# Import existing feature extraction logic
from feature_extraction import load_movie_features, BASE_DIR

# Set visualization style
sns.set_theme(style="whitegrid")

def analyze_and_visualize_user_clusters(user_id=1):
    """
    Perform clustering on a specified user's viewing history, reduce high-dimensional features
    to 2D space for visualization, and finally output an analysis report.
    """
    print(f"🔍 Loading data for User {user_id}...")
    movie_vectors, movie_id_to_index = load_movie_features()
    
    ratings_path = BASE_DIR / "datasets" / "ratings.csv"
    ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'timestamp'], dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
    user_ratings = ratings[ratings['userId'] == user_id]
    
    if user_ratings.empty:
        print(f"⚠️ No ratings found for User {user_id}.")
        return

    # Filter for movieIds present in the feature matrix
    valid_mask = user_ratings['movieId'].isin(movie_id_to_index.keys())
    valid_ratings = user_ratings[valid_mask]
    
    if valid_ratings.empty:
        print("⚠️ No valid movies found for this user in the feature matrix.")
        return

    # --- Calculate weights (reusing data preprocessing logic from feature_extraction.py) ---
    user_mean = float(valid_ratings['rating'].mean())
    max_time = float(valid_ratings['timestamp'].max()) if 'timestamp' in valid_ratings.columns else 0.0

    indices = [movie_id_to_index[mid] for mid in valid_ratings['movieId']]
    vectors_stacked = movie_vectors[indices].toarray()
    
    base_weights = valid_ratings['rating'].to_numpy(dtype=np.float32) - user_mean
    
    if max_time > 0 and 'timestamp' in valid_ratings.columns:
        timestamps = valid_ratings['timestamp'].fillna(0.0).to_numpy(dtype=np.float64)
        days_diff = (max_time - timestamps) / (24 * 3600)
        decays = np.exp(-np.log(2) * days_diff / 365.0)
        decays[timestamps == 0] = 1.0
    else:
        decays = np.ones_like(base_weights)
        
    weights_array = (base_weights * decays).reshape(-1, 1)
    
    # Final weighted feature distribution
    weighted_vectors = vectors_stacked * weights_array

    # --- Perform clustering ---
    n_samples = weighted_vectors.shape[0]
    n_clusters = min(3, n_samples)
    
    if n_clusters < 1:
        print("⚠️ Not enough samples for clustering.")
        return

    print(f"🧠 Running KMeans with {n_clusters} clusters on {n_samples} user-rated movies...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(weighted_vectors)
    cluster_centers = kmeans.cluster_centers_

    # --- Dimensionality reduction & Visualization (use PCA to reduce to 2D) ---
    pca = PCA(n_components=2, random_state=42)
    reduced_vectors = pca.fit_transform(weighted_vectors)
    reduced_centers = pca.transform(cluster_centers)
    
    # Assemble data for plotting
    df_plot = pd.DataFrame(reduced_vectors, columns=['PCA1', 'PCA2'])
    df_plot['Cluster'] = [f"Cluster {label}" for label in cluster_labels]
    df_plot['MovieID'] = valid_ratings['movieId'].values
    df_plot['Rating'] = valid_ratings['rating'].values
    
    # Start plotting
    plt.figure(figsize=(10, 6))
    
    # Plot user's movie data points
    scatter = sns.scatterplot(
        data=df_plot, 
        x='PCA1', 
        y='PCA2', 
        hue='Cluster', 
        palette='viridis', 
        size='Rating',
        sizes=(30, 150),
        alpha=0.7,
        edgecolor='w'
    )
    
    # Plot cluster centers
    plt.scatter(
        reduced_centers[:, 0], 
        reduced_centers[:, 1], 
        c='red', 
        marker='X', 
        s=200, 
        label='Cluster Centers',
        edgecolor='white',
        linewidth=2
    )
    
    plt.title(f'User {user_id} Interest Clusters (PCA Projection)', fontsize=14)
    plt.xlabel(f'PCA1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PCA2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save image
    output_image = BASE_DIR / "scripts" / f"user_{user_id}_clusters.png"
    plt.savefig(output_image)
    print(f"📊 Visualization saved to {output_image}")
    
    # Show on screen
    plt.show()

    # --- Simple Result Analysis Report ---
    print("\n" + "="*40)
    print("📈 Clustering Result Analysis Report")
    print("="*40)
    print(f"Total Movies Analyzed: {n_samples}")
    print(f"User Overall Avg Rating: {user_mean:.2f}\n")
    
    for i in range(n_clusters):
        cluster_movies = df_plot[df_plot['Cluster'] == f"Cluster {i}"]
        avg_rating = cluster_movies['Rating'].mean()
        high_rated = cluster_movies.sort_values(by='Rating', ascending=False)
        top_movie_ids = high_rated['MovieID'].head(3).tolist()
        
        print(f"🌟 Center {i+1} (Cluster {i}):")
        print(f"   - Movies In Cluster: {len(cluster_movies)} ({len(cluster_movies)/n_samples:.1%})")
        print(f"   - Cluster Avg Rating: {avg_rating:.2f}")
        print(f"   - Representative Movie IDs (Top 3): {top_movie_ids}")
        print("-"*40)

if __name__ == "__main__":
    # You can change user_id here to test clustering analysis for different users
    analyze_and_visualize_user_clusters(user_id=1)
