import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

# 导入现有的特征提取逻辑
from feature_extraction import load_movie_features, BASE_DIR

# 设置可视化样式
sns.set_theme(style="whitegrid")

def analyze_and_visualize_user_clusters(user_id=1):
    """
    对指定用户的观影历史进行聚类，并将高维特征降维到 2D 空间进行可视化，
    最后进行结果分析并输出报告。
    """
    print(f"🔍 Loading data for User {user_id}...")
    movie_vectors, movie_id_to_index = load_movie_features()
    
    ratings_path = BASE_DIR / "datasets" / "ratings.csv"
    ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'timestamp'], dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
    user_ratings = ratings[ratings['userId'] == user_id]
    
    if user_ratings.empty:
        print(f"⚠️ No ratings found for User {user_id}.")
        return

    # 过滤出存在于特征矩阵中的movieId
    valid_mask = user_ratings['movieId'].isin(movie_id_to_index.keys())
    valid_ratings = user_ratings[valid_mask]
    
    if valid_ratings.empty:
        print("⚠️ No valid movies found for this user in the feature matrix.")
        return

    # --- 计算权重 (复用 feature_extraction.py 中的数据预处理逻辑) ---
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
    
    # 最终加权后的特征分布
    weighted_vectors = vectors_stacked * weights_array

    # --- 进行聚类 ---
    n_samples = weighted_vectors.shape[0]
    n_clusters = min(3, n_samples)
    
    if n_clusters < 1:
        print("⚠️ Not enough samples for clustering.")
        return

    print(f"🧠 Running KMeans with {n_clusters} clusters on {n_samples} user-rated movies...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(weighted_vectors)
    cluster_centers = kmeans.cluster_centers_

    # --- 降维与可视化 (使用 PCA 将高维特征降低到 2D) ---
    pca = PCA(n_components=2, random_state=42)
    reduced_vectors = pca.fit_transform(weighted_vectors)
    reduced_centers = pca.transform(cluster_centers)
    
    # 组装用于绘图的数据
    df_plot = pd.DataFrame(reduced_vectors, columns=['PCA1', 'PCA2'])
    df_plot['Cluster'] = [f"Cluster {label}" for label in cluster_labels]
    df_plot['MovieID'] = valid_ratings['movieId'].values
    df_plot['Rating'] = valid_ratings['rating'].values
    
    # 开始绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制用户的各个电影点
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
    
    # 绘制聚类中心点
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
    
    # 保存图片
    output_image = BASE_DIR / "scripts" / f"user_{user_id}_clusters.png"
    plt.savefig(output_image)
    print(f"📊 Visualization saved to {output_image}")
    
    # 在屏幕上展示
    plt.show()

    # --- 简单结果分析报告 ---
    print("\n" + "="*40)
    print("📈 聚类结果分析报告")
    print("="*40)
    print(f"总计分析电影数: {n_samples}")
    print(f"用户总体平均分: {user_mean:.2f}\n")
    
    for i in range(n_clusters):
        cluster_movies = df_plot[df_plot['Cluster'] == f"Cluster {i}"]
        avg_rating = cluster_movies['Rating'].mean()
        high_rated = cluster_movies.sort_values(by='Rating', ascending=False)
        top_movie_ids = high_rated['MovieID'].head(3).tolist()
        
        print(f"🌟 中心 {i+1} (Cluster {i}):")
        print(f"   - 包含电影数量: {len(cluster_movies)}部 ({len(cluster_movies)/n_samples:.1%})")
        print(f"   - 该簇平均评分: {avg_rating:.2f}")
        print(f"   - 兴趣代表电影IDs (Top 3): {top_movie_ids}")
        print("-"*40)

if __name__ == "__main__":
    # 你可以修改这里的 user_id 来测试不同用户的聚类分析
    analyze_and_visualize_user_clusters(user_id=1)
