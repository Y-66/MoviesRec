import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from pathlib import Path


class SVDErrorAnalyzer:
    def __init__(self, ratings_rel_path: str = "datasets/ratings.csv"):
        """
        SVD Model Error Analyzer
        """
        self.root_path = self._get_project_root()
        self.ratings_path = self.root_path / ratings_rel_path
        self.output_dir = self.root_path / "analysis_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_df = None

    @staticmethod
    def _get_project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def generate_predictions(self):
        """Train model and generate predictions on a test set for error analysis"""
        if not self.ratings_path.exists():
            raise FileNotFoundError(f"Ratings data not found at: {self.ratings_path}")

        print("1. Loading dataset (using top 200,000 rows for analysis)...")
        # 读取20万条数据以保证有足够的测试样本
        df = pd.read_csv(str(self.ratings_path), usecols=['userId', 'movieId', 'rating'], nrows=200000)

        # Calculate frequencies to analyze cold-start problems later
        user_freq = df['userId'].value_counts().reset_index()
        user_freq.columns = ['userId', 'user_rating_count']

        movie_freq = df['movieId'].value_counts().reset_index()
        movie_freq.columns = ['movieId', 'movie_rating_count']

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

        print("2. Splitting data into 80% Train and 20% Test...")
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        print("3. Training SVD model with balanced parameters (n_factors=50, n_epochs=20)...")
        # 使用上一轮性价比最高的参数组合
        algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
        algo.fit(trainset)

        print("4. Predicting on test set and computing errors...")
        predictions = algo.test(testset)

        # Convert predictions to a DataFrame for analysis
        pred_data = []
        for pred in predictions:
            pred_data.append({
                'userId': pred.uid,
                'movieId': pred.iid,
                'actual': pred.r_ui,
                'predicted': pred.est,
                'error': pred.est - pred.r_ui,
                'abs_error': abs(pred.est - pred.r_ui)
            })

        err_df = pd.DataFrame(pred_data)

        # Merge frequencies
        err_df = err_df.merge(user_freq, on='userId', how='left')
        err_df = err_df.merge(movie_freq, on='movieId', how='left')

        self.error_df = err_df

        csv_path = self.output_dir / "svd_error_analysis.csv"
        self.error_df.to_csv(csv_path, index=False)
        print(f"Prediction errors saved to: {csv_path}")

    def visualize_errors(self):
        """Generate Error Analysis Visualizations"""
        if self.error_df is None:
            raise ValueError("Please run generate_predictions() first.")

        df = self.error_df
        print("\n5. Generating error visualization charts...")
        sns.set_theme(style="whitegrid")

        # ==========================================
        # Plot 1: Overall Error Distribution (Histogram)
        # ==========================================
        plt.figure(figsize=(8, 6))
        sns.histplot(df['error'], bins=50, kde=True, color='purple') # type: ignore
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Plot 1: Distribution of Prediction Errors (Predicted - Actual)', fontsize=14)
        plt.xlabel('Error (Positive = Over-prediction, Negative = Under-prediction)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.savefig(self.output_dir / "error1_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # Plot 2: Prediction Bias by Actual Rating (Violin Plot)
        # ==========================================
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='actual', y='predicted', data=df, palette="muted", inner="quartile")
        # Draw a perfect prediction line (y=x)
        plt.plot([-0.5, 9.5], [0.5, 5.0], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        plt.title('Plot 2: Predicted Spread vs. Actual Ratings', fontsize=14)
        plt.xlabel('Actual Rating', fontsize=12)
        plt.ylabel('Predicted Rating', fontsize=12)
        plt.legend()
        plt.savefig(self.output_dir / "error2_pred_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # Plot 3: Absolute Error by Actual Rating (Box Plot)
        # ==========================================
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='actual', y='abs_error', data=df, palette="coolwarm")
        plt.title('Plot 3: Absolute Error Magnitude by Actual Rating Category', fontsize=14)
        plt.xlabel('Actual Rating', fontsize=12)
        plt.ylabel('Absolute Error (|Predicted - Actual|)', fontsize=12)
        plt.savefig(self.output_dir / "error3_abs_error_by_rating.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # Plot 4: Cold Start Analysis - Error vs User Activity
        # ==========================================
        # Bin users by how many ratings they have given
        df['user_activity_level'] = pd.cut(df['user_rating_count'],
                                           bins=[0, 10, 50, 150, np.inf],
                                           labels=['1-10 (Inactive)', '11-50 (Moderate)', '51-150 (Active)',
                                                   '150+ (Very Active)'])
        plt.figure(figsize=(8, 6))
        sns.barplot(x='user_activity_level', y='abs_error', data=df, palette="Blues_d", errorbar="ci")
        plt.title('Plot 4: Impact of User Activity on Prediction Error (User Cold-Start)', fontsize=14)
        plt.xlabel('User Rating Frequency (Number of ratings by user)', fontsize=12)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.savefig(self.output_dir / "error4_user_activity.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # Plot 5: Item Cold Start Analysis - Error vs Movie Popularity
        # ==========================================
        # Bin movies by how many ratings they received
        df['movie_popularity'] = pd.cut(df['movie_rating_count'],
                                        bins=[0, 10, 50, 200, np.inf],
                                        labels=['1-10 (Niche)', '11-50 (Regular)', '51-200 (Popular)',
                                                '200+ (Blockbuster)'])
        plt.figure(figsize=(8, 6))
        sns.barplot(x='movie_popularity', y='abs_error', data=df, palette="Greens_d", errorbar="ci")
        plt.title('Plot 5: Impact of Movie Popularity on Prediction Error (Item Cold-Start)', fontsize=14)
        plt.xlabel('Movie Popularity (Number of ratings received)', fontsize=12)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.savefig(self.output_dir / "error5_movie_popularity.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All 5 Error Analysis charts generated in: {self.output_dir}")


if __name__ == "__main__":
    analyzer = SVDErrorAnalyzer()
    analyzer.generate_predictions()
    analyzer.visualize_errors()