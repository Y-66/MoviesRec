import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from pathlib import Path


class SVDEvaluator:
    def __init__(self, ratings_rel_path: str = "datasets/ratings.csv"):
        """
        SVD Model Evaluator and Visualizer
        """
        self.root_path = self._get_project_root()
        self.ratings_path = self.root_path / ratings_rel_path
        self.output_dir = self.root_path / "analysis_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_df = None

    @staticmethod
    def _get_project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def run_grid_search(self) -> pd.DataFrame:
        """
        Run Grid Search Cross-Validation
        """
        if not self.ratings_path.exists():
            raise FileNotFoundError(f"Ratings data not found at: {self.ratings_path}")

        print("1. Loading dataset (using top 100,000 rows for faster evaluation)...")
        df = pd.read_csv(str(self.ratings_path), usecols=['userId', 'movieId', 'rating'], nrows=100000)
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

        print("2. Configuring parameter grid...")
        param_grid = {
            'n_factors': [20, 50, 100],
            'n_epochs': [10, 20, 30],
            'lr_all': [0.002, 0.005],
            'reg_all': [0.02, 0.05]
        }

        print("3. Running Grid Search with 3-fold CV...")
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
        gs.fit(data)

        print(f"\nBest RMSE Score: {gs.best_score['rmse']:.4f}")
        print(f"Best Parameters: {gs.best_params['rmse']}")

        self.results_df = pd.DataFrame.from_dict(gs.cv_results)

        csv_path = self.output_dir / "svd_grid_search_results.csv"
        self.results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

        return self.results_df

    def visualize_results(self):
        """Generate and save visualizations in English"""
        if self.results_df is None:
            raise ValueError("Please run run_grid_search() first.")

        df = self.results_df
        print("\n4. Generating analytical charts...")

        sns.set_theme(style="whitegrid")

        # ==========================================
        # Plot 1: Impact of n_factors on RMSE (Boxplot)
        # ==========================================
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='param_n_factors', y='mean_test_rmse', data=df, palette="Set2")
        plt.title('Plot 1: RMSE Distribution across Latent Factors (n_factors)', fontsize=14)
        plt.xlabel('Number of Latent Factors (n_factors)', fontsize=12)
        plt.ylabel('Mean Test RMSE', fontsize=12)
        plt.savefig(self.output_dir / "plot1_n_factors_impact.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # Plot 2: Trend of n_epochs on RMSE (Lineplot)
        # ==========================================
        plt.figure(figsize=(8, 6))
        sns.lineplot(x='param_n_epochs', y='mean_test_rmse', hue='param_lr_all', marker="o", data=df)
        plt.title('Plot 2: Impact of Epochs and Learning Rate on RMSE', fontsize=14)
        plt.xlabel('Number of Epochs (n_epochs)', fontsize=12)
        plt.ylabel('Mean Test RMSE', fontsize=12)
        plt.legend(title='Learning Rate (lr_all)')
        plt.savefig(self.output_dir / "plot2_n_epochs_trend.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # Plot 3: Heatmap of Learning Rate vs. Regularization
        # ==========================================
        subset = df[(df['param_n_factors'] == 50) & (df['param_n_epochs'] == 20)]
        if not subset.empty:
            # We must cast indices/columns to float/string so pivot works smoothly in all Pandas versions
            subset = subset.copy()
            subset['param_lr_all'] = subset['param_lr_all'].astype(float)
            subset['param_reg_all'] = subset['param_reg_all'].astype(float)
            pivot_table = subset.pivot(index='param_lr_all', columns='param_reg_all', values='mean_test_rmse')

            plt.figure(figsize=(6, 5))
            sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Mean Test RMSE'})
            plt.title('Plot 3: RMSE Heatmap (Learning Rate vs. Regularization)\nCondition: n_factors=50, n_epochs=20',
                      fontsize=12)
            plt.xlabel('Regularization Term (reg_all)', fontsize=12)
            plt.ylabel('Learning Rate (lr_all)', fontsize=12)
            plt.savefig(self.output_dir / "plot3_lr_reg_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

        # ==========================================
        # Plot 4: RMSE vs MAE Comparison
        # ==========================================
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='mean_test_mae', y='mean_test_rmse', hue='param_n_factors', size='param_n_epochs',
                        sizes=(50, 200), alpha=0.7, palette="viridis", data=df)
        plt.title('Plot 4: Accuracy Metrics Comparison (RMSE vs MAE)', fontsize=14)
        plt.xlabel('Mean Test MAE', fontsize=12)
        plt.ylabel('Mean Test RMSE', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Latent Factors')
        plt.savefig(self.output_dir / "plot4_rmse_vs_mae.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # Plot 5: Time vs RMSE Trade-off
        # ==========================================
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='mean_fit_time', y='mean_test_rmse', hue='param_n_epochs', style='param_n_factors',
                        s=100, palette="autumn", data=df)
        plt.title('Plot 5: Trade-off between Training Time and RMSE', fontsize=14)
        plt.xlabel('Mean Fit Time (Seconds)', fontsize=12)
        plt.ylabel('Mean Test RMSE', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Epochs')
        plt.savefig(self.output_dir / "plot5_time_vs_rmse.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All 5 visualization charts generated in: {self.output_dir}")


if __name__ == "__main__":
    evaluator = SVDEvaluator()
    evaluator.run_grid_search()
    evaluator.visualize_results()