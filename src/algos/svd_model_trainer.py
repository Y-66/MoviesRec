import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise import dump
from pathlib import Path

class SVDRecommenderTrainer:
    def __init__(self, model_rel_path: str = "models/svd_model.pkl"):
        """
        SVD Model Trainer
        """
        # Unify using encapsulated method to get root directory
        self.root_path = self._get_project_root()
        
        # Concat complete model loading path
        self.model_path = self.root_path / model_rel_path
        self.algo = None

    @staticmethod
    def _get_project_root() -> Path:
        """
        Encapsulate method to get project root directory
        Assuming script location: Project_Root/src/algos/train_svd.py
        """
        return Path(__file__).resolve().parents[2]

    def train_and_save_model(self, ratings_rel_path: str = "datasets/ratings.csv"):
        """
        Offline training: automatically locate dataset and save model
        """
        # 1. Auto-locate absolute path of ratings file
        full_ratings_path = self.root_path / ratings_rel_path

        if not full_ratings_path.exists():
            raise FileNotFoundError(f"Ratings data file not found: {full_ratings_path}")

        print(f"Loading ratings data: {full_ratings_path}")
        
        # 2. Read data (recommend explicitly specifying str path to ensure compatibility)
        df = pd.read_csv(str(full_ratings_path), usecols=['userId', 'movieId', 'rating'])

        # 3. Prepare data format
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

        print("Building full trainset...")
        trainset = data.build_full_trainset()

        print("Training SVD model...")
        self.algo = SVD()
        self.algo.fit(trainset)

        # 4. Save model
        print(f"Training completed! Saving to: {self.model_path}")
        
        # Ensure parent directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        dump.dump(str(self.model_path), algo=self.algo)
        print("Model saved successfully!")

if __name__ == "__main__":
    trainer = SVDRecommenderTrainer()
    # No need to manually pass arguments, it will automatically look for [Root]/datasets/ratings.csv
    trainer.train_and_save_model()