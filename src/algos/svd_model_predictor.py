# test_svd.py
import os
import json
from typing import List, Dict
from surprise import SVD
from surprise import dump
from pathlib import Path
import heapq

# Singleton global cache, avoids repeating loading 1.3GB+ model that leads to MemoryError
_svd_model_cache = None

class SVDRecommenderPredictor:
    def __init__(self, model_rel_path: str = "models/svd_model.pkl"):
        """
        SVD Model Loader and Predictor
        """
        # Use encapsulated method to get root directory
        self.root_path = self._get_project_root()
        # Concat complete model path
        self.model_path = self.root_path / model_rel_path
        self.algo = None

    @staticmethod
    def _get_project_root() -> Path:
        """
        Encapsulate method to get project root directory
        Assuming script location: Project_Root/src/algos/test_svd.py
        """
        return Path(__file__).resolve().parents[2]

    def load_model(self):
        """
        Load the pre-trained model
        """
        global _svd_model_cache
        
        # Use pathlib's exists() method
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}. Please run train_svd.py first.")

        if _svd_model_cache is not None:
            self.algo = _svd_model_cache
            return

        # Convert to string path to ensure compatibility, and load the model
        _, self.algo = dump.load(str(self.model_path))
        _svd_model_cache = self.algo
        print(f"Successfully loaded SVD model from {self.model_path}.")


def get_collaborative_candidates(
        algo: SVD,
        user_id: int,
        candidate_movie_ids: List[int],
        top_k: int = 10
) -> List[Dict]:
    """
    Perform SVD collaborative filtering scoring and sorting based on hard filtering results.
    """
    if algo is None:
        raise ValueError("SVD model not loaded, please pass a valid algo instance first.")

    predict = algo.predict

    # Use generator to calculate scores on demand, avoiding single massive allocation in memory
    # Extract predict.est attribute as score
    scores = ((movie_id, predict(uid=user_id, iid=movie_id).est)
              for movie_id in candidate_movie_ids)

    # Use heapq.nlargest to get the top_k candidate values
    # Time complexity reduced from full sort O(N log N) to O(N log K), greatly improving performance
    top_candidates = heapq.nlargest(top_k, scores, key=lambda x: x[1])

    # Only construct dictionary format and truncate decimals for top_k results returned, further saving memory allocation
    return [
        {"movie_id": int(movie_id), "svd_score": round(float(score), 3)}
        for movie_id, score in top_candidates
    ]


if __name__ == "__main__":
    # 1. Initialize predictor and load model
    predictor = SVDRecommenderPredictor()
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(e)
        exit(1)  # Direct exit if model not found

    # 2. Mock input data
    target_user_id = 1
    hard_filtered_movie_ids = [1, 50, 110, 260, 318, 527, 593, 858, 1196, 1210, 2571, 2959, 356, 4993, 7153]

    # 3. Call core function to get sorted results
    print(f"\nPredicting ratings for user {target_user_id}...")
    result = get_collaborative_candidates(
        algo=predictor.algo,
        user_id=target_user_id,
        candidate_movie_ids=hard_filtered_movie_ids,
        top_k=5  # Get Top 5
    )

    # 4. Print output
    print("\n--- SVD Collaborative Filtering Output Results ---")
    print(json.dumps(result, indent=4))