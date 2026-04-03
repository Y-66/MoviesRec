# test_svd.py
import os
import json
from typing import List, Dict
from surprise import SVD
from surprise import dump
from pathlib import Path

class SVDRecommenderPredictor:
    def __init__(self, model_rel_path: str = "models/svd_model.pkl"):
        """
        SVD 模型加载与预测器
        """
        # 使用封装的方法获取根目录
        self.root_path = self._get_project_root()
        # 拼接完整的模型路径
        self.model_path = self.root_path / model_rel_path
        self.algo = None

    @staticmethod
    def _get_project_root() -> Path:
        """
        封装获取项目根目录的方法
        假设脚本位置: Project_Root/src/algos/test_svd.py
        """
        return Path(__file__).resolve().parents[2]

    def load_model(self):
        """
        加载已经训练好的模型
        """
        # 使用 pathlib 的 exists() 方法
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}。请先运行 train_svd.py。")

        # 转换为字符串路径以确保兼容性，并加载模型
        _, self.algo = dump.load(str(self.model_path))
        print(f"成功从 {self.model_path} 加载 SVD 模型。")


def get_collaborative_candidates(
        algo: SVD,
        user_id: int,
        candidate_movie_ids: List[int],
        top_k: int = 10
) -> List[Dict]:
    """
    根据硬过滤结果进行 SVD 协同过滤打分并排序。
    """
    if algo is None:
        raise ValueError("SVD 模型未加载，请先传入有效的 algo 实例。")

    predictions = []

    # 仅对传入的 candidate_movie_ids 列表中的电影进行预测
    for movie_id in candidate_movie_ids:
        # 使用 SVD 模型预测该用户对该电影的评分
        pred = algo.predict(uid=user_id, iid=movie_id)

        predictions.append({
            "movie_id": int(movie_id),
            "svd_score": round(float(pred.est), 3)  # 保留3位小数
        })

    # 根据 svd_score 进行降序排序 (从高到低)
    predictions.sort(key=lambda x: x["svd_score"], reverse=True)

    # 截取前 top_k 个返回
    return predictions[:top_k]


if __name__ == "__main__":
    # 1. 初始化预测器并加载模型
    predictor = SVDRecommenderPredictor()
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(e)
        exit(1)  # 如果模型没找到，直接退出

    # 2. 模拟输入数据
    target_user_id = 1
    hard_filtered_movie_ids = [1, 50, 110, 260, 318, 527, 593, 858, 1196, 1210, 2571, 2959, 356, 4993, 7153]

    # 3. 调用核心函数得到排序结果
    print(f"\n正在为用户 {target_user_id} 预测评分...")
    result = get_collaborative_candidates(
        algo=predictor.algo,
        user_id=target_user_id,
        candidate_movie_ids=hard_filtered_movie_ids,
        top_k=5  # 取 Top 5
    )

    # 4. 打印输出
    print("\n--- SVD 协同过滤输出结果 ---")
    print(json.dumps(result, indent=4))