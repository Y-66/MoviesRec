# test_svd.py
import os
import json
from typing import List, Dict
from surprise import SVD
from surprise import dump


class SVDRecommenderPredictor:
    def __init__(self, model_path: str = "datasets/svd_model.pkl"):
        """
        SVD 模型加载与预测器
        """
        self.model_path = model_path
        self.algo = None

    def load_model(self):
        """
        加载已经训练好的模型
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}。请先运行 train_svd.py。")

        _, self.algo = dump.load(self.model_path)
        print(f"成功从 {self.model_path} 加载 SVD 模型。")


def get_collaborative_candidates(
        algo: SVD,
        user_id: int,
        candidate_movie_ids: List[int],
        top_k: int = 10
) -> List[Dict]:
    """
    根据硬过滤结果进行 SVD 协同过滤打分并排序。

    参数:
        algo (SVD): 已经加载的 Surprise SVD 模型实例。
        user_id (int): 目标用户的 ID [cite: 27]。
        candidate_movie_ids (List[int]): 硬过滤输出的 ID 列表，仅对这个列表中的电影进行协同过滤算法 [cite: 28]。
        top_k (int, 可选): 召回的数量（例如先选出 10 部），默认为 10 [cite: 29]。

    输出:
        List[Dict]: 排名前k个电影id及其评分 [cite: 30, 31]。格式如:
        [
            {"movie_id": 12, "svd_score": 4.5},
            {"movie_id": 13, "svd_score": 3.7}
        ]
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
        exit(1)  # 如果模型没找到，直接退出，提示用户先去运行训练脚本

    # 2. 模拟输入数据
    # 假设我们要为 user_id = 1 推荐 [cite: 27]
    target_user_id = 1

    # 假设这些是经过"召回层 (SQLite硬规则过滤)"后剩下的电影 ID 列表 [cite: 28]
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