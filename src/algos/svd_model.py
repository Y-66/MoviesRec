# train_svd.py
import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise import dump


class SVDRecommenderTrainer:
    def __init__(self, model_path: str = "datasets/svd_model.pkl"):
        """
        SVD 模型训练器
        """
        self.model_path = model_path
        self.algo = None

    def train_and_save_model(self, ratings_file: str = "datasets/ratings.csv"):
        """
        离线训练：读取原始的 ratings.csv 训练 SVD 模型，并保存到本地
        """
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"找不到评分数据文件: {ratings_file}。请确保路径正确。")

        print(f"正在从 {ratings_file} 加载评分数据...")
        # SVD 只需要 userId, movieId, rating 这三列原始交互数据
        df = pd.read_csv(ratings_file, usecols=['userId', 'movieId', 'rating'])

        # 定义评分范围（通常是 0.5 到 5.0）
        reader = Reader(rating_scale=(0.5, 5.0))

        # 将 DataFrame 转换为 Surprise 支持的数据集格式
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

        print("正在构建全量训练集...")
        trainset = data.build_full_trainset()

        print("正在训练 SVD 模型 (这可能需要几分钟时间)...")
        # 初始化 SVD 算法模型
        self.algo = SVD()
        self.algo.fit(trainset)

        print(f"模型训练完成！正在保存至 {self.model_path}...")
        # 确保 datasets 文件夹存在
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        dump.dump(self.model_path, algo=self.algo)
        print("模型保存成功！")


if __name__ == "__main__":
    # 执行训练
    trainer = SVDRecommenderTrainer()
    trainer.train_and_save_model()