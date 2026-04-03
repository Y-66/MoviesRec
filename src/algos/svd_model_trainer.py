import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise import dump
from pathlib import Path

class SVDRecommenderTrainer:
    def __init__(self, model_rel_path: str = "models/svd_model.pkl"):
        """
        SVD 模型训练器
        """
        # 统一使用封装的方法获取根目录
        self.root_path = self._get_project_root()
        
        # 拼接完整的模型保存路径
        self.model_path = self.root_path / model_rel_path
        self.algo = None

    @staticmethod
    def _get_project_root() -> Path:
        """
        封装获取项目根目录的方法
        假设脚本位置: Project_Root/src/algos/train_svd.py
        """
        return Path(__file__).resolve().parents[2]

    def train_and_save_model(self, ratings_rel_path: str = "datasets/ratings.csv"):
        """
        离线训练：自动定位数据集并保存模型
        """
        # 1. 自动定位评分文件绝对路径
        full_ratings_path = self.root_path / ratings_rel_path

        if not full_ratings_path.exists():
            raise FileNotFoundError(f"找不到评分数据文件: {full_ratings_path}")

        print(f"正在加载评分数据: {full_ratings_path}")
        
        # 2. 读取数据 (建议显式指定 str 路径以确保兼容性)
        df = pd.read_csv(str(full_ratings_path), usecols=['userId', 'movieId', 'rating'])

        # 3. 准备数据格式
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

        print("正在构建全量训练集...")
        trainset = data.build_full_trainset()

        print("正在训练 SVD 模型...")
        self.algo = SVD()
        self.algo.fit(trainset)

        # 4. 保存模型
        print(f"训练完成！正在保存至: {self.model_path}")
        
        # 确保父级目录存在
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        dump.dump(str(self.model_path), algo=self.algo)
        print("模型保存成功！")

if __name__ == "__main__":
    trainer = SVDRecommenderTrainer()
    # 无需手动传参，它会自动去找 [根目录]/datasets/ratings.csv
    trainer.train_and_save_model()