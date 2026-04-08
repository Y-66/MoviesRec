from logging import root

import pandas as pd
import sqlite3
from pathlib import Path

# 1. 读取 links.csv
root_dir = Path(__file__).resolve().parent.parent
links_path = root_dir / 'datasets' / 'links.csv'
links_df = pd.read_csv(links_path)

# 清理空值并确保类型
links_df = links_df.dropna(subset=['tmdbId'])
links_df['tmdbId'] = links_df['tmdbId'].astype(int)

# 2. 连接数据库
conn = sqlite3.connect(root_dir / 'db' / 'movies_data.db')
cursor = conn.cursor()

# 3. 在 movies 表中新增 tmdbId 列
try:
    cursor.execute("ALTER TABLE movies ADD COLUMN tmdbId INTEGER")
except sqlite3.OperationalError:
    pass # 列已存在则跳过

print("正在为 movieId 创建索引以加速查询...")
# 【关键优化 1】建立索引：这会将查询速度提升成百上千倍
cursor.execute("CREATE INDEX IF NOT EXISTS idx_movie_id ON movies(movieId)")

print("正在将 tmdbId 批量更新到数据库...")
# 【关键优化 2】准备批量更新的数据格式 (List of Tuples)
# 将两列数据打包成 [(tmdbId_1, movieId_1), (tmdbId_2, movieId_2), ...] 的形式
data_to_update = list(zip(links_df['tmdbId'], links_df['movieId']))

# 【关键优化 3】使用 executemany 替代 for 循环
# 一次性将所有数据抛给 SQLite 底层 C 语言处理，效率极高
cursor.executemany(
    "UPDATE movies SET tmdbId = ? WHERE movieId = ?",
    data_to_update
)

# 提交并关闭
conn.commit()
conn.close()
print("ID 映射更新完成！")