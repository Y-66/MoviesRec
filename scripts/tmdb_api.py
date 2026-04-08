import sqlite3
import requests
import time
from typing import List, Dict

# ================= 配置区域 =================
DB_PATH = 'db/movies_data.db'
API_KEY = '2fc4340de0512e93d9982b33593fe440'
BASE_URL = 'https://api.themoviedb.org/3/movie/'
IMG_BASE_URL = 'https://image.tmdb.org/t/p/w500'


# ============================================

def fetch_movie_data(tmdb_id):
    """通过 TMDb API 获取电影数据 (底层调用)"""
    url = f"{BASE_URL}{tmdb_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=5)  # 缩短超时时间，避免前端等太久
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            poster_url = f"{IMG_BASE_URL}{poster_path}" if poster_path else None
            overview = data.get('overview')
            title = data.get('title')
            release_date = data.get('release_date')
            popularity = data.get('popularity')
            return poster_url, overview, title, release_date, popularity
        elif response.status_code == 429:
            time.sleep(1)
            return fetch_movie_data(tmdb_id)
    except Exception as e:
        print(f"  [X] API 请求错误 tmdbId {tmdb_id}: {e}")
    return None, None, None, None, None


def get_and_store_movie_details(movie_ids: List[int]) -> Dict[int, Dict[str, str]]:
    """
    【核心对外接口】按需获取推荐电影的海报和简介。
    工作流：检查数据库 -> 如果没有海报 -> 调 API 获取 -> 存入数据库 -> 返回结果

    参数:
        movie_ids (List[int]): 推荐系统输出的电影 ID 列表 (例如: [1, 260, 318])

    返回:
        Dict: 包含海报和简介的字典，格式如下：
        {
            1: {"poster_url": "http...", "overview": "..." },
            260: {"poster_url": "http...", "overview": "..." }
        }
    """
    if not movie_ids:
        return {}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 确保列存在 (首次运行时生效)
    for col, col_type in [("poster_url", "TEXT"), ("overview", "TEXT"), ("title", "TEXT"), ("release_date", "TEXT"), ("popularity", "REAL")]:
        try:
            cursor.execute(f"ALTER TABLE movies ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    results = {}

    for movie_id in movie_ids:
        # 1. 先查本地数据库
        cursor.execute("SELECT tmdbId, poster_url, overview, title, release_date, popularity FROM movies WHERE movieId = ?", (movie_id,))
        row = cursor.fetchone()

        if row is None:
            # 如果数据库里根本没这部电影的数据
            results[movie_id] = {
                "poster_url": None, 
                "overview": None,
                "title": None,
                "release_date": None,
                "popularity": None,
                "link": None
            }
            continue

        tmdb_id, poster_url, overview, title, release_date, popularity = row

        # 2. 判断是否需要实时去 TMDb 拉取数据
        # 如果本地没有海报或简介，且存在 tmdb_id，则触发拉取
        if (not poster_url or not overview or not release_date) and tmdb_id:
            print(f"  -> 实时拉取 movieId {movie_id} (tmdbId: {tmdb_id}) 的补充数据...")
            new_poster, new_overview, new_title, new_release_date, new_popularity = fetch_movie_data(tmdb_id)

            # 3. 如果拉取成功，更新到本地数据库（持久化，下次直接读取）
            if new_poster or new_overview or new_title:
                cursor.execute("""
                               UPDATE movies
                               SET poster_url = ?,
                                   overview   = ?,
                                   title      = ?,
                                   release_date = ?,
                                   popularity = ?
                               WHERE movieId = ?
                               """, (new_poster, new_overview, new_title, new_release_date, new_popularity, movie_id))
                conn.commit()

            # 更新当前变量以便返回给前端
            poster_url = new_poster
            overview = new_overview
            title = new_title
            release_date = new_release_date
            popularity = new_popularity

            # 遵守 API 速率限制
            time.sleep(0.05)

        # 4. 组装结果
        results[movie_id] = {
            "poster_url": poster_url,
            "overview": overview,
            "title": title,
            "release_date": release_date,
            "popularity": popularity,
            "link": f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else None
        }

    conn.close()
    return results


# ==========================================
# 测试与集成示例
# ==========================================
if __name__ == "__main__":
    # 假设这是你刚才用 SVD 跑出来的 Top 5 推荐电影的 ID
    svd_recommended_ids = [3, 50, 110, 260, 318]

    print(f"推荐算法输出的候选电影 ID: {svd_recommended_ids}")
    print("-" * 50)

    # 直接调用封装好的函数
    movie_details = get_and_store_movie_details(svd_recommended_ids)

    # 打印最终准备返回给前端/用户界面的数据
    for m_id, info in movie_details.items():
        print(f"Movie ID    : {m_id}")
        print(f"TMDb Link   : {info['link']}")
        print(f"Title       : {info['title']}")
        print(f"Release Date: {info['release_date']}")
        print(f"Popularity  : {info['popularity']}")
        print(f"Poster      : {info['poster_url']}")
        # 截断简介，方便控制台查看
        short_overview = info['overview'][:50] + "..." if info['overview'] else "暂无简介"
        print(f"Overview    : {short_overview}")
        print("-" * 50)