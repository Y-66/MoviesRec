import sqlite3
import requests
import time
from typing import List, Dict

# ================= Configuration Section =================
DB_PATH = 'db/movies_data.db'
API_KEY = '2fc4340de0512e93d9982b33593fe440'
BASE_URL = 'https://api.themoviedb.org/3/movie/'
IMG_BASE_URL = 'https://image.tmdb.org/t/p/w500'


# ============================================

def fetch_movie_data(tmdb_id):
    """Fetch movie data via TMDb API (low-level call)"""
    url = f"{BASE_URL}{tmdb_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=5)  # Shorten timeout to avoid front-end waiting too long
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
        print(f"  [X] API Request Error tmdbId {tmdb_id}: {e}")
    return None, None, None, None, None


def get_and_store_movie_details(movie_ids: List[int]) -> Dict[int, Dict[str, str]]:
    """
    [Core Interface] Fetch recommended movies posters and overviews on demand.
    Workflow: Check database -> If no poster -> Call API -> Save to DB -> Return result

    Args:
        movie_ids (List[int]): List of recommended movie IDs (e.g., [1, 260, 318])

    Returns:
        Dict: Dict containing posters and overviews, format as follows:
        {
            1: {"poster_url": "http...", "overview": "..." },
            260: {"poster_url": "http...", "overview": "..." }
        }
    """
    if not movie_ids:
        return {}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure columns exist (effective on first run)
    for col, col_type in [("poster_url", "TEXT"), ("overview", "TEXT"), ("title", "TEXT"), ("release_date", "TEXT"), ("popularity", "REAL")]:
        try:
            cursor.execute(f"ALTER TABLE movies ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    results = {}

    for movie_id in movie_ids:
        # 1. Check local database first
        cursor.execute("SELECT tmdbId, poster_url, overview, title, release_date, popularity FROM movies WHERE movieId = ?", (movie_id,))
        row = cursor.fetchone()

        if row is None:
            # If there is no data for this movie in the database at all
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

        # 2. Determine whether real-time fetching from TMDb is needed
        # Trigger fetching if local poster or overview is missing and tmdb_id exists
        if (not poster_url or not overview or not release_date) and tmdb_id:
            print(f"  -> Fetching data in real-time for movieId {movie_id} (tmdbId: {tmdb_id}) supplementary data...")
            new_poster, new_overview, new_title, new_release_date, new_popularity = fetch_movie_data(tmdb_id)

            # 3. If successful, update local database (persistent, read directly next time)
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

            # Update current variables to return to front-end
            poster_url = new_poster
            overview = new_overview
            title = new_title
            release_date = new_release_date
            popularity = new_popularity

            # Respect API rate limits
            time.sleep(0.05)

        # 4. Assemble results
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
# Test and Integration Example
# ==========================================
if __name__ == "__main__":
    # Assuming these are the Top 5 recommended movie IDs you just got from SVD
    svd_recommended_ids = [3, 50, 110, 260, 318]

    print(f"Candidate movies from recommendation algorithm ID: {svd_recommended_ids}")
    print("-" * 50)

    # Directly call the encapsulated function
    movie_details = get_and_store_movie_details(svd_recommended_ids)

    # Print final data ready to be returned to front-end/UI
    for m_id, info in movie_details.items():
        print(f"Movie ID    : {m_id}")
        print(f"TMDb Link   : {info['link']}")
        print(f"Title       : {info['title']}")
        print(f"Release Date: {info['release_date']}")
        print(f"Popularity  : {info['popularity']}")
        print(f"Poster      : {info['poster_url']}")
        # Truncate overview for easy terminal viewing
        short_overview = info['overview'][:50] + "..." if info['overview'] else "No overview available"
        print(f"Overview    : {short_overview}")
        print("-" * 50)