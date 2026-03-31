import pandas as pd
import sqlite3
import re

# Read data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

# 1. Process movies
# Extract year from title
def extract_year(title):
    match = re.search(r'\((\d{4})\)\s*$', str(title))
    if match:
        return match.group(1), re.sub(r'\(\d{4}\)\s*$', '', str(title)).strip()
    return None, str(title).strip()

movies['year'] = movies['title'].apply(lambda x: extract_year(x)[0])
movies['title'] = movies['title'].apply(lambda x: extract_year(x)[1])

# Aggregate ratings per movie
movie_ratings = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count'),
    first_rating_timestamp=('timestamp', 'min'),
    last_rating_timestamp=('timestamp', 'max')
).reset_index()

# Aggregate tags per movie
movie_tags = tags.groupby('movieId').agg(
    tags=('tag', lambda x: '|'.join(set(x.dropna().astype(str)))),
    first_tag_timestamp=('timestamp', 'min'),
    last_tag_timestamp=('timestamp', 'max')
).reset_index()

# Merge movie data
movie_data = movies.merge(movie_ratings, on='movieId', how='left')
movie_data = movie_data.merge(movie_tags, on='movieId', how='left')

# Fill NaN
movie_data['rating_count'] = movie_data['rating_count'].fillna(0).astype(int)
movie_data['tags'] = movie_data['tags'].fillna('')

# Save movie aggregated data
movie_data.to_csv('movies_aggregated.csv', index=False)

# 2. Aggregate user data (unique userId)
# User ratings stats
user_ratings = ratings.groupby('userId').agg(
    rating_count=('rating', 'count'),
    avg_rating=('rating', 'mean'),
    first_rating_timestamp=('timestamp', 'min'),
    last_rating_timestamp=('timestamp', 'max')
).reset_index()

# User tags stats
user_tags = tags.groupby('userId').agg(
    tag_count=('tag', 'count'),
    tags_given=('tag', lambda x: '|'.join(set(x.dropna().astype(str)))),
    first_tag_timestamp=('timestamp', 'min'),
    last_tag_timestamp=('timestamp', 'max')
).reset_index()

# We need all unique users from both ratings and tags
unique_users = pd.DataFrame({'userId': pd.concat([ratings['userId'], tags['userId']]).unique()})
user_data = unique_users.merge(user_ratings, on='userId', how='left')
user_data = user_data.merge(user_tags, on='userId', how='left')

# Fill NaN
user_data['rating_count'] = user_data['rating_count'].fillna(0).astype(int)
user_data['tag_count'] = user_data['tag_count'].fillna(0).astype(int)
user_data['tags_given'] = user_data['tags_given'].fillna('')

# Save user aggregated data
user_data.to_csv('users_aggregated.csv', index=False)

# 3. Create SQLite Database and insert data
conn = sqlite3.connect('movies_data.db')

# Write to database
movie_data.to_sql('movies', conn, if_exists='replace', index=False)
user_data.to_sql('users', conn, if_exists='replace', index=False)

conn.close()

print("Data processing and database creation completed successfully.")
