from logging import root

import pandas as pd
import sqlite3
from pathlib import Path

# 1. Read links.csv
root_dir = Path(__file__).resolve().parent.parent
links_path = root_dir / 'datasets' / 'links.csv'
links_df = pd.read_csv(links_path)

# Clear empty values and ensure type
links_df = links_df.dropna(subset=['tmdbId'])
links_df['tmdbId'] = links_df['tmdbId'].astype(int)

# 2. Connect to database
conn = sqlite3.connect(root_dir / 'db' / 'movies_data.db')
cursor = conn.cursor()

# 3. Add tmdbId column to movies table
try:
    cursor.execute("ALTER TABLE movies ADD COLUMN tmdbId INTEGER")
except sqlite3.OperationalError:
    pass # Skip if column already exists

print("Creating index for movieId to accelerate queries...")
# [Key Optimization 1] Create index: this speeds up queries hundreds or thousands of times
cursor.execute("CREATE INDEX IF NOT EXISTS idx_movie_id ON movies(movieId)")

print("Batch updating tmdbId into database...")
# [Key Optimization 2] Prepare data format for batch update (List of Tuples)
# Pack the two columns into the format [(tmdbId_1, movieId_1), (tmdbId_2, movieId_2), ...]
data_to_update = list(zip(links_df['tmdbId'], links_df['movieId']))

# [Key Optimization 3] Use executemany instead of for loop
# Throw all data to SQLite's underlying C level processing at once, extremely high efficiency
cursor.executemany(
    "UPDATE movies SET tmdbId = ? WHERE movieId = ?",
    data_to_update
)

# Commit and close
conn.commit()
conn.close()
print("ID mapping update completed!")