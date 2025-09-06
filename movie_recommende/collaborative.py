import streamlit as st
import pandas as pd
import numpy as np

from typing import Optional

from sklearn.neighbors import NearestNeighbors


def load_user_ratings() -> Optional[pd.DataFrame]:
    """
    Load user ratings from session state if available, else attempt to read local file.
    Returns None if not available.
    """
    try:
        df = st.session_state.get('user_ratings_df')
    except Exception:
        df = None
    if df is not None:
        return df
    try:
        return pd.read_csv('user_movie_rating.csv')
    except Exception:
        return None


def diagnose_data_linking(merged_df: pd.DataFrame, user_ratings_df: Optional[pd.DataFrame]) -> dict:
    stats = {
        'has_user_ratings': user_ratings_df is not None,
        'unique_movies_in_user_ratings': int(user_ratings_df['Movie_ID'].nunique()) if user_ratings_df is not None else 0,
        'unique_movies_in_merged': int(merged_df['Series_Title'].nunique()) if merged_df is not None else 0,
    }
    return stats


def _build_item_knn_model(user_ratings_df: pd.DataFrame, n_neighbors: int = 20):
    """
    Build item-based KNN on a dense item-user matrix (NumPy). Returns (model, matrix, movieId_to_index)
    """
    ratings = user_ratings_df[['User_ID', 'Movie_ID', 'Rating']].copy()
    ratings['User_ID'] = ratings['User_ID'].astype(int)
    ratings['Movie_ID'] = ratings['Movie_ID'].astype(int)
    ratings['Rating'] = ratings['Rating'].astype(float)

    unique_movie_ids = np.sort(ratings['Movie_ID'].unique())
    movie_id_to_index = {mid: idx for idx, mid in enumerate(unique_movie_ids)}

    unique_users = np.sort(ratings['User_ID'].unique())
    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}

    # Initialize dense matrix (items x users)
    item_user_matrix = np.zeros((len(unique_movie_ids), len(unique_users)), dtype=float)

    for _, r in ratings.iterrows():
        i = movie_id_to_index[int(r['Movie_ID'])]
        j = user_id_to_index[int(r['User_ID'])]
        item_user_matrix[i, j] = float(r['Rating'])

    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=min(n_neighbors, len(unique_movie_ids)))
    knn.fit(item_user_matrix)

    return knn, item_user_matrix, movie_id_to_index


def _title_to_movie_id(merged_df: pd.DataFrame, title: str) -> Optional[int]:
    try:
        row = merged_df.loc[merged_df['Series_Title'].str.lower() == title.lower()].iloc[0]
        return int(row['Movie_ID']) if 'Movie_ID' in row.index else None
    except Exception:
        return None


def _movie_id_to_title(merged_df: pd.DataFrame, movie_id: int) -> Optional[str]:
    try:
        row = merged_df.loc[merged_df['Movie_ID'] == movie_id].iloc[0]
        return str(row['Series_Title'])
    except Exception:
        return None


def collaborative_filtering_enhanced(
    merged_df: pd.DataFrame,
    movie_title: str,
    top_n: int = 10
) -> pd.DataFrame:
    if merged_df is None or merged_df.empty or not movie_title:
        return pd.DataFrame()

    user_ratings_df = load_user_ratings()
    if user_ratings_df is None or user_ratings_df.empty:
        return pd.DataFrame()

    if 'Movie_ID' not in merged_df.columns:
        merged_df = merged_df.copy()
        merged_df['Movie_ID'] = range(len(merged_df))

    knn, item_user_matrix, movie_id_to_index = _build_item_knn_model(user_ratings_df, n_neighbors=max(top_n + 5, 20))

    target_movie_id = _title_to_movie_id(merged_df, movie_title)
    if target_movie_id is None or target_movie_id not in movie_id_to_index:
        return pd.DataFrame()

    target_index = movie_id_to_index[target_movie_id]

    distances, indices = knn.kneighbors(item_user_matrix[target_index].reshape(1, -1), n_neighbors=min(top_n + 1, item_user_matrix.shape[0]))
    distances = distances.flatten()
    indices = indices.flatten()

    neighbor_indices = []
    neighbor_scores = []
    for d, idx in zip(distances, indices):
        if idx == target_index:
            continue
        sim = 1.0 - float(d)
        neighbor_indices.append(idx)
        neighbor_scores.append(sim)
        if len(neighbor_indices) >= top_n:
            break

    index_to_movie_id = {idx: mid for mid, idx in movie_id_to_index.items()}
    neighbor_movie_ids = [index_to_movie_id[idx] for idx in neighbor_indices]

    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'

    results = []
    for mid, score in zip(neighbor_movie_ids, neighbor_scores):
        title = _movie_id_to_title(merged_df, mid)
        if title is None:
            continue
        row = merged_df.loc[merged_df['Series_Title'] == title].iloc[0]
        rec = { 'Series_Title': title, 'CFScore': score }
        for c in [genre_col, rating_col]:
            if c in row.index:
                rec[c] = row[c]
        results.append(rec)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    ordered_cols = ['Series_Title', 'CFScore'] + [c for c in [genre_col, rating_col] if c in df.columns]
    df = df[ordered_cols]

    return df.reset_index(drop=True)


collaborative_knn = collaborative_filtering_enhanced
