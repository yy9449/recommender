import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors


def _detect_columns(merged_df: pd.DataFrame):
    """Detect common column names used across datasets."""
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
    return rating_col, genre_col, year_col


@st.cache_data
def load_user_ratings() -> pd.DataFrame | None:
    """Load user ratings from Streamlit session or local CSV if available."""
    try:
        # Prefer dataset loaded by main app
        if 'user_ratings_df' in st.session_state and st.session_state['user_ratings_df'] is not None:
            return st.session_state['user_ratings_df']

        # Fallback: try local CSV
        try:
            return pd.read_csv('user_movie_rating.csv')
        except Exception:
            return None
    except Exception:
        return None


def _build_user_item_matrix(user_ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Create a user-item ratings matrix (users x movies)."""
    user_item = user_ratings_df.pivot_table(
        index='User_ID',
        columns='Movie_ID',
        values='Rating',
        fill_value=0
    )
    return user_item


def _get_movie_id_for_title(merged_df: pd.DataFrame, title: str) -> int | None:
    """Map a movie title to Movie_ID using the merged dataset."""
    try:
        match = merged_df[merged_df['Series_Title'].str.lower() == str(title).lower()]
        if match.empty:
            return None
        if 'Movie_ID' in match.columns:
            return int(match.iloc[0]['Movie_ID'])
        # If no Movie_ID present, synthesize a positional index
        return int(match.index[0])
    except Exception:
        return None


@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8, k: int = 20) -> pd.DataFrame | None:
    """
    Item-based KNN collaborative filtering.

    - Builds a user-item rating matrix from user ratings
    - Uses cosine distance KNN to find similar movies to the target
    - Returns a dataframe with recommended titles and metadata
    """
    if not target_movie:
        return None

    user_ratings_df = load_user_ratings()
    if user_ratings_df is None or user_ratings_df.empty:
        return None

    # Ensure Movie_ID exists in merged dataset
    if 'Movie_ID' not in merged_df.columns:
        merged_df = merged_df.copy()
        merged_df['Movie_ID'] = range(len(merged_df))

    user_item = _build_user_item_matrix(user_ratings_df)
    if user_item.empty or user_item.shape[1] < 2:
        return None

    target_movie_id = _get_movie_id_for_title(merged_df, target_movie)
    if target_movie_id is None or target_movie_id not in user_item.columns:
        return None

    # Fit KNN on item (movie) vectors: shape (n_movies, n_users)
    item_matrix = user_item.T.values
    try:
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(item_matrix)
    except Exception:
        return None

    # Locate index of target in the item matrix
    movie_ids = list(user_item.columns)
    try:
        target_idx = movie_ids.index(target_movie_id)
    except ValueError:
        return None

    distances, indices = knn.kneighbors([item_matrix[target_idx]], n_neighbors=min(k + 1, len(movie_ids)))
    distances = distances.flatten()
    indices = indices.flatten()

    # Exclude the target itself (distance=0 at position 0)
    neighbor_pairs = [(idx, distances[i]) for i, idx in enumerate(indices) if movie_ids[idx] != target_movie_id]
    neighbor_pairs = neighbor_pairs[:top_n * 2]
    neighbor_indices = [idx for idx, _ in neighbor_pairs]
    neighbor_ids = [movie_ids[idx] for idx in neighbor_indices]

    # Similarity scores (cosine similarity = 1 - distance)
    neighbor_sims = {movie_ids[idx]: float(1.0 - dist) for idx, dist in neighbor_pairs}

    # Build results from neighbors
    rating_col, genre_col, _ = _detect_columns(merged_df)
    neighbor_movies = merged_df[merged_df['Movie_ID'].isin(neighbor_ids)].copy()
    if neighbor_movies.empty:
        return None

    # Preserve neighbor order by KNN
    order_map = {mid: i for i, mid in enumerate(neighbor_ids)}
    neighbor_movies['rank_order'] = neighbor_movies['Movie_ID'].map(order_map)
    neighbor_movies['CF_Score'] = neighbor_movies['Movie_ID'].map(neighbor_sims)
    # Normalize CF_Score to 0..1 across neighbors
    try:
        max_cf = float(neighbor_movies['CF_Score'].max())
        if max_cf > 0:
            neighbor_movies['CF_Score'] = neighbor_movies['CF_Score'] / max_cf
    except Exception:
        pass
    neighbor_movies = neighbor_movies.sort_values('rank_order').drop(columns=['rank_order'])

    # Return the top_n
    cols = [c for c in ['Series_Title', genre_col, rating_col, 'CF_Score'] if c in neighbor_movies.columns]
    return neighbor_movies[cols].head(top_n)


def diagnose_data_linking(merged_df: pd.DataFrame | None = None, user_ratings_df: pd.DataFrame | None = None) -> dict:
    """Provide basic diagnostics for dataset linking between movies and user ratings."""
    info = {}
    try:
        if merged_df is not None:
            info['movies_total'] = int(len(merged_df))
            info['has_movie_id'] = bool('Movie_ID' in merged_df.columns)
        if user_ratings_df is None and 'user_ratings_df' in st.session_state:
            user_ratings_df = st.session_state['user_ratings_df']
        if user_ratings_df is not None:
            info['ratings_total'] = int(len(user_ratings_df))
            info['unique_users'] = int(user_ratings_df['User_ID'].nunique()) if 'User_ID' in user_ratings_df.columns else 0
            info['unique_movies_in_ratings'] = int(user_ratings_df['Movie_ID'].nunique()) if 'Movie_ID' in user_ratings_df.columns else 0
        if merged_df is not None and user_ratings_df is not None and 'Movie_ID' in merged_df.columns and 'Movie_ID' in user_ratings_df.columns:
            linked = user_ratings_df['Movie_ID'].isin(set(merged_df['Movie_ID']))
            info['linked_ratings'] = int(linked.sum())
        return info
    except Exception:
        return info
