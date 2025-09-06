import pandas as pd
import streamlit as st
import re
from typing import List, Optional, Tuple


def _get_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Pick canonical column names present in the merged dataframe."""
    genre_col = 'Genre_y' if 'Genre_y' in df.columns else 'Genre'
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'
    director_col = 'Director_y' if 'Director_y' in df.columns else 'Director'
    return genre_col, rating_col, director_col


def _normalize_text(value: Optional[str]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    return str(value).strip().lower()


def _normalize_genres(value: Optional[str]) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    parts = [p.strip() for p in text.split(',') if p and p.strip()]
    tokens = []
    for p in parts:
        t = p.replace('-', ' ').replace('_', ' ').strip().lower()
        t = re.sub(r"\s+", " ", t)
        if t:
            tokens.append(t)
    return tokens


def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return default
    s = str(value)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
    return default


def _rating_bounds(df: pd.DataFrame, rating_col: str) -> Tuple[Optional[float], Optional[float]]:
    vals = df[rating_col].apply(lambda v: _to_float(v, None))
    vals = vals.dropna()
    if vals.empty:
        return None, None
    return float(vals.min()), float(vals.max())


def _jaccard(a: List[str], b: List[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return (inter / union) if union > 0 else 0.0


def _score_movie_to_movie(
    target_genres: List[str],
    target_director: str,
    target_rating: Optional[float],
    cand_genres: List[str],
    cand_director: str,
    cand_rating: Optional[float],
    rating_min: Optional[float],
    rating_max: Optional[float],
) -> float:
    # Genre similarity via Jaccard
    genre_sim = _jaccard(target_genres, cand_genres)

    # Director match (exact, normalized)
    director_sim = 1.0 if target_director and (target_director == cand_director) else 0.0

    # Rating similarity (closer is better), normalized to [0,1]
    rating_sim = 0.0
    if (target_rating is not None) and (cand_rating is not None) and (rating_min is not None) and (rating_max is not None) and (rating_max > rating_min):
        diff = abs(target_rating - cand_rating)
        rating_sim = 1.0 - (diff / (rating_max - rating_min))
        rating_sim = max(0.0, min(1.0, rating_sim))

    # Weighted sum (focus on genres, then director, then rating)
    return 0.7 * genre_sim + 0.2 * director_sim + 0.1 * rating_sim


def _score_genre_query(
    query_genres: List[str],
    cand_genres: List[str],
    cand_rating: Optional[float],
    rating_min: Optional[float],
    rating_max: Optional[float],
) -> float:
    # Genre similarity is primary; rating lifts higher-rated titles slightly
    genre_sim = _jaccard(query_genres, cand_genres)

    rating_bonus = 0.0
    if (cand_rating is not None) and (rating_min is not None) and (rating_max is not None) and (rating_max > rating_min):
        rating_norm = (cand_rating - rating_min) / (rating_max - rating_min)
        rating_bonus = max(0.0, min(1.0, rating_norm))

    return 0.9 * genre_sim + 0.1 * rating_bonus


def content_based_filtering_enhanced(
    merged_df: pd.DataFrame,
    target_movie: Optional[str] = None,
    genre: Optional[str] = None,
    top_n: int = 8,
) -> Optional[pd.DataFrame]:
    """Basic Content-Based filtering using only Genre, Rating, Director, and Series_Title.

    - If target_movie is provided: find similar movies by genre overlap, same director, and close rating.
    - If genre is provided (and no target_movie): find movies matching the genre and prefer higher ratings.

    Returns a DataFrame with columns: Series_Title, <Genre Col>, <Rating Col>.
    """
    if merged_df is None or merged_df.empty:
        return None

    genre_col, rating_col, director_col = _get_columns(merged_df)

    # Precompute useful values
    rating_min, rating_max = _rating_bounds(merged_df, rating_col)

    if target_movie:
        # Exact match only; main UI passes a selected title
        mask = merged_df['Series_Title'] == target_movie
        if not mask.any():
            return None
        target_row = merged_df[mask].iloc[0]

        target_genres = _normalize_genres(target_row.get(genre_col, ''))
        target_director = _normalize_text(target_row.get(director_col, ''))
        target_rating = _to_float(target_row.get(rating_col, None), None)

        scores = []
        for idx, row in merged_df.iterrows():
            if row['Series_Title'] == target_movie:
                continue
            cand_genres = _normalize_genres(row.get(genre_col, ''))
            # require at least one shared genre to keep it simple and relevant
            if _jaccard(target_genres, cand_genres) <= 0.0:
                continue
            cand_director = _normalize_text(row.get(director_col, ''))
            cand_rating = _to_float(row.get(rating_col, None), None)

            score = _score_movie_to_movie(
                target_genres,
                target_director,
                target_rating,
                cand_genres,
                cand_director,
                cand_rating,
                rating_min,
                rating_max,
            )
            scores.append((score, idx))

        if not scores:
            return merged_df.head(0)[['Series_Title', genre_col, rating_col]]  # empty with correct columns

        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scores[:top_n]]
        result_df = merged_df.iloc[top_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
    
    if genre:
        query_genres = _normalize_genres(genre)
        scores = []
        for idx, row in merged_df.iterrows():
            cand_genres = _normalize_genres(row.get(genre_col, ''))
            if _jaccard(query_genres, cand_genres) <= 0.0:
                continue
            cand_rating = _to_float(row.get(rating_col, None), None)
            score = _score_genre_query(
                query_genres,
                cand_genres,
                cand_rating,
                rating_min,
                rating_max,
            )
            scores.append((score, idx))

        if not scores:
            return merged_df.head(0)[['Series_Title', genre_col, rating_col]]  # empty with correct columns

        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scores[:top_n]]
        result_df = merged_df.iloc[top_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
        
    return None