import streamlit as st
import pandas as pd
import numpy as np

from typing import Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix


def _get_columns(merged_df: pd.DataFrame) -> Tuple[str, str]:
    """Resolve column names for genre and rating depending on merge outcome."""
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre' if 'Genre' in merged_df.columns else None)
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    if genre_col is None:
        raise ValueError("Genre column not found (expected 'Genre' or 'Genre_y').")
    if rating_col is None:
        raise ValueError("Rating column not found (expected 'IMDB_Rating' or 'Rating').")
    return genre_col, rating_col


def _prepare_text_series(s: pd.Series) -> pd.Series:
    """Basic cleanup and lowercasing for text fields."""
    return s.fillna("").astype(str).str.lower().str.strip()


def _prepare_genre_series(s: pd.Series) -> pd.Series:
    """Turn comma-separated genres into space-separated tokens (lowercased)."""
    s = s.fillna("").astype(str)
    tokens = s.apply(lambda x: " ".join([g.strip().replace(" ", "_") for g in x.split(",") if g.strip()]))
    return tokens.str.lower()


def _build_feature_matrix(merged_df: pd.DataFrame) -> Tuple[csr_matrix, TfidfVectorizer, TfidfVectorizer, MinMaxScaler, str, str]:
    """
    Build weighted sparse feature matrix from title, genre and rating.
    Weights: genre=8.0, rating=1.5, title=0.5
    """
    genre_col, rating_col = _get_columns(merged_df)

    title_text = _prepare_text_series(merged_df['Series_Title'])
    genre_text = _prepare_genre_series(merged_df[genre_col])

    title_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    genre_vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)

    title_tfidf = title_vectorizer.fit_transform(title_text)
    genre_tfidf = genre_vectorizer.fit_transform(genre_text)

    ratings = merged_df[rating_col].astype(float).fillna(merged_df[rating_col].astype(float).mean())
    scaler = MinMaxScaler()
    rating_scaled = scaler.fit_transform(ratings.values.reshape(-1, 1))
    rating_sparse = csr_matrix(rating_scaled)

    weighted_title = title_tfidf * 0.5
    weighted_genre = genre_tfidf * 8.0
    weighted_rating = rating_sparse * 1.5

    combined = hstack([weighted_title, weighted_genre, weighted_rating])
    return combined.tocsr(), title_vectorizer, genre_vectorizer, scaler, genre_col, rating_col


def _build_query_vector(
    movie_title: Optional[str],
    genre_input: Optional[str],
    avg_rating: float,
    title_vectorizer: TfidfVectorizer,
    genre_vectorizer: TfidfVectorizer,
    scaler: MinMaxScaler
) -> csr_matrix:
    """Create a query vector when no exact movie row is used (e.g., only genre selected)."""
    title_q = (movie_title or "").strip().lower()
    genre_q = (genre_input or "").strip().lower()

    # Vectorize title and genre
    title_vec = title_vectorizer.transform([title_q]) * 0.5
    # Normalize genre tokens similarly to training
    genre_tokens = " ".join([g.strip().replace(" ", "_") for g in genre_q.split(",") if g.strip()])
    genre_vec = genre_vectorizer.transform([genre_tokens]) * 8.0

    rating_vec = scaler.transform(np.array([[avg_rating]]))
    rating_vec = csr_matrix(rating_vec) * 1.5

    return hstack([title_vec, genre_vec, rating_vec]).tocsr()


def content_based_filtering_enhanced(
    merged_df: pd.DataFrame,
    movie_title: Optional[str],
    genre_input: Optional[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Content-based recommender using TF-IDF features with weights and cosine similarity.

    - Features: Series_Title (w=0.5), Genre (w=8.0), IMDB_Rating (w=1.5)
    - Similarity: cosine between weighted feature vectors
    - If both movie_title and genre_input provided, filter results by genre where possible
    """
    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    combined, title_v, genre_v, scaler, genre_col, rating_col = _build_feature_matrix(merged_df)

    # Resolve query by movie index if available
    idx = None
    if movie_title:
        # Match case-insensitively
        matches = merged_df.index[merged_df['Series_Title'].str.lower() == movie_title.lower()].tolist()
        if matches:
            idx = matches[0]

    if idx is not None:
        sims = cosine_similarity(combined[idx], combined).ravel()
    else:
        # Build a query vector from inputs
        avg_rating = float(merged_df[rating_col].astype(float).mean())
        q_vec = _build_query_vector(movie_title, genre_input, avg_rating, title_v, genre_v, scaler)
        sims = cosine_similarity(q_vec, combined).ravel()

    # Rank and prepare results
    merged_df = merged_df.copy()
    merged_df['ContentScore'] = sims

    # Exclude the query movie itself if present
    if idx is not None:
        merged_df.loc[idx, 'ContentScore'] = -np.inf

    # Optional: filter by genre if provided
    results = merged_df
    if genre_input:
        try:
            mask = merged_df[genre_col].astype(str).str.lower().str.contains(genre_input.strip().lower())
            # If filtering removes everything, keep original list
            if mask.any():
                results = merged_df[mask].copy()
        except Exception:
            # Fallback: keep all if any parsing issue
            results = merged_df

    results = results.sort_values('ContentScore', ascending=False).head(top_n)

    # Keep essential columns for UI
    keep_cols = ['Series_Title', rating_col, genre_col, 'ContentScore']
    keep_cols = [c for c in keep_cols if c in results.columns]

    return results[keep_cols].reset_index(drop=True)
 