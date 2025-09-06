import streamlit as st
import pandas as pd
import numpy as np

from typing import Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def find_genre_column(merged_df: pd.DataFrame) -> str:
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre' if 'Genre' in merged_df.columns else None)
    if genre_col is None:
        raise ValueError("Genre column not found (expected 'Genre' or 'Genre_y').")
    return genre_col


def find_rating_column(merged_df: pd.DataFrame) -> str:
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    if rating_col is None:
        raise ValueError("Rating column not found (expected 'IMDB_Rating' or 'Rating').")
    return rating_col


def _prepare_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.strip()


def _prepare_genre_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    tokens = s.apply(lambda x: " ".join([g.strip().replace(" ", "_") for g in x.split(",") if g.strip()]))
    return tokens.str.lower()


def create_content_features(merged_df: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer, TfidfVectorizer, MinMaxScaler, str, str]:
    """
    Build weighted dense feature matrix from title, genre and rating.
    Weights: genre=8.0, rating=1.5, title=0.5
    Returns (features, title_vectorizer, genre_vectorizer, scaler, genre_col, rating_col)
    """
    genre_col = find_genre_column(merged_df)
    rating_col = find_rating_column(merged_df)

    title_text = _prepare_text_series(merged_df['Series_Title'])
    genre_text = _prepare_genre_series(merged_df[genre_col])

    title_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    genre_vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)

    title_tfidf = title_vectorizer.fit_transform(title_text).toarray()
    genre_tfidf = genre_vectorizer.fit_transform(genre_text).toarray()

    ratings = merged_df[rating_col].astype(float).fillna(merged_df[rating_col].astype(float).mean())
    scaler = MinMaxScaler()
    rating_scaled = scaler.fit_transform(ratings.values.reshape(-1, 1))

    weighted_title = title_tfidf * 0.5
    weighted_genre = genre_tfidf * 8.0
    weighted_rating = rating_scaled * 1.5

    combined = np.hstack([weighted_title, weighted_genre, weighted_rating])
    return combined, title_vectorizer, genre_vectorizer, scaler, genre_col, rating_col


def _build_query_vector(
    movie_title: Optional[str],
    genre_input: Optional[str],
    avg_rating: float,
    title_vectorizer: TfidfVectorizer,
    genre_vectorizer: TfidfVectorizer,
    scaler: MinMaxScaler
) -> np.ndarray:
    title_q = (movie_title or "").strip().lower()
    genre_q = (genre_input or "").strip().lower()

    title_vec = title_vectorizer.transform([title_q]).toarray() * 0.5
    genre_tokens = " ".join([g.strip().replace(" ", "_") for g in genre_q.split(",") if g.strip()])
    genre_vec = genre_vectorizer.transform([genre_tokens]).toarray() * 8.0

    rating_vec = scaler.transform(np.array([[avg_rating]])) * 1.5

    return np.hstack([title_vec, genre_vec, rating_vec])


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

    combined, title_v, genre_v, scaler, genre_col, rating_col = create_content_features(merged_df)

    idx = None
    if movie_title:
        matches = merged_df.index[merged_df['Series_Title'].str.lower() == movie_title.lower()].tolist()
        if matches:
            idx = matches[0]

    if idx is not None:
        sims = cosine_similarity(combined[idx].reshape(1, -1), combined).ravel()
    else:
        avg_rating = float(merged_df[rating_col].astype(float).mean())
        q_vec = _build_query_vector(movie_title, genre_input, avg_rating, title_v, genre_v, scaler)
        sims = cosine_similarity(q_vec, combined).ravel()

    merged_df = merged_df.copy()
    merged_df['ContentScore'] = sims

    if idx is not None:
        merged_df.loc[idx, 'ContentScore'] = -np.inf

    results = merged_df
    if genre_input:
        try:
            mask = merged_df[genre_col].astype(str).str.lower().str.contains(genre_input.strip().lower())
            if mask.any():
                results = merged_df[mask].copy()
        except Exception:
            results = merged_df

    results = results.sort_values('ContentScore', ascending=False).head(top_n)

    keep_cols = ['Series_Title', rating_col, genre_col, 'ContentScore']
    keep_cols = [c for c in keep_cols if c in results.columns]

    return results[keep_cols].reset_index(drop=True)
 