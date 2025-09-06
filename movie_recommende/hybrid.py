import streamlit as st
import pandas as pd
import numpy as np

from typing import Optional

from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced, load_user_ratings


ALPHA = 0.4  # Content-based weight
BETA = 0.4   # Collaborative weight
GAMMA = 0.1  # Popularity weight
DELTA = 0.1  # Recency weight


def _resolve_columns(merged_df: pd.DataFrame) -> tuple[str, str, str, str]:
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    votes_col = 'No_of_Votes' if 'No_of_Votes' in merged_df.columns else None
    year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else ('Year' if 'Year' in merged_df.columns else None)
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return rating_col, votes_col, year_col, genre_col


def _compute_popularity(merged_df: pd.DataFrame, rating_col: str, votes_col: Optional[str]) -> pd.Series:
    # Popularity: IMDB_Rating × log(votes)
    ratings = merged_df[rating_col].astype(float).fillna(merged_df[rating_col].astype(float).mean())
    if votes_col is None or votes_col not in merged_df.columns:
        # If votes missing, fallback to just normalized rating
        popularity = ratings
    else:
        # Clean numeric votes (remove commas)
        votes_str = merged_df[votes_col].fillna('0').astype(str).str.replace(',', '', regex=False)
        votes = pd.to_numeric(votes_str, errors='coerce').fillna(0.0)
        popularity = ratings * np.log1p(votes)
    # Normalize 0..1
    pop_min, pop_max = popularity.min(), popularity.max()
    if pop_max > pop_min:
        popularity = (popularity - pop_min) / (pop_max - pop_min)
    else:
        popularity = popularity * 0.0
    return popularity


def _compute_recency(merged_df: pd.DataFrame, year_col: Optional[str]) -> pd.Series:
    # Recency: exponential decay based on year
    if year_col is None or year_col not in merged_df.columns:
        return pd.Series(np.zeros(len(merged_df)), index=merged_df.index)
    years = pd.to_numeric(merged_df[year_col], errors='coerce').fillna(merged_df[year_col].mode().iloc[0] if merged_df[year_col].mode().size > 0 else 2000)
    current_year = pd.Timestamp.today().year
    # Half-life like decay: newer => closer to 1
    decay_rate = 0.05  # tunable
    recency = np.exp(-decay_rate * (current_year - years))
    # Normalize 0..1
    rmin, rmax = recency.min(), recency.max()
    if rmax > rmin:
        recency = (recency - rmin) / (rmax - rmin)
    else:
        recency = recency * 0.0
    return pd.Series(recency, index=merged_df.index)


def smart_hybrid_recommendation(
    merged_df: pd.DataFrame,
    movie_title: Optional[str],
    genre_input: Optional[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Hybrid recommendations blending Content, Collaborative, Popularity, and Recency.
    FinalScore = 0.4*Content + 0.4*Collaborative + 0.1*Popularity + 0.1*Recency
    """
    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    rating_col, votes_col, year_col, genre_col = _resolve_columns(merged_df)

    # Content-based component
    cb_df = content_based_filtering_enhanced(merged_df, movie_title, genre_input, top_n=max(top_n * 3, 25))
    if cb_df is None or cb_df.empty:
        # If content fails, degrade to popularity/recency
        cb_df = merged_df[['Series_Title']].copy()
        cb_df['ContentScore'] = 0.0
        if rating_col in merged_df.columns:
            cb_df[rating_col] = merged_df[rating_col]
        if genre_col in merged_df.columns:
            cb_df[genre_col] = merged_df[genre_col]

    # Collaborative component
    cf_df = collaborative_filtering_enhanced(merged_df, movie_title, top_n=max(top_n * 3, 25)) if movie_title else pd.DataFrame()

    # Popularity and Recency
    popularity = _compute_popularity(merged_df, rating_col, votes_col)
    recency = _compute_recency(merged_df, year_col)

    base = merged_df[['Series_Title']].copy()
    base['Popularity'] = popularity.values
    base['Recency'] = recency.values

    # Merge components
    combined = base.merge(cb_df[['Series_Title', 'ContentScore']], on='Series_Title', how='left')
    if not cf_df.empty:
        combined = combined.merge(cf_df[['Series_Title', 'CFScore']], on='Series_Title', how='left')
    else:
        combined['CFScore'] = 0.0

    # Fill NaNs
    combined['ContentScore'] = combined['ContentScore'].fillna(0.0)
    combined['CFScore'] = combined['CFScore'].fillna(0.0)

    # Normalize content and collaborative to 0..1 for blending fairness
    for col in ['ContentScore', 'CFScore']:
        cmin, cmax = combined[col].min(), combined[col].max()
        if cmax > cmin:
            combined[col] = (combined[col] - cmin) / (cmax - cmin)
        else:
            combined[col] = 0.0

    # Final score
    combined['FinalScore'] = (
        ALPHA * combined['ContentScore'] +
        BETA * combined['CFScore'] +
        GAMMA * combined['Popularity'] +
        DELTA * combined['Recency']
    )

    # Optional genre filtering
    if genre_input:
        try:
            mask = merged_df.set_index('Series_Title')[genre_col].astype(str).str.lower().str.contains(genre_input.strip().lower())
            mask = mask.reindex(combined['Series_Title']).fillna(False)
            if mask.any():
                combined = combined[mask.values]
        except Exception:
            pass

    # Exclude the exact query movie from the top list
    if movie_title:
        combined = combined[combined['Series_Title'].str.lower() != movie_title.lower()]

    # Attach metadata for UI
    meta_cols = {}
    if rating_col in merged_df.columns:
        meta_cols['IMDB_Rating'] = merged_df.set_index('Series_Title')[rating_col]
    if genre_col in merged_df.columns:
        meta_cols['Genre'] = merged_df.set_index('Series_Title')[genre_col]

    for new_col, series in meta_cols.items():
        combined[new_col] = combined['Series_Title'].map(series)

    combined = combined.sort_values('FinalScore', ascending=False).head(top_n)

    # Reorder for UI
    keep_cols = ['Series_Title', 'IMDB_Rating', 'Genre', 'FinalScore']
    keep_cols = [c for c in keep_cols if c in combined.columns]

    return combined[keep_cols].reset_index(drop=True)
