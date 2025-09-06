# content_based.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """
    Creates the final, best-performing 'soup' with a balanced weighting of features.
    Handles missing columns gracefully.
    """
    df_copy = df.copy()

    # --- Final Optimized Weights ---
    w_overview = 4.0
    w_genre = 3.0
    w_title = 2.0
    w_director = 1.0

    # Clean and prepare the selected features with fallbacks
    # Overview - check multiple possible column names INCLUDING _x and _y variants
    overview_cols = ['Overview', 'Overview_y', 'Overview_x', 'Plot', 'Description', 'Summary']
    overview = None
    for col in overview_cols:
        if col in df_copy.columns:
            overview = df_copy[col].fillna('').astype(str)
            break
    if overview is None:
        overview = pd.Series([''] * len(df_copy), index=df_copy.index)
    
    # Handle both Genre and Genre_y columns
    genre_col = 'Genre_y' if 'Genre_y' in df_copy.columns else 'Genre_x' if 'Genre_x' in df_copy.columns else 'Genre'
    genre = df_copy[genre_col].fillna('').astype(str) if genre_col in df_copy.columns else pd.Series([''] * len(df_copy))

    # Handle Series_Title
    title = df_copy['Series_Title'].fillna('').astype(str)

    # Handle Director
    director = df_copy['Director'].fillna('').astype(str) if 'Director' in df_copy.columns else pd.Series([''] * len(df_copy))
    
    # Apply weights
    soup = (
        (w_overview * (overview + ' ')) +
        (w_genre * (genre + ' ')) +
        (w_title * (title + ' ')) +
        (w_director * (director + ' '))
    ).str.lower().str.replace(',', ' ')

    return soup

@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str = None, genre_filter: str = None, top_n: int = 10):
    """
    Enhanced content-based filtering that can handle either a movie, a genre, or both.
    Uses an optimized soup creation method.
    """
    # Handle genre-only filtering
    if not target_movie and genre_filter:
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        
        if genre_col not in merged_df.columns or rating_col not in merged_df.columns:
            return pd.DataFrame()
            
        genre_filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
        genre_filtered = genre_filtered.sort_values(by=rating_col, ascending=False)
        
        return genre_filtered.head(top_n)
    
    # Movie-based recommendations (with optional genre filtering)
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    try:
        soup = _create_optimized_soup(merged_df)
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(soup)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n * 2 + 1]  # Get more to allow for genre filtering
        movie_indices = [i[0] for i in sim_scores]
        
        results = merged_df.iloc[movie_indices]
        
        # Apply genre filter if provided
        if genre_filter:
            genre_col = 'Genre_y' if 'Genre_y' in results.columns else 'Genre_x' if 'Genre_x' in results.columns else 'Genre'
            if genre_col in results.columns:
                results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
        
        return results.head(top_n)

    except Exception as e:
        st.error(f"An error occurred during content-based filtering: {e}")
        return pd.DataFrame()
