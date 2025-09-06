# content_based.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """
    Creates a weighted 'soup' of features for content-based filtering.
    Handles various possible column names after merges.
    """
    df_copy = df.copy()

    w_overview = 4.0
    w_genre = 3.0
    w_title = 2.0
    w_director = 1.0

    # FIX 1: Robust handling for 'Overview' column
    overview_text = None
    for col in ['Overview', 'Overview_y', 'Overview_x', 'Plot']:
        if col in df_copy.columns:
            overview_text = df_copy[col].fillna('').astype(str)
            break
    if overview_text is None:
        overview_text = pd.Series([''] * len(df_copy), index=df_copy.index)
    
    # FIX 2: Robust handling for 'Genre' column
    genre_col_name = 'Genre_y' if 'Genre_y' in df_copy.columns else 'Genre_x' if 'Genre_x' in df_copy.columns else 'Genre'
    genre_text = df_copy[genre_col_name].fillna('').astype(str) if genre_col_name in df_copy.columns else pd.Series([''] * len(df_copy), index=df_copy.index)

    title = df_copy['Series_Title'].fillna('').astype(str)
    director = df_copy['Director'].fillna('').astype(str)

    # Combine features with weights
    soup = (
        (overview_text + ' ') * int(w_overview) +
        (genre_text + ' ') * int(w_genre) +
        (title + ' ') * int(w_title) +
        director
    )
    return soup

@st.cache_data
def content_based_filtering_enhanced(
    merged_df: pd.DataFrame, 
    target_movie: str = None, 
    genre_filter: str = None, # <-- FIX 3: Added genre_filter parameter
    top_n: int = 10
):
    """
    Generates movie recommendations using TF-IDF, with optional genre filtering.
    """
    if not target_movie and not genre_filter:
        return pd.DataFrame()

    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
    
    # Handle genre-only case
    if genre_filter and not target_movie:
        if genre_col in merged_df.columns:
            genre_filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
            # Sort by rating as a simple popularity measure
            rating_col = 'IMDB_Rating' if 'IMDB_Rating' in genre_filtered.columns else 'Rating'
            if rating_col in genre_filtered.columns:
                genre_filtered = genre_filtered.sort_values(by=rating_col, ascending=False)
            return genre_filtered.head(top_n)
        return pd.DataFrame() # Return empty if no genre column found
    
    # Movie-based recommendations
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    try:
        soup = _create_optimized_soup(merged_df)
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(soup)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        # Get more results to have a buffer for genre filtering
        sim_scores = sim_scores[1:(top_n * 2) + 1] 
        movie_indices = [i[0] for i in sim_scores]
        
        results = merged_df.iloc[movie_indices]
        
        # FIX 4: Apply genre filter to the results if provided
        if genre_filter:
            if genre_col in results.columns:
                results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
        
        return results.head(top_n)

    except Exception as e:
        st.error(f"An error occurred in content-based filtering: {e}")
        return pd.DataFrame()
