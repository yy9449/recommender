# content_based.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """
    Creates the final, best-performing 'soup' with a balanced weighting of features.
    """
    df_copy = df.copy()

    # --- Final Optimized Weights ---
    w_overview = 4.0
    w_genre = 3.0
    w_title = 2.0
    w_director = 1.0

    # Clean and prepare the selected features
    overview = df_copy['Overview'].fillna('').astype(str)
    
    # Handle both Genre and Genre_y columns
    genre_col = 'Genre_y' if 'Genre_y' in df_copy.columns else 'Genre'
    genre = df_copy[genre_col].fillna('').astype(str)
    
    title = df_copy['Series_Title'].fillna('').astype(str)
    director = df_copy['Director'].fillna('').astype(str)

    # Combine the features with their weights
    soup = (
        (overview + ' ') * int(w_overview) +
        (genre + ' ') * int(w_genre) +
        (title + ' ') * int(w_title) +
        (director + ' ') * int(w_director)
    )
    return soup

@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str = None, genre_filter: str = None, top_n: int = 10):
    """
    Generates movie recommendations using the final, optimized TF-IDF content model.
    Can work with movie similarity, genre filtering, or both.
    """
    
    # If neither movie nor genre is provided, return empty
    if not target_movie and not genre_filter:
        return pd.DataFrame()
    
    # Handle genre filtering
    if genre_filter and not target_movie:
        # Pure genre-based recommendations
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        genre_filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
        
        if genre_filtered.empty:
            return pd.DataFrame()
        
        # Sort by rating and return top results
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in genre_filtered.columns else 'Rating'
        genre_filtered = genre_filtered.sort_values(by=rating_col, ascending=False)
        
        return genre_filtered.head(top_n)
    
    # Movie-based recommendations (with optional genre filtering)
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    soup = _create_optimized_soup(merged_df)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n * 2 + 1]  # Get more to allow for genre filtering
    movie_indices = [i[0] for i in sim_scores]
    
    results = merged_df.iloc[movie_indices]
    
    # Apply genre filter if provided
    if genre_filter:
        genre_col = 'Genre_y' if 'Genre_y' in results.columns else 'Genre'
        results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
    
    return results.head(top_n)
