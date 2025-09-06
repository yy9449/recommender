# content_based.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Import column utilities (place column_utils.py in the same directory)
try:
    from column_utils import (
        get_genre_column, get_overview_column, get_rating_column, 
        get_director_column, safe_get_column_data, apply_genre_filter
    )
except ImportError:
    # Fallback functions if column_utils.py is not available
    def get_genre_column(df):
        for col in ['Genre', 'Genre_y', 'Genre_x', 'Genres']:
            if col in df.columns:
                return col
        return None
    
    def get_overview_column(df):
        for col in ['Overview', 'Overview_y', 'Overview_x', 'Plot', 'Description', 'Summary']:
            if col in df.columns:
                return col
        return None
    
    def get_rating_column(df):
        for col in ['IMDB_Rating', 'Rating', 'IMDB_Rating_y', 'IMDB_Rating_x']:
            if col in df.columns:
                return col
        return None
    
    def get_director_column(df):
        for col in ['Director', 'Director_y', 'Director_x']:
            if col in df.columns:
                return col
        return None
    
    def safe_get_column_data(df, column_name, default_value=''):
        if column_name and column_name in df.columns:
            return df[column_name].fillna(default_value).astype(str)
        else:
            return pd.Series([default_value] * len(df), index=df.index)
    
    def apply_genre_filter(df, genre_filter):
        genre_col = get_genre_column(df)
        if genre_col:
            return df[df[genre_col].str.contains(genre_filter, case=False, na=False)]
        else:
            return pd.DataFrame()

def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """
    Creates the final, best-performing 'soup' with a balanced weighting of features.
    Handles missing columns gracefully using column utilities.
    """
    df_copy = df.copy()

    # --- Final Optimized Weights ---
    w_overview = 4.0
    w_genre = 3.0
    w_title = 2.0
    w_director = 1.0

    # Get column names using utilities
    overview_col = get_overview_column(df_copy)
    genre_col = get_genre_column(df_copy)
    director_col = get_director_column(df_copy)
    
    # Clean and prepare the selected features with safe column access
    overview = safe_get_column_data(df_copy, overview_col, '')
    genre = safe_get_column_data(df_copy, genre_col, '')
    title = safe_get_column_data(df_copy, 'Series_Title', '')
    director = safe_get_column_data(df_copy, director_col, '')

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
        genre_filtered = apply_genre_filter(merged_df, genre_filter)
        
        if genre_filtered.empty:
            return pd.DataFrame()
        
        # Sort by rating and return top results
        rating_col = get_rating_column(genre_filtered)
        if rating_col:
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
            results = apply_genre_filter(results, genre_filter)
        
        return results.head(top_n)
    
    except Exception as e:
        # Return empty DataFrame if there's any error
        print(f"Error in content-based filtering: {e}")
        return pd.DataFrame()
