# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Define recommendation weights
ALPHA = 0.5  # Content-based
BETA = 0.3   # Collaborative (reduced as it's a placeholder)
GAMMA = 0.15 # Popularity
DELTA = 0.05 # Recency

def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    """Calculate popularity score based on votes and ratings"""
    votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else 'Votes'
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'
    
    log_votes = np.log1p(df[votes_col].fillna(0))
    rating = df[rating_col].fillna(df[rating_col].mean())
    return rating * log_votes

def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    """Calculate recency score based on release year"""
    year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
    current_year = pd.to_datetime('today').year
    decay_rate = 0.98
    age = current_year - df[year_col].fillna(df[year_col].mode()[0] if not df[year_col].mode().empty else 2000)
    return decay_rate ** age

@st.cache_data
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame, 
    user_ratings_df: pd.DataFrame = None,
    target_movie: str = None,
    genre_filter: str = None,  # <-- FIX 1: Added genre_filter
    top_n: int = 10
):
    """
    Generates hybrid recommendations blending multiple strategies.
    Handles both movie-based and genre-based recommendations.
    """
    if not target_movie and not genre_filter:
        return pd.DataFrame()

    # Handle genre-only case
    if genre_filter and not target_movie:
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
        genre_filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
        
        if genre_filtered.empty:
            return pd.DataFrame()
            
        # For genre-only, just use popularity and recency
        popularity_scores = _calculate_popularity(genre_filtered)
        recency_scores = _calculate_recency(genre_filtered)
        
        scaler = MinMaxScaler()
        scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
        scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()
        
        genre_filtered['hybrid_score'] = 0.7 * scaled_popularity + 0.3 * scaled_recency
        return genre_filtered.sort_values('hybrid_score', ascending=False).head(top_n)

    # Movie-based recommendations
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
        
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    # FIX 2: Robust column handling for Overview and Genre
    # Handle Overview column
    overview_text = None
    for col in ['Overview', 'Overview_y', 'Overview_x', 'Plot']:
        if col in merged_df.columns:
            overview_text = merged_df[col].fillna('')
            break
    if overview_text is None:
        overview_text = pd.Series([''] * len(merged_df), index=merged_df.index)

    # Handle Genre column
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
    genre_text = merged_df[genre_col].fillna('') if genre_col in merged_df.columns else pd.Series([''] * len(merged_df), index=merged_df.index)

    # 1. Content Similarity
    soup = overview_text + ' ' + genre_text
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(soup)
    content_sim_matrix = cosine_similarity(tfidf_matrix)
    content_scores = content_sim_matrix[idx]
    
    # 2. Collaborative Similarity (Placeholder)
    collab_scores = np.zeros(len(merged_df))
    # Note: A full collaborative implementation is complex; this is a simplified placeholder.

    # 3. Popularity & Recency
    popularity_scores = _calculate_popularity(merged_df)
    recency_scores = _calculate_recency(merged_df)
    
    # 4. Scale and Combine
    scaler = MinMaxScaler()
    scaled_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    scaled_collab = scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()
    scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
    scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()
    
    final_scores = (
        ALPHA * scaled_content +
        BETA * scaled_collab +
        GAMMA * scaled_popularity +
        DELTA * scaled_recency
    )
    
    sim_scores = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx] # Exclude the movie itself
    
    movie_indices = [i[0] for i in sim_scores]
    results = merged_df.iloc[movie_indices]

    # Apply genre filter if provided
    if genre_filter:
        results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
        
    return results.head(top_n)
