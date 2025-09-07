# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Define recommendation weights ---
ALPHA = 0.5  # Content-based
BETA = 0.5   # Collaborative (using popularity as a proxy here)

def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    """Calculates a popularity score."""
    log_votes = np.log1p(df['No_of_Votes'].fillna(0))
    rating = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
    return rating * log_votes

@st.cache_data
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame, 
    user_ratings_df: pd.DataFrame, 
    target_movie: str,
    top_n: int = 10
):
    """
    Generates hybrid recommendations blending content and popularity.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    # --- 1. Content-Based Similarity ---
    # Uses CORRECTED column names
    soup = (
        merged_df['Overview_x'].fillna('') + ' ' + 
        merged_df['Genre_x'].fillna('') + ' ' + 
        merged_df['Director'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]
    content_scores = cosine_sim[idx]

    # --- 2. Popularity Score (as a proxy for collaborative filtering) ---
    popularity_scores = _calculate_popularity(merged_df)
    
    # --- 3. Scale and Combine ---
    scaler = MinMaxScaler()
    scaled_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
    
    # Combine scores with weights
    final_scores = (ALPHA * scaled_content) + (BETA * scaled_popularity)
    
    # --- 4. Get Recommendations ---
    # Add scores to the dataframe
    merged_df['hybrid_score'] = final_scores
    
    # Sort by score and get the top N, excluding the target movie itself
    recommendations = merged_df.sort_values(by='hybrid_score', ascending=False)
    recommendations = recommendations[recommendations['Series_Title'] != target_movie]
    
    return recommendations.head(top_n)
