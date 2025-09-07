# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Your defined recommendation weights ---
ALPHA = 0.4  # Content-based
BETA = 0.4   # Collaborative (proxied by content for this real-time model)
GAMMA = 0.1  # Popularity
DELTA = 0.1  # Recency

def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    """Calculates a popularity score using the correct column names."""
    log_votes = np.log1p(df['No_of_Votes'].fillna(0))
    rating = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
    return rating * log_votes

def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    """Calculates a recency score using the correct 'Released_Year' column."""
    current_year = pd.to_datetime('today').year
    decay_rate = 0.98
    age = current_year - pd.to_numeric(df['Released_Year'], errors='coerce').fillna(df['Released_Year'].mode()[0])
    age = age.clip(lower=0)
    return decay_rate ** age

@st.cache_data
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame,
    user_ratings_df: pd.DataFrame,
    target_movie: str,
    top_n: int = 10
):
    """
    Generates hybrid recommendations blending all four strategies with corrected column names.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    # --- 1. Content-Based Similarity ---
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
    collaborative_scores = content_scores # Using content as a proxy for the collaborative signal

    # --- 2. Popularity & Recency Scores ---
    popularity_scores = _calculate_popularity(merged_df)
    recency_scores = _calculate_recency(merged_df)

    # --- 3. Scale All Scores ---
    scaler = MinMaxScaler()
    scaled_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    scaled_collaborative = scaler.fit_transform(collaborative_scores.reshape(-1, 1)).flatten()
    scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
    scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()

    # --- 4. Combine with Your Weights ---
    final_scores = (
        ALPHA * scaled_content +
        BETA * scaled_collaborative +
        GAMMA * scaled_popularity +
        DELTA * scaled_recency
    )
    
    # --- 5. Generate Final Recommendations ---
    merged_df['hybrid_score'] = final_scores
    recommendations = merged_df.sort_values(by='hybrid_score', ascending=False)
    recommendations = recommendations[recommendations['Series_Title'] != target_movie]
    
    return recommendations.head(top_n)
