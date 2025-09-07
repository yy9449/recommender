# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Your defined recommendation weights ---
ALPHA = 0.4  # Content-based
BETA = 0.4   # Collaborative
GAMMA = 0.1  # Popularity
DELTA = 0.1  # Recency

def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    """Calculates a popularity score based on votes and ratings."""
    # Ensure using the correct column 'No_of_Votes' from imdb_top_1000.csv
    log_votes = np.log1p(df['No_of_Votes'].fillna(0))
    rating = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
    return rating * log_votes

def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    """Calculates a recency score based on the movie's release year."""
    current_year = pd.to_datetime('today').year
    decay_rate = 0.98
    # Ensure using 'Released_Year' from imdb_top_1000.csv
    age = current_year - df['Released_Year'].fillna(df['Released_Year'].mode()[0])
    # Clip age at 0 to prevent issues with future-dated movies
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
    Generates hybrid recommendations by blending content, collaborative signals,
    popularity, and recency scores using your specified weights.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    # --- 1. Content-Based Similarity ---
    # Using the corrected post-merge column names ('_x')
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

    # --- 2. Collaborative Similarity (Placeholder/Proxy) ---
    # For a real-time app, a full collaborative calculation is complex.
    # We will use the Content score as the primary base for both content and collab weights.
    # A more advanced implementation could use a pre-trained model here.
    collaborative_scores = content_scores # Using content as a proxy

    # --- 3. Popularity & Recency Scores ---
    popularity_scores = _calculate_popularity(merged_df)
    recency_scores = _calculate_recency(merged_df)

    # --- 4. Scale All Scores to be between 0 and 1 ---
    scaler = MinMaxScaler()
    scaled_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    scaled_collaborative = scaler.fit_transform(collaborative_scores.reshape(-1, 1)).flatten()
    scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
    scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()

    # --- 5. Combine Scores with Your Weights ---
    final_scores = (
        ALPHA * scaled_content +
        BETA * scaled_collaborative +
        GAMMA * scaled_popularity +
        DELTA * scaled_recency
    )
    
    # --- 6. Get Recommendations ---
    merged_df['hybrid_score'] = final_scores
    recommendations = merged_df.sort_values(by='hybrid_score', ascending=False)
    recommendations = recommendations[recommendations['Series_Title'] != target_movie]
    
    return recommendations.head(top_n)
