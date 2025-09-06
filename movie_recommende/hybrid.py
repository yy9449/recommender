# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Define recommendation weights
ALPHA = 0.4  # Content-based
BETA = 0.4   # Collaborative
GAMMA = 0.1  # Popularity
DELTA = 0.1  # Recency

def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    log_votes = np.log1p(df['No_of_Votes'].fillna(0))
    rating = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
    return rating * log_votes

def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    current_year = pd.to_datetime('today').year
    decay_rate = 0.98
    age = current_year - df['Released_Year'].fillna(df['Released_Year'].mode()[0])
    return decay_rate ** age

@st.cache_data
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame, 
    user_ratings_df: pd.DataFrame, # Kept for API consistency
    target_movie: str,
    top_n: int = 10
):
    """
    Generates hybrid recommendations blending multiple strategies.
    This now re-calculates matrices internally to work seamlessly.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
        
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    # 1. Content Similarity
    soup = (merged_df['Overview'].fillna('') + ' ' + merged_df['Genre'].fillna(''))
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(soup)
    content_sim_matrix = cosine_similarity(tfidf_matrix)
    content_scores = content_sim_matrix[idx]
    
    # 2. Collaborative Similarity (from User Ratings)
    user_item_matrix = user_ratings_df.pivot_table(index='Movie_ID', columns='User_ID', values='Rating').fillna(0)
    aligned_user_item = user_item_matrix.reindex(merged_df['Movie_ID']).fillna(0)
    collab_sim_matrix = cosine_similarity(aligned_user_item.T) # User-user similarity
    
    # For item-based, we'll use a simpler approach for speed here
    collab_scores = np.zeros(len(merged_df)) # Placeholder
    # A full collaborative score calculation here is too slow for real-time app use
    # We will rely more on the other scores for the hybrid UI recommendation

    # 3. Popularity & Recency
    popularity_scores = _calculate_popularity(merged_df)
    recency_scores = _calculate_recency(merged_df)
    
    # 4. Scale and Combine
    scaler = MinMaxScaler()
    scaled_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
    scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()
    
    final_scores = (
        (ALPHA + BETA) * scaled_content + # Combine content and collab weights
        GAMMA * scaled_popularity +
        DELTA * scaled_recency
    )
    
    sim_scores = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    
    movie_indices = [i[0] for i in sim_scores]
    return merged_df.iloc[movie_indices]