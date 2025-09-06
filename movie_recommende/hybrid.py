# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st  # <-- Added Streamlit import

# Define weights
ALPHA = 0.4  # Content-based
BETA = 0.4   # Collaborative
GAMMA = 0.1  # Popularity
DELTA = 0.1  # Recency

def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    """Calculates popularity score: IMDB_Rating * log(votes)."""
    log_votes = np.log1p(df['No_of_Votes'])
    return df['IMDB_Rating'] * log_votes

def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    """Calculates recency score based on exponential decay."""
    current_year = pd.to_datetime('today').year
    decay_rate = 0.98
    age = current_year - df['Released_Year']
    return decay_rate ** age

@st.cache_data # <-- Added Streamlit decorator for performance
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame, 
    _user_ratings_df: pd.DataFrame, # Placeholder for consistent API with main.py
    content_sim_matrix: np.ndarray,
    collab_sim_matrix: np.ndarray,
    target_movie: str,
    top_n: int = 10
):
    """
    Generates hybrid recommendations by blending content, collaborative, popularity, and recency scores.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame(columns=merged_df.columns)
        
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]
    
    # 1. Get Scores
    content_scores = content_sim_matrix[idx]
    collab_scores = collab_sim_matrix[idx]
    popularity_scores = _calculate_popularity(merged_df)
    recency_scores = _calculate_recency(merged_df)
    
    # 2. Scale Scores
    scaler = MinMaxScaler()
    scaled_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    scaled_collab = scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()
    scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
    scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()
    
    # 3. Calculate Final Hybrid Score
    final_scores = (
        ALPHA * scaled_content +
        BETA * scaled_collab +
        GAMMA * scaled_popularity +
        DELTA * scaled_recency
    )
    
    # 4. Get Top N recommendations
    sim_scores = list(enumerate(final_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    
    movie_indices = [i[0] for i in sim_scores]
    return merged_df.iloc[movie_indices]