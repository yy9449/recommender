# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Define recommendation weights for blending
ALPHA = 0.5  # Content-based
BETA = 0.5   # Collaborative signals (approximated)

@st.cache_data
def smart_hybrid_recommendation(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """Generates hybrid recommendations by blending content scores with popularity as a proxy for collaborative signals."""
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
        
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    # 1. Get Content-Based Scores
    soup = (merged_df['Overview'].fillna('') + ' ' + merged_df['Genre'].fillna(''))
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(soup)
    content_sim_matrix = cosine_similarity(tfidf_matrix)
    content_scores = content_sim_matrix[idx]
    
    # 2. Get Popularity Score (as a simple proxy for collaborative filtering)
    scaler = MinMaxScaler()
    popularity = merged_df['IMDB_Rating'].fillna(0) * np.log1p(merged_df['No_of_Votes'].fillna(0))
    popularity_scores = scaler.fit_transform(popularity.values.reshape(-1, 1)).flatten()

    # 3. Combine Scores
    final_scores = (ALPHA * content_scores) + (BETA * popularity_scores)
    
    # 4. Get Top N recommendations
    sim_scores = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1] # Exclude the movie itself
    
    movie_indices = [i[0] for i in sim_scores]
    return merged_df.iloc[movie_indices]