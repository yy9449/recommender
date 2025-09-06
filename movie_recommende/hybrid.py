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
    """Calculate popularity score based on votes and ratings."""
    try:
        log_votes = np.log1p(df['No_of_Votes'].fillna(0))
        rating = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
        return rating * log_votes
    except Exception:
        # Fallback if columns don't exist
        return pd.Series([1.0] * len(df), index=df.index)

def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    """Calculate recency score based on release year."""
    try:
        current_year = pd.Timestamp.now().year
        decay_rate = 0.98
        released_year = df['Released_Year'].fillna(df['Released_Year'].mode()[0] if not df['Released_Year'].mode().empty else current_year - 10)
        age = current_year - released_year
        return decay_rate ** age
    except Exception:
        # Fallback if columns don't exist
        return pd.Series([1.0] * len(df), index=df.index)

@st.cache_data
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame, 
    user_ratings_df: pd.DataFrame,
    target_movie: str,
    top_n: int = 10
):
    """
    Generates hybrid recommendations blending multiple strategies.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
        
    try:
        idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

        # 1. Content Similarity
        overview = merged_df['Overview'].fillna('').astype(str)
        genre = merged_df['Genre'].fillna('').astype(str)
        soup = overview + ' ' + genre
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(soup)
        content_sim_matrix = cosine_similarity(tfidf_matrix)
        content_scores = content_sim_matrix[idx]
        
        # 2. Collaborative Similarity (simplified for performance)
        # Create a basic user-item similarity
        try:
            if user_ratings_df is not None and not user_ratings_df.empty:
                user_item_matrix = user_ratings_df.pivot_table(
                    index='Movie_ID', 
                    columns='User_ID', 
                    values='Rating'
                ).fillna(0)
                
                # Align with merged_df
                aligned_ratings = user_item_matrix.reindex(merged_df['Movie_ID']).fillna(0)
                
                if len(aligned_ratings) > 1:
                    collab_sim_matrix = cosine_similarity(aligned_ratings.values)
                    if idx < len(collab_sim_matrix):
                        collab_scores = collab_sim_matrix[idx]
                    else:
                        collab_scores = np.zeros(len(merged_df))
                else:
                    collab_scores = np.zeros(len(merged_df))
            else:
                collab_scores = np.zeros(len(merged_df))
        except Exception as e:
            print(f"Collaborative filtering error: {e}")
            collab_scores = np.zeros(len(merged_df))

        # 3. Popularity & Recency
        popularity_scores = _calculate_popularity(merged_df)
        recency_scores = _calculate_recency(merged_df)
        
        # 4. Scale and Combine
        scaler = MinMaxScaler()
        
        # Ensure all scores are the same length
        n_movies = len(merged_df)
        if len(content_scores) != n_movies:
            content_scores = np.pad(content_scores, (0, max(0, n_movies - len(content_scores))), 'constant')[:n_movies]
        if len(collab_scores) != n_movies:
            collab_scores = np.pad(collab_scores, (0, max(0, n_movies - len(collab_scores))), 'constant')[:n_movies]
            
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
        
        # Get top recommendations (excluding the target movie)
        sim_scores = list(enumerate(final_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Remove the target movie from recommendations
        sim_scores = [x for x in sim_scores if x[0] != idx]
        sim_scores = sim_scores[:top_n]
        
        movie_indices = [i[0] for i in sim_scores]
        return merged_df.iloc[movie_indices]
        
    except Exception as e:
        print(f"Error in hybrid recommendation: {e}")
        return pd.DataFrame()
