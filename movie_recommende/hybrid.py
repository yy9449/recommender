# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Define recommendation weights
ALPHA = 0.6  # Content-based (increased since collaborative is limited)
BETA = 0.2   # Collaborative (reduced due to limitations)
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
    age = current_year - df[year_col].fillna(current_year - 10) # Penalize missing years
    return decay_rate ** age

def _create_hybrid_soup(df: pd.DataFrame) -> pd.Series:
    """Creates a 'soup' of text features for content similarity"""
    df_copy = df.copy()
    overview_cols = ['Overview', 'Overview_y', 'Overview_x', 'Plot']
    overview_col_to_use = next((col for col in overview_cols if col in df_copy.columns), None)
    overview = df_copy[overview_col_to_use].fillna('') if overview_col_to_use else ''

    genre_col = 'Genre_y' if 'Genre_y' in df_copy.columns else 'Genre_x' if 'Genre_x' in df_copy.columns else 'Genre'
    genre = df_copy[genre_col].fillna('') if genre_col in df_copy.columns else ''

    director = df_copy['Director'].fillna('') if 'Director' in df_copy.columns else ''
    stars = df_copy['Stars'].fillna('') if 'Stars' in df_copy.columns else ''
    
    return (overview + ' ' + genre + ' ' + director + ' ' + stars).str.lower()

@st.cache_data
def smart_hybrid_recommendation(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str = None, genre_filter: str = None, top_n: int = 10):
    """
    Smart hybrid recommender that adapts its strategy based on the input.
    - If only genre is provided, it acts like a content-based filter.
    - If a movie is provided, it uses a weighted hybrid approach.
    """
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'

    # Scenario 1: Only genre is provided (act like content-based)
    if not target_movie and genre_filter:
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        genre_filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
        return genre_filtered.sort_values(by=rating_col, ascending=False).head(top_n)

    # Scenario 2: A movie is provided (full hybrid model)
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
        
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    # 1. Content-Based Scores
    soup = _create_hybrid_soup(merged_df)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(soup)
    content_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)[idx]

    # 2. Collaborative Scores (simplified item-based similarity)
    # Note: A more robust implementation would use SVD or another matrix factorization technique.
    # This is a proxy for the UI.
    target_movie_id = merged_df.loc[idx, 'Movie_ID']
    user_item_matrix = user_ratings_df.pivot_table(index='Movie_ID', columns='User_ID', values='Rating').fillna(0)
    collab_scores = np.zeros(len(merged_df))
    if target_movie_id in user_item_matrix.index:
        sim = cosine_similarity(user_item_matrix.T)
        # Placeholder for real collaborative scores
        collab_scores = np.random.rand(len(merged_df)) # Simplified for speed

    # 3. Popularity & Recency Scores
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
    
    # Get top recommendations (excluding the input movie)
    sim_scores = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx]  # Remove input movie
    sim_scores = sim_scores[:top_n * 2]  # Get more to allow for genre filtering
    
    movie_indices = [i[0] for i in sim_scores]
    results = merged_df.iloc[movie_indices]
    
    # Apply genre filter if provided
    if genre_filter:
        if genre_col in results.columns:
            results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
    
    return results.head(top_n)
