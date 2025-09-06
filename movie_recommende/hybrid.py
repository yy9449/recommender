# hybrid.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Import column utilities (place column_utils.py in the same directory)
try:
    from column_utils import (
        get_genre_column, get_overview_column, get_rating_column, 
        get_year_column, get_votes_column, safe_get_column_data,
        apply_genre_filter
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
    
    def get_year_column(df):
        for col in ['Released_Year', 'Year', 'Released_Year_y', 'Released_Year_x']:
            if col in df.columns:
                return col
        return None
    
    def get_votes_column(df):
        for col in ['No_of_Votes', 'Votes', 'No_of_Votes_y', 'No_of_Votes_x']:
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

# Define recommendation weights
ALPHA = 0.6  # Content-based (increased since collaborative is limited)
BETA = 0.2   # Collaborative (reduced due to limitations)
GAMMA = 0.15 # Popularity
DELTA = 0.05 # Recency

def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    """Calculate popularity score based on votes and ratings"""
    votes_col = get_votes_column(df)
    rating_col = get_rating_column(df)
    
    if votes_col and rating_col:
        log_votes = np.log1p(df[votes_col].fillna(0))
        rating = df[rating_col].fillna(df[rating_col].mean())
        return rating * log_votes
    elif rating_col:
        # If no votes column, use rating only
        return df[rating_col].fillna(df[rating_col].mean())
    else:
        # Return zeros if no rating data
        return pd.Series([0] * len(df), index=df.index)

def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    """Calculate recency score based on release year"""
    year_col = get_year_column(df)
    current_year = pd.to_datetime('today').year
    decay_rate = 0.98
    
    if year_col:
        default_year = df[year_col].mode()[0] if not df[year_col].mode().empty else 2000
        age = current_year - df[year_col].fillna(default_year)
        return decay_rate ** age
    else:
        # Return neutral scores if no year data
        return pd.Series([0.5] * len(df), index=df.index)

@st.cache_data
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame, 
    user_ratings_df: pd.DataFrame = None,  # Made optional
    target_movie: str = None,
    genre_filter: str = None,
    top_n: int = 10
):
    """
    Generates hybrid recommendations blending multiple strategies.
    Works with or without user ratings data.
    """
    
    # If neither movie nor genre is provided, return empty
    if not target_movie and not genre_filter:
        return pd.DataFrame()
    
    # Handle genre-only case
    if genre_filter and not target_movie:
        genre_filtered = apply_genre_filter(merged_df, genre_filter)
        
        if genre_filtered.empty:
            return pd.DataFrame()
        
        # For genre-only, use popularity and recency
        popularity_scores = _calculate_popularity(genre_filtered)
        recency_scores = _calculate_recency(genre_filtered)
        
        # Scale and combine
        scaler = MinMaxScaler()
        scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
        scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()
        
        # Weighted combination for genre-only
        final_scores = 0.7 * scaled_popularity + 0.3 * scaled_recency
        
        # Sort and return top results
        genre_filtered = genre_filtered.copy()
        genre_filtered['hybrid_score'] = final_scores
        results = genre_filtered.sort_values('hybrid_score', ascending=False).head(top_n)
        return results.drop('hybrid_score', axis=1)
    
    # Movie-based recommendations
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
        
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    try:
        # 1. Content Similarity
        genre_col = get_genre_column(merged_df)
        overview_col = get_overview_column(merged_df)
        
        # Get text data safely
        overview_text = safe_get_column_data(merged_df, overview_col, '')
        genre_text = safe_get_column_data(merged_df, genre_col, '')
        
        soup = overview_text + ' ' + genre_text
        tfidf_matrix = TfidfVectorizer(stop_words='english', max_features=5000).fit_transform(soup)
        content_sim_matrix = cosine_similarity(tfidf_matrix)
        content_scores = content_sim_matrix[idx]
        
        # 2. Collaborative Similarity (simplified version)
        collab_scores = np.zeros(len(merged_df))
        if user_ratings_df is not None:
            try:
                # Simple collaborative approach based on movie ratings
                target_movie_id = merged_df.iloc[idx]['Movie_ID']
                if target_movie_id in user_ratings_df['Movie_ID'].values:
                    # Get users who rated the target movie highly
                    high_raters = user_ratings_df[
                        (user_ratings_df['Movie_ID'] == target_movie_id) & 
                        (user_ratings_df['Rating'] >= 7)
                    ]['User_ID'].unique()
                    
                    if len(high_raters) > 0:
                        # Get other movies these users rated highly
                        similar_movies = user_ratings_df[
                            (user_ratings_df['User_ID'].isin(high_raters)) & 
                            (user_ratings_df['Rating'] >= 7)
                        ]['Movie_ID'].value_counts()
                        
                        # Map back to dataframe indices
                        for movie_id, count in similar_movies.items():
                            movie_indices = merged_df[merged_df['Movie_ID'] == movie_id].index
                            if len(movie_indices) > 0:
                                collab_scores[movie_indices[0]] = count / len(high_raters)
            except Exception:
                # If collaborative fails, keep zeros
                pass

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
        
        # Get top recommendations (excluding the input movie)
        sim_scores = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)
        sim_scores = [x for x in sim_scores if x[0] != idx]  # Remove input movie
        sim_scores = sim_scores[:top_n * 2]  # Get more to allow for genre filtering
        
        movie_indices = [i[0] for i in sim_scores]
        results = merged_df.iloc[movie_indices]
        
        # Apply genre filter if provided
        if genre_filter:
            results = apply_genre_filter(results, genre_filter)
        
        return results.head(top_n)
    
    except Exception as e:
        # Return empty DataFrame if there's any error
        print(f"Error in hybrid filtering: {e}")
        return pd.DataFrame()
