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
    age = current_year - df[year_col].fillna(df[year_col].mode()[0] if not df[year_col].mode().empty else 2000)
    return decay_rate ** age

@st.cache_data
def smart_hybrid_recommendation(
    merged_df: pd.DataFrame, 
    user_ratings_df: pd.DataFrame = None,
    target_movie: str = None,
    genre_filter: str = None,  # <-- THIS LINE IS THE FIX
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
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
        if genre_col in merged_df.columns:
            genre_filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
        else:
            return pd.DataFrame()
        
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
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
        
        # Handle overview column with multiple possible names INCLUDING _x and _y variants
        overview_text = None
        for col in ['Overview', 'Overview_y', 'Overview_x', 'Plot', 'Description', 'Summary']:
            if col in merged_df.columns:
                overview_text = merged_df[col].fillna('')
                break
        if overview_text is None:
            overview_text = pd.Series([''] * len(merged_df), index=merged_df.index)
        
        # Handle genre column
        genre_text = merged_df[genre_col].fillna('') if genre_col in merged_df.columns else pd.Series([''] * len(merged_df), index=merged_df.index)
        
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
            if genre_col in results.columns:
                results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
        
        return results.head(top_n)
    
    except Exception as e:
        # Return empty DataFrame if there's any error
        print(f"Error in hybrid filtering: {e}")
        return pd.DataFrame()
