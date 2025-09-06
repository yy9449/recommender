# collaborative.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates recommendations using item-based collaborative filtering with KNN.
    """
    # Check if target movie exists
    target_movie_id_series = merged_df[merged_df['Series_Title'] == target_movie]['Movie_ID']
    if target_movie_id_series.empty:
        return pd.DataFrame()
    
    target_movie_id = target_movie_id_series.iloc[0]

    # Create user-item matrix
    try:
        user_item_matrix = user_ratings_df.pivot_table(
            index='Movie_ID', 
            columns='User_ID', 
            values='Rating'
        ).fillna(0)
    except KeyError as e:
        # If the expected columns don't exist, return empty DataFrame
        print(f"Missing column in user_ratings_df: {e}")
        return pd.DataFrame()
    
    # Check if target movie has ratings
    if target_movie_id not in user_item_matrix.index:
        return pd.DataFrame()  # Not enough rating data

    # Fit KNN model
    try:
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(user_item_matrix.values)
        
        # Get the index of target movie in the matrix
        query_index = user_item_matrix.index.get_loc(target_movie_id)
        
        # Find similar movies
        distances, indices = knn.kneighbors(
            user_item_matrix.iloc[query_index, :].values.reshape(1, -1), 
            n_neighbors=min(top_n + 1, len(user_item_matrix))
        )
        
        # Get recommended movie IDs (excluding the target movie itself)
        recommended_movie_ids = [user_item_matrix.index[i] for i in indices.flatten()[1:]]
        
        # Return the recommended movies from merged_df
        recommendations = merged_df[merged_df['Movie_ID'].isin(recommended_movie_ids)]
        
        # Sort by the order of recommendations
        recommendations = recommendations.set_index('Movie_ID').loc[recommended_movie_ids].reset_index()
        
        return recommendations[:top_n]
        
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return pd.DataFrame()

def load_user_ratings(file_path: str = None):
    """
    Placeholder function for loading user ratings.
    This function is referenced in main.py but not actually used.
    """
    if file_path:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading user ratings: {e}")
            return None
    return None

def diagnose_data_linking(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame):
    """
    Placeholder function for diagnosing data linking issues.
    This function is referenced in main.py but not actually used.
    """
    if merged_df is None or user_ratings_df is None:
        return "One or both dataframes are None"
    
    movie_ids_in_merged = set(merged_df['Movie_ID'].unique()) if 'Movie_ID' in merged_df.columns else set()
    movie_ids_in_ratings = set(user_ratings_df['Movie_ID'].unique()) if 'Movie_ID' in user_ratings_df.columns else set()
    
    overlap = movie_ids_in_merged.intersection(movie_ids_in_ratings)
    
    return {
        'merged_df_movies': len(movie_ids_in_merged),
        'user_ratings_movies': len(movie_ids_in_ratings),
        'overlap': len(overlap),
        'overlap_percentage': len(overlap) / len(movie_ids_in_merged) * 100 if movie_ids_in_merged else 0
    }
