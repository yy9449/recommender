# collaborative.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st  # <-- Added Streamlit import

def _get_movie_id(title: str, merged_df: pd.DataFrame):
    """Helper to get Movie_ID from Series_Title."""
    result = merged_df[merged_df['Series_Title'] == title]
    if not result.empty:
        return result['Movie_ID'].iloc[0]
    return None

@st.cache_data # <-- Added Streamlit decorator for performance
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates recommendations using item-based KNN collaborative filtering.
    """
    target_movie_id = _get_movie_id(target_movie, merged_df)
    if target_movie_id is None or user_ratings_df is None or user_ratings_df.empty:
        return pd.DataFrame(columns=merged_df.columns)

    # Create user-item matrix
    user_item_matrix = user_ratings_df.pivot_table(index='Movie_ID', columns='User_ID', values='Rating').fillna(0)
    
    if target_movie_id not in user_item_matrix.index:
        # Not enough rating data for this movie
        return pd.DataFrame(columns=merged_df.columns)

    # Fit KNN model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix.values)
    
    # Find nearest neighbors
    query_index = user_item_matrix.index.get_loc(target_movie_id)
    distances, indices = knn.kneighbors(user_item_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=top_n + 1)
    
    # Get recommended movie IDs and titles
    recommended_movie_ids = []
    for i in range(1, len(distances.flatten())): # Start from 1 to exclude the movie itself
        movie_id = user_item_matrix.index[indices.flatten()[i]]
        recommended_movie_ids.append(movie_id)
        
    result_df = merged_df[merged_df['Movie_ID'].isin(recommended_movie_ids)]
    return result_df