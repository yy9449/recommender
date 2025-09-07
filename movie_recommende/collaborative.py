# collaborative.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates recommendations using an item-based K-Nearest Neighbors (KNN) model.
    It finds movies similar to the target_movie based on user rating patterns.
    """
    # Step 1: Find the Movie_ID for the selected movie title
    target_movie_id_series = merged_df[merged_df['Series_Title'] == target_movie]['Movie_ID']
    
    # Handle case where the movie title is not found
    if target_movie_id_series.empty:
        print(f"Warning: Movie '{target_movie}' not found in the movie database.")
        return pd.DataFrame()
    target_movie_id = target_movie_id_series.iloc[0]

    # Step 2: Create the user-item matrix from ratings data
    # Rows: Movie_ID, Columns: User_ID, Values: Rating
    try:
        user_item_matrix = user_ratings_df.pivot_table(
            index='Movie_ID', 
            columns='User_ID', 
            values='Rating'
        ).fillna(0)
    except Exception as e:
        print(f"Error creating pivot table: {e}")
        return pd.DataFrame()

    # Handle case where the movie has no ratings
    if target_movie_id not in user_item_matrix.index:
        print(f"Warning: No ratings found for '{target_movie}'. Cannot use collaborative filtering.")
        return pd.DataFrame()

    # Step 3: Fit the KNN model
    # We use cosine similarity to find movies with similar rating vectors
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix.values)
    
    # Step 4: Find the nearest neighbors for the target movie
    query_index = user_item_matrix.index.get_loc(target_movie_id)
    distances, indices = knn.kneighbors(
        user_item_matrix.iloc[query_index, :].values.reshape(1, -1), 
        n_neighbors=top_n + 1  # +1 to include the movie itself
    )
    
    # Step 5: Get the recommended Movie_IDs and format the output
    recommended_movie_indices = indices.flatten()
    
    # Exclude the first item (which is the target movie itself)
    recommended_movie_ids = [user_item_matrix.index[i] for i in recommended_movie_indices[1:]]
    
    # Retrieve movie details from the main dataframe
    recommendations_df = merged_df[merged_df['Movie_ID'].isin(recommended_movie_ids)].copy()
    
    # Add a 'Genre' column if it doesn't exist for consistency with other functions
    if 'Genre' not in recommendations_df.columns and 'Genre_x' in recommendations_df.columns:
        recommendations_df['Genre'] = recommendations_df['Genre_x']
        
    return recommendations_df
