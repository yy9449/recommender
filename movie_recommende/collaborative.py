# collaborative.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates movie recommendations using an item-based K-Nearest Neighbors (KNN) model.
    This function now explicitly requires the user_ratings_df to be passed as an argument.
    """
    # Step 1: Validate inputs
    if user_ratings_df is None or user_ratings_df.empty:
        st.warning("User ratings data is not available. Cannot perform collaborative filtering.")
        return pd.DataFrame()

    target_movie_id_series = merged_df[merged_df['Series_Title'] == target_movie]['Movie_ID']
    if target_movie_id_series.empty:
        return pd.DataFrame()
    target_movie_id = target_movie_id_series.iloc[0]

    # Step 2: Create the user-item matrix from the provided ratings data
    user_item_matrix = user_ratings_df.pivot_table(index='Movie_ID', columns='User_ID', values='Rating').fillna(0)
    
    if target_movie_id not in user_item_matrix.index:
        st.warning(f"No rating data found for '{target_movie}' to use for collaborative filtering.")
        return pd.DataFrame()

    # Step 3: Fit the KNN model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix.values)
    
    # Step 4: Find the nearest neighbors for the target movie
    query_index = user_item_matrix.index.get_loc(target_movie_id)
    distances, indices = knn.kneighbors(
        user_item_matrix.iloc[query_index, :].values.reshape(1, -1), 
        n_neighbors=top_n + 1
    )
    
    # Step 5: Format and return the recommendations
    recommended_movie_ids = [user_item_matrix.index[i] for i in indices.flatten()][1:]
    recommendations_df = merged_df[merged_df['Movie_ID'].isin(recommended_movie_ids)].copy()
    
    return recommendations_df
