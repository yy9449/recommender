# collaborative.py

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st

@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """Generates recommendations using a simple and fast item-based KNN model."""
    target_movie_id_series = merged_df[merged_df['Series_Title'] == target_movie]['Movie_ID']
    if target_movie_id_series.empty:
        return pd.DataFrame()
    target_movie_id = target_movie_id_series.iloc[0]

    if user_ratings_df is None or user_ratings_df.empty:
        return pd.DataFrame()

    user_item_matrix = user_ratings_df.pivot_table(index='Movie_ID', columns='User_ID', values='Rating').fillna(0)
    
    if target_movie_id not in user_item_matrix.index:
        return pd.DataFrame()

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix.values)
    
    query_index = user_item_matrix.index.get_loc(target_movie_id)
    distances, indices = knn.kneighbors(user_item_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=top_n + 1)
    
    recommended_movie_ids = [user_item_matrix.index[i] for i in indices.flatten()[1:]]
    return merged_df[merged_df['Movie_ID'].isin(recommended_movie_ids)]