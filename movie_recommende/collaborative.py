# collaborative.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

@st.cache_data
def load_user_ratings():
    """
    Loads the user ratings dataframe from the local 'user_movie_rating.csv' file.
    Returns None if the file is not found or is empty.
    """
    try:
        df = pd.read_csv('user_movie_rating.csv')
        if df.empty:
            st.warning("Warning: 'user_movie_rating.csv' is empty.")
            return None
        return df
    except FileNotFoundError:
        st.error("Error: 'user_movie_rating.csv' not found. Collaborative filtering will not work.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading user ratings: {e}")
        return None

def collaborative_knn(merged_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Performs item-based K-Nearest Neighbors collaborative filtering.
    """
    ratings_df = load_user_ratings()
    if ratings_df is None:
        return pd.DataFrame()

    # Find the Movie_ID for the selected movie title
    target_movie_id_series = merged_df.loc[merged_df['Series_Title'] == target_movie, 'Movie_ID']
    if target_movie_id_series.empty:
        return pd.DataFrame()
    target_movie_id = target_movie_id_series.iloc[0]

    # Create the user-item matrix for the KNN model
    user_item_matrix = ratings_df.pivot_table(index='Movie_ID', columns='User_ID', values='Rating').fillna(0)
    
    if target_movie_id not in user_item_matrix.index:
        st.warning(f"No rating data available for '{target_movie}' to use collaborative filtering.")
        return pd.DataFrame()

    # Fit the KNN model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix.values)
    
    # Find the nearest neighbors for the target movie
    query_index = user_item_matrix.index.get_loc(target_movie_id)
    distances, indices = model.kneighbors(
        user_item_matrix.iloc[query_index, :].values.reshape(1, -1),
        n_neighbors=top_n + 1  # +1 to account for the movie itself
    )
    
    # Get the Movie_IDs of recommendations and exclude the target movie
    recommended_ids = [user_item_matrix.index[i] for i in indices.flatten()][1:]
    
    # Retrieve movie details from the main dataframe
    recommendations = merged_df[merged_df['Movie_ID'].isin(recommended_ids)].copy()
    
    # Ensure column name consistency for the UI display
    if 'Genre' not in recommendations.columns and 'Genre_x' in recommendations.columns:
        recommendations['Genre'] = recommendations['Genre_x']
        
    return recommendations

@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, user_ratings_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    This is a wrapper for the KNN logic to maintain a consistent API for main.py.
    The 'user_ratings_df' argument is ignored because load_user_ratings() is now called internally.
    """
    return collaborative_knn(merged_df, target_movie, top_n=top_n)

@st.cache_data
def diagnose_data_linking(merged_df: pd.DataFrame):
    """
    Checks for potential issues in the data that could affect recommendations and
    returns a dictionary of diagnostic results.
    """
    issues = {}
    issues['has_movie_id'] = 'Movie_ID' in merged_df.columns
    issues['unique_titles'] = merged_df['Series_Title'].nunique()
    issues['total_rows'] = len(merged_df)
    
    ratings = load_user_ratings()
    issues['ratings_loaded'] = ratings is not None and not ratings.empty
    
    if issues['ratings_loaded'] and issues['has_movie_id']:
        movie_ids_in_main_df = set(merged_df['Movie_ID'].unique())
        movie_ids_in_ratings = set(ratings['Movie_ID'].unique())
        issues['overlapping_ids'] = len(movie_ids_in_main_df.intersection(movie_ids_in_ratings))
        issues['percent_overlap'] = (issues['overlapping_ids'] / len(movie_ids_in_main_df)) * 100 if len(movie_ids_in_main_df) > 0 else 0
    else:
        issues['overlapping_ids'] = 0
        issues['percent_overlap'] = 0.0
        
    return issues
