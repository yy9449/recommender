# content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st  # <-- Added Streamlit import

def _create_weighted_soup(df: pd.DataFrame) -> pd.Series:
    """Combines weighted features into a single string for vectorization."""
    df_copy = df.copy()
    
    # Define weights
    w_genre = 8.0
    w_rating = 1.5
    w_title = 0.5
    
    # Clean and prepare features
    df_copy['Genre'] = df_copy['Genre'].fillna('').astype(str)
    df_copy['Series_Title'] = df_copy['Series_Title'].fillna('').astype(str)
    
    # Normalize rating to be part of the text soup
    df_copy['IMDB_Rating_str'] = (df_copy['IMDB_Rating'].fillna(0) * 10).astype(int).astype(str)

    # Create weighted soup
    soup = (
        (df_copy['Genre'] + ' ') * int(w_genre) +
        (df_copy['Series_Title'] + ' ') * int(w_title) + 
        (df_copy['IMDB_Rating_str'] + ' ') * int(w_rating)
    )
    return soup

@st.cache_data # <-- Added Streamlit decorator for performance
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates movie recommendations based on weighted content features.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame(columns=merged_df.columns)

    # Create the feature matrix
    soup = _create_weighted_soup(merged_df)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get index of the target movie
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    # Get similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return merged_df.iloc[movie_indices]