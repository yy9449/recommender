# content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """
    Creates the 'soup' of text features using the CORRECTED column names ('_x' suffixes)
    that result from the data merge in main.py.
    """
    df_copy = df.copy()

    # --- Use the correct column names after the merge ---
    overview = df_copy['Overview_x'].fillna('').astype(str)
    genre = df_copy['Genre_x'].fillna('').astype(str)
    title = df_copy['Series_Title'].fillna('').astype(str)
    director = df_copy['Director'].fillna('').astype(str)

    # Combine the features
    soup = overview + ' ' + genre + ' ' + title + ' ' + director
    return soup

@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates movie recommendations using a TF-IDF content model.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    # Create the feature soup using the corrected helper function
    soup = _create_optimized_soup(merged_df)

    # Initialize and fit the TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the target movie
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    # Get pairwise similarity scores, sorted
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top N most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:top_n + 1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the final recommended movies
    return merged_df.iloc[movie_indices]
