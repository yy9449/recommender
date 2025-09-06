# content_based.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """
    Creates the final, best-performing 'soup' with a balanced weighting of features.
    """
    df_copy = df.copy()

    # --- Final Optimized Weights ---
    w_overview = 4.0
    w_genre = 3.0
    w_title = 2.0
    w_director = 1.0

    # Clean and prepare the selected features
    overview = df_copy['Overview'].fillna('').astype(str)
    genre = df_copy['Genre'].fillna('').astype(str)
    title = df_copy['Series_Title'].fillna('').astype(str)
    director = df_copy['Director'].fillna('').astype(str)

    # Combine the features with their weights
    soup = (
        (overview + ' ') * int(w_overview) +
        (genre + ' ') * int(w_genre) +
        (title + ' ') * int(w_title) +
        (director + ' ') * int(w_director)
    )
    return soup

@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates movie recommendations using the final, optimized TF-IDF content model.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    soup = _create_optimized_soup(merged_df)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    return merged_df.iloc[movie_indices]