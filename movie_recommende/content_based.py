# content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """Creates the best-performing 'soup' with a balanced weighting of features."""
    w_overview, w_genre, w_title, w_director = 4, 3, 2, 1
    soup = (
        (df['Overview'].fillna('') + ' ') * w_overview +
        (df['Genre'].fillna('') + ' ') * w_genre +
        (df['Series_Title'].fillna('') + ' ') * w_title +
        (df['Director'].fillna('') + ' ') * w_director
    )
    return soup

@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """Generates movie recommendations using the optimized TF-IDF content model."""
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