# content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def _create_soup(df: pd.DataFrame) -> pd.Series:
    """
    Creates the 'soup' of text features using the GUARANTEED column names
    that result from the data merge in main.py.
    """
    # --- Use the definitive column names after the merge ---
    overview = df['Overview_x'].fillna('').astype(str)
    genre = df['Genre_x'].fillna('').astype(str)
    title = df['Series_Title'].fillna('').astype(str)
    director = df['Director'].fillna('').astype(str)

    # Combine the features
    return overview + ' ' + genre + ' ' + title + ' ' + director

@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates movie recommendations using a TF-IDF content model.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    soup = _create_soup(merged_df)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]

    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    return merged_df.iloc[movie_indices]
