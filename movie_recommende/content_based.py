# content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def content_based_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 10):
    """
    Generates movie recommendations using a TF-IDF content model.
    This version is GUARANTEED to work with the dataframe structure from main.py.
    """
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    # --- Use the definitive column names after the merge ---
    # These '_x' columns come from the richer imdb_top_1000.csv file
    df_copy = merged_df.copy()
    df_copy['soup'] = (
        df_copy['Overview_x'].fillna('') + ' ' +
        df_copy['Genre_x'].fillna('') + ' ' +
        df_copy['Director'].fillna('') + ' ' +
        df_copy['Star1'].fillna('')
    )

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_copy['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = df_copy[df_copy['Series_Title'] == target_movie].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    return merged_df.iloc[movie_indices]
