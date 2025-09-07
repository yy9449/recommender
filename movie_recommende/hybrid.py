import pandas as pd
import numpy as np
import streamlit as st
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced

@st.cache_data
def hybrid_recommendation_enhanced(merged_df, target_movie=None, genre=None, top_n=5):
    """Enhanced hybrid recommendation system"""
    if not target_movie and not genre:
        return None
    
    # Get recommendations from both approaches
    collab_recs = collaborative_filtering_enhanced(merged_df, target_movie, top_n * 2) if target_movie else None
    content_recs = content_based_filtering_enhanced(merged_df, target_movie, genre, top_n * 2)
    
    if collab_recs is None and content_recs is None:
        return None
    
    movie_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    
    # Weight collaborative filtering results (60%)
    if collab_recs is not None:
        for idx, row in collab_recs.iterrows():
            title = row['Series_Title']
            score = row[rating_col] * 0.6
            movie_scores[title] = movie_scores.get(title, 0) + score
    
    # Weight content-based results (40%)
    if content_recs is not None:
        for idx, row in content_recs.iterrows():
            title = row['Series_Title']
            score = row[rating_col] * 0.4
            movie_scores[title] = movie_scores.get(title, 0) + score
    
    # Sort by combined score
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_movies:
        return None
    
    # Get the final recommendation dataframe
    rec_titles = [movie[0] for movie in sorted_movies]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    
    # Preserve the order of recommendations
    result_df = result_df.set_index('Series_Title').loc[rec_titles].reset_index()
    
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)
