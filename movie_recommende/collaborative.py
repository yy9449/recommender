import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from content_based import find_similar_titles

@st.cache_data
def create_user_item_matrix(merged_df):
    """Create a synthetic user-item matrix based on movie characteristics"""
    np.random.seed(42)
    
    user_types = {
        'action_lover': {'Action': 5, 'Adventure': 4, 'Thriller': 4, 'Drama': 2, 'Comedy': 2, 'Romance': 1},
        'drama_fan': {'Drama': 5, 'Romance': 4, 'Biography': 4, 'Action': 2, 'Comedy': 3, 'Thriller': 2},
        'comedy_fan': {'Comedy': 5, 'Romance': 4, 'Family': 4, 'Action': 2, 'Drama': 3, 'Horror': 1},
        'thriller_fan': {'Thriller': 5, 'Mystery': 4, 'Crime': 4, 'Horror': 3, 'Action': 4, 'Comedy': 2},
        'classic_lover': {'Drama': 4, 'Romance': 4, 'Biography': 5, 'History': 5, 'War': 4, 'Comedy': 3}
    }
    
    user_movie_ratings = {}
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    for user_type, preferences in user_types.items():
        user_ratings = []
        for _, movie in merged_df.iterrows():
            rating = 0
            if pd.notna(movie[genre_col]):
                genres = [g.strip() for g in movie[genre_col].split(',')]
                genre_scores = [preferences.get(genre, 0) for genre in genres]
                if genre_scores:
                    base_rating = np.mean(genre_scores)
                    rating = max(1, min(5, base_rating + np.random.normal(0, 0.5)))
                    if np.random.random() < 0.3:
                        rating = 0
            user_ratings.append(rating)
        user_movie_ratings[user_type] = user_ratings
    
    rating_matrix = np.array(list(user_movie_ratings.values()))
    user_names = list(user_movie_ratings.keys())
    
    return rating_matrix, user_names

@st.cache_data
def collaborative_filtering_enhanced(merged_df, target_movie, top_n=5):
    """Enhanced collaborative filtering"""
    if not target_movie:
        return None
    
    similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
    if not similar_titles:
        return None
    
    target_title = similar_titles[0]
    target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
    
    rating_matrix, user_names = create_user_item_matrix(merged_df)
    user_similarity = cosine_similarity(rating_matrix)
    
    target_movie_idx = merged_df.index.get_loc(target_idx)
    target_ratings = rating_matrix[:, target_movie_idx]
    
    user_scores = []
    for user_idx, rating in enumerate(target_ratings):
        if rating > 3:
            avg_similarity = np.mean([user_similarity[user_idx][other_idx] 
                                    for other_idx in range(len(user_names)) 
                                    if other_idx != user_idx])
            user_scores.append((user_idx, rating * avg_similarity))
    
    if not user_scores:
        return None
    
    user_scores.sort(key=lambda x: x[1], reverse=True)
    top_users = user_scores[:3]
    
    movie_scores = {}
    for user_idx, user_weight in top_users:
        user_ratings = rating_matrix[user_idx]
        for movie_idx, rating in enumerate(user_ratings):
            if rating > 3 and movie_idx != target_movie_idx:
                movie_title = merged_df.iloc[movie_idx]['Series_Title']
                if movie_title not in movie_scores:
                    movie_scores[movie_title] = 0
                movie_scores[movie_title] += rating * user_weight
    
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not recommendations:
        return None
    
    rec_titles = [rec[0] for rec in recommendations]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)
