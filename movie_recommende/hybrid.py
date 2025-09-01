import pandas as pd
import numpy as np
import streamlit as st
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced, load_user_ratings

@st.cache_data
def hybrid_recommendation_enhanced(merged_df, target_movie=None, genre=None, top_n=5, use_both=False):
    """
    Enhanced hybrid recommendation system that can handle:
    1. Movie title only
    2. Genre only 
    3. Both movie title and genre
    4. Real user rating data integration
    """
    if not target_movie and not genre:
        return None
    
    # Load user ratings for collaborative filtering
    user_ratings_df = load_user_ratings()
    
    # Initialize results storage
    movie_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    # Determine which algorithms to use based on inputs
    if target_movie and genre and use_both:
        # Both inputs provided - use all three approaches
        st.info("🔄 Using hybrid approach with both movie title and genre preferences")
        
        # 1. Collaborative filtering based on movie (40%)
        collab_recs = collaborative_filtering_enhanced(merged_df, target_movie, top_n * 2)
        if collab_recs is not None:
            for idx, row in collab_recs.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * 0.4
                movie_scores[title] = movie_scores.get(title, 0) + score
        
        # 2. Content-based filtering on movie (30%)
        content_recs_movie = content_based_filtering_enhanced(merged_df, target_movie, None, top_n * 2)
        if content_recs_movie is not None:
            for idx, row in content_recs_movie.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * 0.3
                movie_scores[title] = movie_scores.get(title, 0) + score
        
        # 3. Content-based filtering on genre (30%)
        content_recs_genre = content_based_filtering_enhanced(merged_df, None, genre, top_n * 2)
        if content_recs_genre is not None:
            for idx, row in content_recs_genre.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * 0.3
                movie_scores[title] = movie_scores.get(title, 0) + score
        
        # Bonus for movies that appear in multiple recommendations
        title_counts = {}
        if collab_recs is not None:
            for title in collab_recs['Series_Title']:
                title_counts[title] = title_counts.get(title, 0) + 1
        if content_recs_movie is not None:
            for title in content_recs_movie['Series_Title']:
                title_counts[title] = title_counts.get(title, 0) + 1
        if content_recs_genre is not None:
            for title in content_recs_genre['Series_Title']:
                title_counts[title] = title_counts.get(title, 0) + 1
        
        # Apply consensus bonus
        for title, count in title_counts.items():
            if count > 1:
                bonus_multiplier = 1.2 if count == 2 else 1.5
                if title in movie_scores:
                    movie_scores[title] *= bonus_multiplier
    
    elif target_movie and not genre:
        # Movie title only - use collaborative + content-based on movie
        st.info("🎬 Using movie-based recommendations")
        
        # Collaborative filtering (60%)
        collab_recs = collaborative_filtering_enhanced(merged_df, target_movie, top_n * 2)
        if collab_recs is not None:
            for idx, row in collab_recs.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * 0.6
                movie_scores[title] = movie_scores.get(title, 0) + score
        
        # Content-based filtering (40%)
        content_recs = content_based_filtering_enhanced(merged_df, target_movie, None, top_n * 2)
        if content_recs is not None:
            for idx, row in content_recs.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * 0.4
                movie_scores[title] = movie_scores.get(title, 0) + score
    
    elif genre and not target_movie:
        # Genre only - use content-based filtering with genre weighting
        st.info("🎭 Using genre-based recommendations")
        
        content_recs = content_based_filtering_enhanced(merged_df, None, genre, top_n * 2)
        if content_recs is not None:
            for idx, row in content_recs.iterrows():
                title = row['Series_Title']
                # Add genre bonus for exact genre matches
                movie_info = merged_df[merged_df['Series_Title'] == title].iloc[0]
                movie_genres = []
                if pd.notna(movie_info[genre_col]):
                    movie_genres = [g.strip().lower() for g in movie_info[genre_col].split(',')]
                
                base_score = row[rating_col]
                genre_bonus = 1.0
                
                if genre.lower() in movie_genres:
                    genre_bonus = 1.3  # 30% bonus for exact genre match
                
                score = base_score * genre_bonus
                movie_scores[title] = score
    
    # Sort by combined score (high to low)
    if not movie_scores:
        return None
    
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Get the final recommendation dataframe
    rec_titles = [movie[0] for movie in sorted_movies]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    
    # Preserve the order of recommendations but also sort by IMDB rating as secondary
    title_to_score = dict(sorted_movies)
    result_df = result_df.copy()
    result_df['recommendation_score'] = result_df['Series_Title'].map(title_to_score)
    
    # Sort by recommendation score first, then by IMDB rating
    result_df = result_df.sort_values(['recommendation_score', rating_col], ascending=[False, False])
    
    # Remove the helper column and return
    result_df = result_df.drop('recommendation_score', axis=1)
    
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

@st.cache_data  
def get_genre_based_user_recommendations(merged_df, genre, top_n=5):
    """Get recommendations based on genre popularity and ratings"""
    if not genre:
        return None
    
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    
    # Filter movies by genre
    genre_movies = merged_df[merged_df[genre_col].str.contains(genre, case=False, na=False)]
    
    if genre_movies.empty:
        return None
    
    # Sort by rating (high to low)
    genre_movies = genre_movies.sort_values(by=rating_col, ascending=False)
    
    return genre_movies[['Series_Title', genre_col, rating_col]].head(top_n)

@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=5):
    """
    Smart hybrid that automatically determines the best approach based on available inputs
    """
    if target_movie and genre:
        # Both inputs - use enhanced hybrid
        return hybrid_recommendation_enhanced(merged_df, target_movie, genre, top_n, use_both=True)
    elif target_movie:
        # Movie only - use movie-based hybrid
        return hybrid_recommendation_enhanced(merged_df, target_movie, None, top_n)
    elif genre:
        # Genre only - use genre-based recommendations
        return hybrid_recommendation_enhanced(merged_df, None, genre, top_n)
    else:
        return None

@st.cache_data
def hybrid_recommendation_system(merged_df, target_movie=None, genre=None, top_n=5):
    """
    Alias for smart_hybrid_recommendation for backward compatibility
    """
    return smart_hybrid_recommendation(merged_df, target_movie, genre, top_n)
