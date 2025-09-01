import pandas as pd
import numpy as np
import streamlit as st
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced, load_user_ratings

@st.cache_data
def content_heavy_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=5):
    """
    Content-Heavy Hybrid: 60% Content-Based + 40% Collaborative
    Better for systems with limited or synthetic user data
    """
    if not target_movie and not genre:
        return None
    
    # Load user ratings to determine data quality
    user_ratings_df = load_user_ratings()
    has_real_user_data = user_ratings_df is not None and 'user_ratings_df' in st.session_state
    
    # Initialize results storage
    movie_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    # Adjust weights based on data quality
    if has_real_user_data:
        content_weight = 0.6
        collaborative_weight = 0.4
        st.info("🎯 Using Content-Heavy Hybrid: 60% Content + 40% Collaborative (Real User Data)")
    else:
        content_weight = 0.75
        collaborative_weight = 0.25
        st.info("🎯 Using Content-Heavy Hybrid: 75% Content + 25% Collaborative (Synthetic Data)")
    
    # Scenario 1: Both movie and genre provided
    if target_movie and genre:
        st.info("🔄 Using enhanced content-heavy approach with movie and genre preferences")
        
        # Content-based on movie (35%)
        content_recs_movie = content_based_filtering_enhanced(merged_df, target_movie, None, top_n * 2)
        if content_recs_movie is not None:
            for idx, row in content_recs_movie.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * (content_weight * 0.6)  # 60% of content weight
                movie_scores[title] = movie_scores.get(title, 0) + score
        
        # Content-based on genre (25%)
        content_recs_genre = content_based_filtering_enhanced(merged_df, None, genre, top_n * 2)
        if content_recs_genre is not None:
            for idx, row in content_recs_genre.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * (content_weight * 0.4)  # 40% of content weight
                movie_scores[title] = movie_scores.get(title, 0) + score
        
        # Collaborative filtering (40% or 25% based on data quality)
        if target_movie:  # Collaborative needs a movie reference
            collab_recs = collaborative_filtering_enhanced(merged_df, target_movie, top_n * 2)
            if collab_recs is not None:
                for idx, row in collab_recs.iterrows():
                    title = row['Series_Title']
                    score = row[rating_col] * collaborative_weight
                    movie_scores[title] = movie_scores.get(title, 0) + score
        
        # Bonus for movies matching the selected genre
        for title, base_score in movie_scores.items():
            movie_info = merged_df[merged_df['Series_Title'] == title]
            if not movie_info.empty:
                movie_genres = []
                if pd.notna(movie_info.iloc[0][genre_col]):
                    movie_genres = [g.strip().lower() for g in movie_info.iloc[0][genre_col].split(',')]
                
                if genre.lower() in movie_genres:
                    movie_scores[title] *= 1.2  # 20% bonus for genre match
    
    # Scenario 2: Movie only
    elif target_movie and not genre:
        st.info("🎬 Using content-heavy movie-based recommendations")
        
        # Content-based filtering (60% or 75%)
        content_recs = content_based_filtering_enhanced(merged_df, target_movie, None, top_n * 2)
        if content_recs is not None:
            for idx, row in content_recs.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * content_weight
                movie_scores[title] = movie_scores.get(title, 0) + score
        
        # Collaborative filtering (40% or 25%)
        collab_recs = collaborative_filtering_enhanced(merged_df, target_movie, top_n * 2)
        if collab_recs is not None:
            for idx, row in collab_recs.iterrows():
                title = row['Series_Title']
                score = row[rating_col] * collaborative_weight
                movie_scores[title] = movie_scores.get(title, 0) + score
    
    # Scenario 3: Genre only
    elif genre and not target_movie:
        st.info("🎭 Using content-based genre recommendations")
        
        # Pure content-based with enhanced genre weighting
        content_recs = content_based_filtering_enhanced(merged_df, None, genre, top_n * 2)
        if content_recs is not None:
            for idx, row in content_recs.iterrows():
                title = row['Series_Title']
                movie_info = merged_df[merged_df['Series_Title'] == title].iloc[0]
                
                # Calculate genre match strength
                movie_genres = []
                if pd.notna(movie_info[genre_col]):
                    movie_genres = [g.strip().lower() for g in movie_info[genre_col].split(',')]
                
                base_score = row[rating_col]
                
                # Enhanced scoring for genre matches
                if genre.lower() in movie_genres:
                    genre_bonus = 1.5  # 50% bonus for direct match
                else:
                    genre_bonus = 1.0
                
                # Additional quality boost
                quality_boost = (base_score / 10.0) * 0.3 + 0.7
                
                final_score = base_score * genre_bonus * quality_boost
                movie_scores[title] = final_score
    
    # Apply consensus bonuses for movies appearing in multiple recommendation sources
    if target_movie and (content_recs_movie is not None or collab_recs is not None):
        # Track which movies appear in multiple sources
        content_titles = set()
        collab_titles = set()
        
        if 'content_recs_movie' in locals() and content_recs_movie is not None:
            content_titles = set(content_recs_movie['Series_Title'])
        if 'content_recs' in locals() and content_recs is not None:
            content_titles = set(content_recs['Series_Title'])
        if 'collab_recs' in locals() and collab_recs is not None:
            collab_titles = set(collab_recs['Series_Title'])
        
        # Apply consensus bonus
        consensus_movies = content_titles.intersection(collab_titles)
        for title in consensus_movies:
            if title in movie_scores:
                movie_scores[title] *= 1.3  # 30% bonus for consensus
    
    # Sort by combined score
    if not movie_scores:
        return None
    
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get diverse recommendations (avoid too many from same director/year)
    diverse_recommendations = []
    used_directors = set()
    used_years = set()
    
    for movie_title, score in sorted_movies:
        if len(diverse_recommendations) >= top_n:
            break
            
        movie_info = merged_df[merged_df['Series_Title'] == movie_title]
        if movie_info.empty:
            continue
        
        director = movie_info.iloc[0].get('Director', 'Unknown')
        year = movie_info.iloc[0].get('Released_Year', movie_info.iloc[0].get('Year', 'Unknown'))
        
        # Diversity check - limit same director/year
        director_count = sum(1 for d in used_directors if d == director)
        year_count = sum(1 for y in used_years if y == year)
        
        # Allow some repetition but prioritize diversity
        if (director_count < 2 and year_count < 3) or len(diverse_recommendations) < top_n // 2:
            diverse_recommendations.append((movie_title, score))
            used_directors.add(director)
            used_years.add(year)
    
    # Fill remaining slots if needed
    if len(diverse_recommendations) < top_n:
        remaining_slots = top_n - len(diverse_recommendations)
        existing_titles = {title for title, _ in diverse_recommendations}
        
        for movie_title, score in sorted_movies:
            if movie_title not in existing_titles and remaining_slots > 0:
                diverse_recommendations.append((movie_title, score))
                remaining_slots -= 1
    
    # Create final result dataframe
    final_titles = [title for title, _ in diverse_recommendations[:top_n]]
    result_df = merged_df[merged_df['Series_Title'].isin(final_titles)]
    
    # Preserve recommendation order but sort by IMDB rating as secondary
    title_to_score = dict(diverse_recommendations[:top_n])
    result_df = result_df.copy()
    result_df['rec_score'] = result_df['Series_Title'].map(title_to_score)
    result_df = result_df.sort_values(['rec_score', rating_col], ascending=[False, False])
    result_df = result_df.drop('rec_score', axis=1)
    
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=5):
    """
    Updated smart hybrid using content-heavy approach
    """
    return content_heavy_hybrid_recommendation(merged_df, target_movie, genre, top_n)

# Backward compatibility
@st.cache_data
def hybrid_recommendation_system(merged_df, target_movie=None, genre=None, top_n=5):
    """
    Alias for content_heavy_hybrid_recommendation
    """
    return content_heavy_hybrid_recommendation(merged_df, target_movie, genre, top_n)
