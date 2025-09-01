import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from content_based import find_similar_titles
import os

@st.cache_data
def load_user_ratings():
    """Load real user ratings from CSV file"""
    try:
        # Try different possible file paths for user_movie_rating.csv
        user_ratings_df = None
        for path in ["user_movie_rating.csv", "./user_movie_rating.csv", "data/user_movie_rating.csv", "../user_movie_rating.csv"]:
            if os.path.exists(path):
                user_ratings_df = pd.read_csv(path)
                st.success(f"✅ Found user_movie_rating.csv at: {path}")
                break
        
        if user_ratings_df is not None:
            # Validate required columns
            required_cols = ['User_ID', 'Movie_ID', 'Rating']
            if all(col in user_ratings_df.columns for col in required_cols):
                return user_ratings_df
            else:
                st.warning(f"⚠️ user_movie_rating.csv missing required columns: {required_cols}")
                return None
        else:
            st.info("📋 user_movie_rating.csv not found, using synthetic data")
            return None
            
    except Exception as e:
        st.error(f"Error loading user ratings: {str(e)}")
        return None

@st.cache_data
def create_user_item_matrix_from_real_data(merged_df, user_ratings_df):
    """Create user-item matrix from real user rating data"""
    if user_ratings_df is None:
        return create_user_item_matrix_synthetic(merged_df)
    
    try:
        # Create a mapping between Movie_ID and our merged_df index
        # Assuming Movie_ID corresponds to the index or a specific column in merged_df
        movie_id_to_title = {}
        
        # Try to map Movie_ID to Series_Title
        if 'Movie_ID' in merged_df.columns:
            movie_id_to_title = dict(zip(merged_df['Movie_ID'], merged_df['Series_Title']))
        else:
            # If no Movie_ID column, use index as Movie_ID
            movie_id_to_title = dict(zip(range(len(merged_df)), merged_df['Series_Title']))
        
        # Filter user ratings to only include movies in our dataset
        valid_movie_ids = set(movie_id_to_title.keys())
        filtered_ratings = user_ratings_df[user_ratings_df['Movie_ID'].isin(valid_movie_ids)]
        
        if filtered_ratings.empty:
            st.warning("⚠️ No matching movies found between user ratings and movie dataset")
            return create_user_item_matrix_synthetic(merged_df)
        
        # Create user-item matrix
        user_movie_matrix = filtered_ratings.pivot(index='User_ID', columns='Movie_ID', values='Rating').fillna(0)
        
        # Ensure the matrix covers all movies in our dataset
        all_movie_ids = list(movie_id_to_title.keys())
        missing_movies = set(all_movie_ids) - set(user_movie_matrix.columns)
        
        for movie_id in missing_movies:
            user_movie_matrix[movie_id] = 0
        
        # Reorder columns to match merged_df order
        user_movie_matrix = user_movie_matrix.reindex(columns=all_movie_ids, fill_value=0)
        
        rating_matrix = user_movie_matrix.values
        user_names = [f"User_{uid}" for uid in user_movie_matrix.index]
        
        st.info(f"📊 Real user data loaded: {len(user_names)} users, {len(all_movie_ids)} movies")
        
        return rating_matrix, user_names
        
    except Exception as e:
        st.error(f"Error processing user ratings: {str(e)}")
        return create_user_item_matrix_synthetic(merged_df)

@st.cache_data
def create_user_item_matrix_synthetic(merged_df):
    """Create a synthetic user-item matrix based on movie characteristics (fallback)"""
    np.random.seed(42)
    
    user_types = {
        'action_lover': {'Action': 5, 'Adventure': 4, 'Thriller': 4, 'Drama': 2, 'Comedy': 2, 'Romance': 1},
        'drama_fan': {'Drama': 5, 'Romance': 4, 'Biography': 4, 'Action': 2, 'Comedy': 3, 'Thriller': 2},
        'comedy_fan': {'Comedy': 5, 'Romance': 4, 'Family': 4, 'Action': 2, 'Drama': 3, 'Horror': 1},
        'thriller_fan': {'Thriller': 5, 'Mystery': 4, 'Crime': 4, 'Horror': 3, 'Action': 4, 'Comedy': 2},
        'classic_lover': {'Drama': 4, 'Romance': 4, 'Biography': 5, 'History': 5, 'War': 4, 'Comedy': 3},
        'sci_fi_fan': {'Sci-Fi': 5, 'Fantasy': 4, 'Action': 4, 'Adventure': 3, 'Thriller': 3, 'Drama': 2},
        'horror_fan': {'Horror': 5, 'Thriller': 4, 'Mystery': 4, 'Sci-Fi': 3, 'Action': 3, 'Comedy': 1},
        'family_viewer': {'Family': 5, 'Animation': 5, 'Comedy': 4, 'Adventure': 4, 'Fantasy': 3, 'Drama': 2}
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
    
    st.info("📊 Using synthetic user data for recommendations")
    
    return rating_matrix, user_names

@st.cache_data
def collaborative_filtering_knn(merged_df, target_movie=None, user_ratings_df=None, top_n=5, n_neighbors=10):
    """KNN-based collaborative filtering using real user data"""
    if not target_movie:
        return None
    
    similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
    if not similar_titles:
        return None
    
    target_title = similar_titles[0]
    target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
    
    # Use real user data if available, otherwise fall back to synthetic
    rating_matrix, user_names = create_user_item_matrix_from_real_data(merged_df, user_ratings_df)
    
    # Ensure we have enough neighbors
    actual_neighbors = min(n_neighbors, len(rating_matrix) - 1)
    if actual_neighbors < 1:
        return collaborative_filtering_enhanced_fallback(merged_df, target_movie, top_n)
    
    # Use KNN with cosine distance for finding similar users
    knn_model = NearestNeighbors(n_neighbors=actual_neighbors, metric='cosine', algorithm='brute')
    knn_model.fit(rating_matrix)
    
    target_movie_idx = merged_df.index.get_loc(target_idx)
    target_ratings = rating_matrix[:, target_movie_idx]
    
    # Find users who rated this movie (rating > 0)
    active_users = np.where(target_ratings > 0)[0]
    
    if len(active_users) == 0:
        return collaborative_filtering_enhanced_fallback(merged_df, target_movie, top_n)
    
    # Get recommendations from similar users
    movie_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    
    for user_idx in active_users:
        # Find similar users using KNN
        distances, neighbor_indices = knn_model.kneighbors([rating_matrix[user_idx]])
        
        user_weight = target_ratings[user_idx] / 5.0  # Normalize user rating
        
        for i, neighbor_idx in enumerate(neighbor_indices[0]):
            if neighbor_idx != user_idx:  # Skip self
                similarity = max(0.1, 1 - distances[0][i])  # Convert distance to similarity
                neighbor_ratings = rating_matrix[neighbor_idx]
                
                # Get recommendations from this similar user
                for movie_idx, rating in enumerate(neighbor_ratings):
                    if rating > 3.5 and movie_idx != target_movie_idx:  # Higher threshold for better quality
                        movie_title = merged_df.iloc[movie_idx]['Series_Title']
                        if movie_title not in movie_scores:
                            movie_scores[movie_title] = 0
                        
                        # Calculate score: user_rating * similarity * user_weight * imdb_factor
                        imdb_rating = merged_df.iloc[movie_idx][rating_col]
                        imdb_rating = imdb_rating if pd.notna(imdb_rating) else 7.0
                        imdb_factor = (imdb_rating / 10.0) * 0.5 + 0.5  # Scale IMDB rating influence
                        
                        score = rating * similarity * user_weight * imdb_factor
                        movie_scores[movie_title] += score
    
    if not movie_scores:
        return collaborative_filtering_enhanced_fallback(merged_df, target_movie, top_n)
    
    # Sort by combined score (high to low)
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]
    
    # Get result dataframe and sort by IMDB rating
    rec_titles = [rec[0] for rec in recommendations[:top_n * 2]]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    result_df = result_df.sort_values(by=rating_col, ascending=False)
    
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

@st.cache_data
def collaborative_filtering_enhanced_fallback(merged_df, target_movie, top_n=5):
    """Fallback collaborative filtering (original implementation)"""
    if not target_movie:
        return None
    
    similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
    if not similar_titles:
        return None
    
    target_title = similar_titles[0]
    target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
    
    rating_matrix, user_names = create_user_item_matrix_synthetic(merged_df)
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
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    
    for user_idx, user_weight in top_users:
        user_ratings = rating_matrix[user_idx]
        for movie_idx, rating in enumerate(user_ratings):
            if rating > 3 and movie_idx != target_movie_idx:
                movie_title = merged_df.iloc[movie_idx]['Series_Title']
                if movie_title not in movie_scores:
                    movie_scores[movie_title] = 0
                # Include IMDB rating in scoring
                imdb_rating = merged_df.iloc[movie_idx][rating_col] if pd.notna(merged_df.iloc[movie_idx][rating_col]) else 7.0
                combined_score = (rating * user_weight * 0.7) + (imdb_rating * 0.3)
                movie_scores[movie_title] += combined_score
    
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not recommendations:
        return None
    
    rec_titles = [rec[0] for rec in recommendations]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    
    # Sort by IMDB rating (high to low)
    result_df = result_df.sort_values(by=rating_col, ascending=False)
    
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

# Main functions for the simplified interface
@st.cache_data
def collaborative_filtering_enhanced(merged_df, target_movie, top_n=5):
    """Main collaborative filtering function using KNN"""
    user_ratings_df = load_user_ratings()
    return collaborative_filtering_knn(merged_df, target_movie, user_ratings_df, top_n, n_neighbors=10)
