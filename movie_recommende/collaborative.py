import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from content_based import find_similar_titles
import os

@st.cache_data
def load_user_ratings():
    """Load real user ratings from CSV file or session state"""
    try:
        # First check if user uploaded the file (stored in session state)
        if 'user_ratings_df' in st.session_state:
            user_ratings_df = st.session_state['user_ratings_df']
            st.success("✅ Using uploaded user_movie_rating.csv from session")
            
            # Validate required columns
            required_cols = ['User_ID', 'Movie_ID', 'Rating']
            if all(col in user_ratings_df.columns for col in required_cols):
                return user_ratings_df
            else:
                st.warning(f"⚠️ Uploaded user_movie_rating.csv missing required columns: {required_cols}")
                return None
        
        # If not in session state, try to find it in local filesystem
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
            st.info("📋 user_movie_rating.csv not found in local directory, using synthetic data")
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
        st.info(f"📊 Processing user ratings: {len(user_ratings_df)} ratings from {user_ratings_df['User_ID'].nunique()} users")
        
        # Create a mapping between Movie_ID and our merged_df
        movie_id_to_index = {}
        
        # Try different approaches to map Movie_ID to our dataset
        if 'Movie_ID' in merged_df.columns:
            # Direct Movie_ID mapping
            movie_id_to_index = dict(zip(merged_df['Movie_ID'], merged_df.index))
        else:
            # Use index as Movie_ID (common approach)
            movie_id_to_index = dict(zip(range(len(merged_df)), merged_df.index))
        
        # Filter user ratings to only include movies in our dataset
        valid_movie_ids = set(movie_id_to_index.keys())
        filtered_ratings = user_ratings_df[user_ratings_df['Movie_ID'].isin(valid_movie_ids)].copy()
        
        if filtered_ratings.empty:
            st.warning("⚠️ No matching movies found between user ratings and movie dataset")
            st.info("💡 Make sure Movie_ID in user_movie_rating.csv corresponds to movie indices in your dataset")
            return create_user_item_matrix_synthetic(merged_df)
        
        st.success(f"✅ Successfully matched {len(filtered_ratings)} ratings with movies in dataset")
        
        # Map Movie_ID to our dataset indices
        filtered_ratings['dataset_index'] = filtered_ratings['Movie_ID'].map(movie_id_to_index)
        
        # Create user-item matrix using dataset indices
        user_movie_matrix = filtered_ratings.pivot(
            index='User_ID', 
            columns='dataset_index', 
            values='Rating'
        ).fillna(0)
        
        # Ensure the matrix covers all movies in our dataset
        all_indices = list(range(len(merged_df)))
        missing_movies = set(all_indices) - set(user_movie_matrix.columns)
        
        for movie_idx in missing_movies:
            user_movie_matrix[movie_idx] = 0
        
        # Reorder columns to match merged_df order
        user_movie_matrix = user_movie_matrix.reindex(columns=all_indices, fill_value=0)
        
        rating_matrix = user_movie_matrix.values
        user_names = [f"User_{uid}" for uid in user_movie_matrix.index]
        
        st.success(f"🎯 Created user-item matrix: {len(user_names)} users × {len(all_indices)} movies")
        
        # Show some statistics
        non_zero_ratings = np.count_nonzero(rating_matrix)
        total_possible = rating_matrix.size
        sparsity = (1 - (non_zero_ratings / total_possible)) * 100
        
        st.info(f"📈 Matrix Statistics: {non_zero_ratings:,} ratings ({sparsity:.1f}% sparse)")
        
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
    
    st.info("📊 Using synthetic user data for collaborative filtering")
    
    return rating_matrix, user_names

@st.cache_data
def collaborative_filtering_knn(merged_df, target_movie=None, user_ratings_df=None, top_n=5, n_neighbors=10):
    """KNN-based collaborative filtering using real or synthetic user data"""
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
        st.warning("⚠️ Not enough user data for collaborative filtering")
        return None
    
    # Use KNN with cosine distance for finding similar users
    knn_model = NearestNeighbors(n_neighbors=actual_neighbors, metric='cosine', algorithm='brute')
    knn_model.fit(rating_matrix)
    
    target_movie_idx = merged_df.index.get_loc(target_idx)
    target_ratings = rating_matrix[:, target_movie_idx]
    
    # Find users who rated this movie (rating > 0)
    active_users = np.where(target_ratings > 0)[0]
    
    if len(active_users) == 0:
        st.warning(f"⚠️ No users found who rated '{target_title}'")
        return None
    
    st.info(f"🎯 Found {len(active_users)} users who rated '{target_title}'")
    
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
        st.warning("⚠️ No suitable recommendations found from collaborative filtering")
        return None
    
    # Sort by combined score (high to low)
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]
    
    # Get result dataframe and sort by IMDB rating
    rec_titles = [rec[0] for rec in recommendations[:top_n * 2]]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    result_df = result_df.sort_values(by=rating_col, ascending=False)
    
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

# Main function for the interface
@st.cache_data
def collaborative_filtering_enhanced(merged_df, target_movie, top_n=5):
    """Main collaborative filtering function using KNN"""
    user_ratings_df = load_user_ratings()
    return collaborative_filtering_knn(merged_df, target_movie, user_ratings_df, top_n, n_neighbors=10)
