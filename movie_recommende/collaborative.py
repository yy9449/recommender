import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st
from content_based import find_similar_titles
import os

@st.cache_data
def diagnose_data_linking(merged_df, user_ratings_df):
    """Diagnostic function to understand data linking between datasets"""
    if user_ratings_df is None:
        st.warning("No user ratings data available for diagnosis")
        return
    
    st.subheader("🔍 Data Linking Diagnosis")
    
    # Check columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Movies Dataset (merged_df):**")
        st.write(f"- Total movies: {len(merged_df)}")
        st.write(f"- Columns: {list(merged_df.columns)}")
        if 'Movie_ID' in merged_df.columns:
            movie_id_range = f"{merged_df['Movie_ID'].min()} - {merged_df['Movie_ID'].max()}"
            st.write(f"- Movie_ID range: {movie_id_range}")
            st.write(f"- Unique Movie_IDs: {merged_df['Movie_ID'].nunique()}")
        else:
            st.error("❌ No Movie_ID column found in movies dataset")
    
    with col2:
        st.write("**User Ratings Dataset:**")
        st.write(f"- Total ratings: {len(user_ratings_df)}")
        st.write(f"- Columns: {list(user_ratings_df.columns)}")
        st.write(f"- Unique users: {user_ratings_df['User_ID'].nunique()}")
        st.write(f"- Unique movies rated: {user_ratings_df['Movie_ID'].nunique()}")
        rating_range = f"{user_ratings_df['Movie_ID'].min()} - {user_ratings_df['Movie_ID'].max()}"
        st.write(f"- Movie_ID range: {rating_range}")
        st.write(f"- Rating range: {user_ratings_df['Rating'].min()} - {user_ratings_df['Rating'].max()}")
    
    # Check overlap
    if 'Movie_ID' in merged_df.columns:
        movies_set = set(merged_df['Movie_ID'])
        ratings_set = set(user_ratings_df['Movie_ID'])
        overlap = movies_set & ratings_set
        
        st.write("**🔗 Data Overlap Analysis:**")
        st.write(f"- Movies in both datasets: {len(overlap)} / {len(movies_set)} movies")
        st.write(f"- Coverage: {(len(overlap) / len(movies_set)) * 100:.1f}% of movie dataset")
        st.write(f"- User ratings for available movies: {len(user_ratings_df[user_ratings_df['Movie_ID'].isin(overlap)])}")
        
        if len(overlap) == 0:
            st.error("❌ No Movie_ID overlap found! Check that Movie_ID values match between files.")
        elif len(overlap) < len(movies_set) * 0.1:
            st.warning(f"⚠️ Very low overlap ({len(overlap)} movies). This may limit collaborative filtering effectiveness.")
        else:
            st.success(f"✅ Good overlap found: {len(overlap)} movies with user ratings available.")
        
        # Show sample Movie_IDs from each dataset
        with st.expander("🔍 Sample Movie_ID Values"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 10 Movie_IDs in movies dataset:**")
                st.write(sorted(list(movies_set))[:10])
            with col2:
                st.write("**First 10 Movie_IDs in user ratings:**")
                st.write(sorted(list(ratings_set))[:10])
            
            if overlap:
                st.write("**Sample overlapping Movie_IDs:**")
                st.write(sorted(list(overlap))[:10])

# Silent load user ratings function
@st.cache_data
def load_user_ratings():
    """Load real user ratings from CSV file or session state - silent version"""
    try:
        # First check if user uploaded the file (stored in session state)
        if 'user_ratings_df' in st.session_state:
            user_ratings_df = st.session_state['user_ratings_df']
            
            # Validate required columns
            required_cols = ['User_ID', 'Movie_ID', 'Rating']
            if all(col in user_ratings_df.columns for col in required_cols):
                return user_ratings_df
            else:
                return None
        
        # If not in session state, try to find it in local filesystem
        user_ratings_df = None
        for path in ["user_movie_rating.csv", "./user_movie_rating.csv", "data/user_movie_rating.csv", "../user_movie_rating.csv"]:
            if os.path.exists(path):
                user_ratings_df = pd.read_csv(path)
                break
        
        if user_ratings_df is not None:
            # Validate required columns
            required_cols = ['User_ID', 'Movie_ID', 'Rating']
            if all(col in user_ratings_df.columns for col in required_cols):
                return user_ratings_df
            else:
                return None
        else:
            return None
            
    except Exception as e:
        return None

@st.cache_data
def create_enhanced_item_similarity_matrix(merged_df):
    """Create item-item similarity matrix using multiple IMDb features"""
    # Extract features for similarity calculation
    features = []
    
    # Rating column
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    for _, movie in merged_df.iterrows():
        feature_vector = []
        
        # 1. Genre features (one-hot encoded)
        all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
                     'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 
                     'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
        
        movie_genres = []
        if pd.notna(movie[genre_col]):
            movie_genres = [g.strip() for g in movie[genre_col].split(',')]
        
        genre_features = [1 if genre in movie_genres else 0 for genre in all_genres]
        feature_vector.extend(genre_features)
        
        # 2. Certificate/Age rating similarity
        cert_mapping = {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5, 'Not Rated': 3, 'Approved': 3}
        cert = movie.get('Certificate', 'Not Rated')
        cert_score = cert_mapping.get(cert, 3) / 5.0  # Normalize to 0-1
        feature_vector.append(cert_score)
        
        # 3. IMDB Rating (normalized)
        rating = movie.get(rating_col, 7.0)
        if pd.notna(rating):
            rating_norm = rating / 10.0
        else:
            rating_norm = 0.7
        feature_vector.append(rating_norm)
        
        # 4. Vote count (log-normalized to handle wide range)
        votes = movie.get('No_of_Votes', 100000)
        if pd.notna(votes) and votes > 0:
            votes_norm = min(np.log10(votes) / 7.0, 1.0)  # Log normalize, cap at 1
        else:
            votes_norm = 0.5
        feature_vector.append(votes_norm)
        
        # 5. Runtime (normalized)
        runtime = movie.get('Runtime', 120)
        if pd.notna(runtime) and isinstance(runtime, (int, float)) and runtime > 0:
            runtime_norm = min(runtime / 200.0, 1.0)
        elif pd.notna(runtime) and isinstance(runtime, str):
            # Extract number from string like "120 min"
            import re
            runtime_match = re.search(r'(\d+)', str(runtime))
            if runtime_match:
                runtime_norm = min(int(runtime_match.group(1)) / 200.0, 1.0)
            else:
                runtime_norm = 0.6
        else:
            runtime_norm = 0.6
        feature_vector.append(runtime_norm)
        
        # 6. Year (normalized)
        year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
        year = movie.get(year_col, 2000)
        if pd.notna(year) and isinstance(year, (int, float)):
            year_norm = (year - 1920) / (2025 - 1920)
        else:
            year_norm = 0.5
        feature_vector.append(year_norm)
        
        # 7. Director similarity (hash-based feature)
        director = movie.get('Director', 'Unknown')
        director_hash = hash(str(director)) % 1000 / 1000.0  # Normalize hash
        feature_vector.append(director_hash)
        
        features.append(feature_vector)
    
    features_array = np.array(features)
    
    # Standardize features to give equal weight
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Calculate item-item similarity matrix
    similarity_matrix = cosine_similarity(features_scaled)
    
    return similarity_matrix

@st.cache_data
def create_synthetic_user_profiles_enhanced(merged_df, n_users=50):
    """Create more realistic synthetic user profiles based on actual movie characteristics"""
    np.random.seed(42)
    
    # Define more nuanced user archetypes based on real viewing patterns
    user_archetypes = {
        # Genre-based preferences
        'action_blockbuster': {'genres': ['Action', 'Adventure', 'Thriller'], 'min_rating': 6.5, 'min_votes': 100000, 'year_pref': 'recent'},
        'indie_drama': {'genres': ['Drama', 'Biography', 'Romance'], 'min_rating': 7.5, 'min_votes': 50000, 'year_pref': 'any'},
        'comedy_casual': {'genres': ['Comedy', 'Family', 'Romance'], 'min_rating': 6.0, 'min_votes': 75000, 'year_pref': 'recent'},
        'horror_thriller': {'genres': ['Horror', 'Thriller', 'Mystery'], 'min_rating': 6.8, 'min_votes': 80000, 'year_pref': 'any'},
        'sci_fi_fantasy': {'genres': ['Sci-Fi', 'Fantasy', 'Adventure'], 'min_rating': 7.0, 'min_votes': 120000, 'year_pref': 'any'},
        'classic_cinema': {'genres': ['Drama', 'Biography', 'History'], 'min_rating': 8.0, 'min_votes': 200000, 'year_pref': 'classic'},
        'family_friendly': {'genres': ['Family', 'Animation', 'Comedy'], 'min_rating': 6.5, 'min_votes': 150000, 'year_pref': 'recent'},
        'crime_thriller': {'genres': ['Crime', 'Thriller', 'Mystery'], 'min_rating': 7.2, 'min_votes': 100000, 'year_pref': 'any'},
        
        # Quality-based preferences
        'quality_seeker': {'genres': [], 'min_rating': 8.5, 'min_votes': 500000, 'year_pref': 'any'},
        'mainstream_viewer': {'genres': ['Action', 'Comedy', 'Drama'], 'min_rating': 6.0, 'min_votes': 200000, 'year_pref': 'recent'},
        'cult_films': {'genres': ['Horror', 'Sci-Fi', 'Thriller'], 'min_rating': 7.5, 'min_votes': 50000, 'year_pref': 'classic'},
        'award_season': {'genres': ['Drama', 'Biography', 'Romance'], 'min_rating': 7.8, 'min_votes': 300000, 'year_pref': 'recent'},
    }
    
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
    
    user_movie_matrix = []
    user_names = []
    
    for i in range(n_users):
        # Select archetype with some randomness
        archetype_name = np.random.choice(list(user_archetypes.keys()))
        archetype = user_archetypes[archetype_name]
        
        user_ratings = []
        user_name = f"{archetype_name}_{i % 5}"
        user_names.append(user_name)
        
        for _, movie in merged_df.iterrows():
            rating = 0
            
            # Check if movie matches user preferences
            movie_rating = movie.get(rating_col, 6.0)
            movie_votes = movie.get('No_of_Votes', 100000)
            movie_year = movie.get(year_col, 2000)
            movie_genres = []
            
            if pd.notna(movie[genre_col]):
                movie_genres = [g.strip() for g in movie[genre_col].split(',')]
            
            # Rating threshold check
            if pd.notna(movie_rating) and movie_rating >= archetype['min_rating']:
                
                # Genre preference check
                genre_match = False
                if not archetype['genres']:  # Quality seeker - no genre preference
                    genre_match = True
                else:
                    genre_match = any(genre in movie_genres for genre in archetype['genres'])
                
                if genre_match:
                    # Year preference check - FIXED VERSION
                    year_match = True
                    if archetype['year_pref'] == 'recent' and pd.notna(movie_year):
                        # Ensure movie_year is numeric
                        try:
                            year_value = int(movie_year) if isinstance(movie_year, (int, float, str)) and str(movie_year).isdigit() else 2000
                            year_match = year_value >= 2000
                        except (ValueError, TypeError):
                            year_match = True  # Default to True if year can't be parsed
                    elif archetype['year_pref'] == 'classic' and pd.notna(movie_year):
                        try:
                            year_value = int(movie_year) if isinstance(movie_year, (int, float, str)) and str(movie_year).isdigit() else 2000
                            year_match = year_value < 2000
                        except (ValueError, TypeError):
                            year_match = True  # Default to True if year can't be parsed
                    
                    if year_match:
                        # Vote popularity check
                        if pd.notna(movie_votes) and movie_votes >= archetype['min_votes']:
                            # Generate rating based on how well it matches preferences
                            base_rating = movie_rating / 10.0 * 5  # Convert to 1-5 scale
                            
                            # Add some preference-based variation
                            preference_bonus = 0
                            if len(set(archetype['genres']) & set(movie_genres)) > 1:
                                preference_bonus = 0.5  # Multiple genre match
                            
                            final_rating = min(5.0, base_rating + preference_bonus + np.random.normal(0, 0.3))
                            
                            # Only include ratings above 2.5 (simulate that users don't rate movies they really dislike)
                            if final_rating >= 2.5:
                                rating = max(1.0, final_rating)
                            
                            # Add some sparsity - not every user rates every qualifying movie
                            if np.random.random() < 0.4:  # 40% chance of rating
                                rating = rating
                            else:
                                rating = 0
            
            user_ratings.append(rating)
        
        user_movie_matrix.append(user_ratings)
    
    return np.array(user_movie_matrix), user_names

@st.cache_data
def item_based_collaborative_filtering(merged_df, target_movie, top_n=5):
    """Item-based collaborative filtering using movie similarity"""
    similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
    if not similar_titles:
        return None
    
    target_title = similar_titles[0]
    target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
    movie_index = merged_df.index.get_loc(target_idx)
    
    # Get item-item similarity matrix
    similarity_matrix = create_enhanced_item_similarity_matrix(merged_df)
    
    # Get similarities for target movie
    target_similarities = similarity_matrix[movie_index]
    
    # Get top similar movies (excluding the target movie itself)
    similar_indices = np.argsort(target_similarities)[::-1][1:top_n*3]  # Get more to filter by quality
    
    # Filter by quality and create results
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    candidates = []
    for idx in similar_indices:
        movie_data = merged_df.iloc[idx]
        similarity_score = target_similarities[idx]
        movie_rating = movie_data.get(rating_col, 6.0)
        
        # Quality filter - prefer higher rated movies
        if pd.notna(movie_rating) and movie_rating >= 6.5:
            candidates.append({
                'title': movie_data['Series_Title'],
                'similarity': similarity_score,
                'rating': movie_rating,
                'combined_score': similarity_score * 0.7 + (movie_rating / 10.0) * 0.3
            })
    
    # Sort by combined score and take top N
    candidates.sort(key=lambda x: x['combined_score'], reverse=True)
    top_candidates = candidates[:top_n]
    
    if not top_candidates:
        return None
    
    # Create result dataframe
    result_titles = [c['title'] for c in top_candidates]
    result_df = merged_df[merged_df['Series_Title'].isin(result_titles)]
    
    # Preserve order
    title_order = {title: i for i, title in enumerate(result_titles)}
    result_df = result_df.copy()
    result_df['order'] = result_df['Series_Title'].map(title_order)
    result_df = result_df.sort_values('order').drop('order', axis=1)
    
    return result_df[['Series_Title', genre_col, rating_col]]

@st.cache_data
def user_based_collaborative_filtering_enhanced(merged_df, target_movie, user_ratings_df=None, top_n=5):
    """Enhanced user-based collaborative filtering - silent version"""
    similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
    if not similar_titles:
        return None
    
    target_title = similar_titles[0]
    target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
    
    # Use real user data if available, otherwise enhanced synthetic
    if user_ratings_df is not None:
        rating_matrix, user_names = create_user_item_matrix_from_real_data(merged_df, user_ratings_df)
    else:
        rating_matrix, user_names = create_synthetic_user_profiles_enhanced(merged_df, n_users=75)
    
    if rating_matrix is None:
        return None
    
    # KNN for finding similar users
    n_neighbors = min(15, len(rating_matrix) - 1)
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    knn_model.fit(rating_matrix)
    
    target_movie_idx = merged_df.index.get_loc(target_idx)
    target_ratings = rating_matrix[:, target_movie_idx]
    
    # Find users who rated this movie highly (rating > 3.0)
    active_users = np.where(target_ratings > 3.0)[0]
    
    if len(active_users) == 0:
        return item_based_collaborative_filtering(merged_df, target_movie, top_n)
    
    # Aggregate recommendations from similar users
    movie_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    
    for user_idx in active_users:
        user_rating = target_ratings[user_idx]
        
        # Find similar users
        distances, neighbor_indices = knn_model.kneighbors([rating_matrix[user_idx]])
        
        for i, neighbor_idx in enumerate(neighbor_indices[0]):
            if neighbor_idx != user_idx:
                similarity = max(0.1, 1 - distances[0][i])
                neighbor_ratings = rating_matrix[neighbor_idx]
                
                # Get recommendations from similar user
                for movie_idx, rating in enumerate(neighbor_ratings):
                    if rating > 3.5 and movie_idx != target_movie_idx:
                        movie_title = merged_df.iloc[movie_idx]['Series_Title']
                        
                        if movie_title not in movie_scores:
                            movie_scores[movie_title] = {'total_score': 0, 'count': 0}
                        
                        # Weight by user similarity and original user's rating
                        weight = similarity * (user_rating / 5.0)
                        score = rating * weight
                        
                        movie_scores[movie_title]['total_score'] += score
                        movie_scores[movie_title]['count'] += weight
    
    # Calculate final scores
    final_scores = {}
    for title, data in movie_scores.items():
        if data['count'] > 0:
            avg_score = data['total_score'] / data['count']
            # Boost by IMDB rating
            movie_info = merged_df[merged_df['Series_Title'] == title]
            if not movie_info.empty:
                imdb_rating = movie_info.iloc[0][rating_col]
                if pd.notna(imdb_rating):
                    quality_boost = (imdb_rating / 10.0) * 0.3 + 0.7
                    final_scores[title] = avg_score * quality_boost
    
    if not final_scores:
        return item_based_collaborative_filtering(merged_df, target_movie, top_n)
    
    # Get top recommendations
    top_titles = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result_titles = [title for title, _ in top_titles]
    
    result_df = merged_df[merged_df['Series_Title'].isin(result_titles)]
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    # Preserve order
    title_order = {title: i for i, title in enumerate(result_titles)}
    result_df = result_df.copy()
    result_df['order'] = result_df['Series_Title'].map(title_order)
    result_df = result_df.sort_values('order').drop('order', axis=1)
    
    return result_df[['Series_Title', genre_col, rating_col]]

@st.cache_data
def create_user_item_matrix_from_real_data(merged_df, user_ratings_df):
    """Create user-item matrix from real user rating data - silent version"""
    if user_ratings_df is None:
        return create_synthetic_user_profiles_enhanced(merged_df)
    
    try:
        # Check if merged_df has Movie_ID column (from movies.csv)
        if 'Movie_ID' not in merged_df.columns:
            return create_synthetic_user_profiles_enhanced(merged_df)
        
        # Create mapping from Movie_ID to dataset row indices
        movie_id_to_index = dict(zip(merged_df['Movie_ID'], merged_df.index))
        
        # Filter user ratings to only include movies that exist in our dataset
        valid_movie_ids = set(movie_id_to_index.keys())
        user_movie_ids = set(user_ratings_df['Movie_ID'].unique())
        
        # Find intersection
        common_movie_ids = valid_movie_ids & user_movie_ids
        if not common_movie_ids:
            return create_synthetic_user_profiles_enhanced(merged_df)
        
        # Filter ratings to common movies only
        filtered_ratings = user_ratings_df[user_ratings_df['Movie_ID'].isin(common_movie_ids)].copy()
        
        # Map Movie_ID to dataset row indices
        filtered_ratings['dataset_index'] = filtered_ratings['Movie_ID'].map(movie_id_to_index)
        
        # Remove any rows where mapping failed
        filtered_ratings = filtered_ratings.dropna(subset=['dataset_index'])
        filtered_ratings['dataset_index'] = filtered_ratings['dataset_index'].astype(int)
        
        # Create user-item matrix with dataset indices as columns
        user_movie_matrix = filtered_ratings.pivot_table(
            index='User_ID', 
            columns='dataset_index', 
            values='Rating',
            fill_value=0
        )
        
        # Ensure matrix covers all movies in dataset (fill missing movies with 0)
        all_dataset_indices = list(range(len(merged_df)))
        missing_indices = set(all_dataset_indices) - set(user_movie_matrix.columns)
        
        for idx in missing_indices:
            user_movie_matrix[idx] = 0
        
        # Reorder columns to match dataset order
        user_movie_matrix = user_movie_matrix.reindex(columns=all_dataset_indices, fill_value=0)
        
        # Convert to numpy array and create user names
        rating_matrix = user_movie_matrix.values
        user_names = [f"User_{uid}" for uid in user_movie_matrix.index]
        
        return rating_matrix, user_names
        
    except Exception as e:
        return create_synthetic_user_profiles_enhanced(merged_df)

# Main interface function
@st.cache_data
def collaborative_filtering_enhanced(merged_df, target_movie, top_n=5):
    """Enhanced collaborative filtering that chooses best approach based on available data - silent version"""
    if not target_movie:
        return None
    
    user_ratings_df = load_user_ratings()
    
    if user_ratings_df is not None:
        return user_based_collaborative_filtering_enhanced(merged_df, target_movie, user_ratings_df, top_n)
    else:
        # Try user-based first (with enhanced synthetic data)
        user_result = user_based_collaborative_filtering_enhanced(merged_df, target_movie, None, top_n)
        
        # If user-based fails or gives poor results, fall back to item-based
        if user_result is None or len(user_result) < top_n // 2:
            item_result = item_based_collaborative_filtering(merged_df, target_movie, top_n)
            
            # Combine results if both exist
            if user_result is not None and item_result is not None:
                # Merge and deduplicate
                combined = pd.concat([user_result, item_result]).drop_duplicates(subset='Series_Title')
                return combined.head(top_n)
            elif item_result is not None:
                return item_result
            else:
                return user_result
        
        return user_result
