import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from content_based import content_based_filtering_enhanced, create_content_features
from collaborative import collaborative_filtering_enhanced, load_user_ratings
import warnings
warnings.filterwarnings('ignore')

class SVDHybridRecommender:
    def __init__(self, merged_df):
        self.merged_df = merged_df
        self.rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        self.genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        self.scaler = StandardScaler()
        self.svd_model = None
        self.hybrid_features = None
        
    def create_hybrid_feature_matrix(self):
        """Create combined feature matrix for SVD decomposition"""
        
        # Get content features (TF-IDF from overview, genre, director, stars)
        try:
            content_features = create_content_features(self.merged_df)
            if hasattr(content_features, 'toarray'):
                content_features = content_features.toarray()
        except:
            # Fallback: create basic content features
            content_features = self._create_basic_content_features()
        
        # Get collaborative features (user-item interactions)
        collab_features = self._create_collaborative_features()
        
        # Get metadata features (ratings, votes, year, runtime)
        metadata_features = self._create_metadata_features()
        
        # Combine all features
        if collab_features is not None:
            # Stack all feature types horizontally
            combined_features = np.hstack([
                content_features,
                collab_features,
                metadata_features
            ])
        else:
            # No collaborative data available
            combined_features = np.hstack([
                content_features,
                metadata_features
            ])
        
        # Standardize features for SVD
        combined_features_scaled = self.scaler.fit_transform(combined_features)
        
        return combined_features_scaled
    
    def _create_basic_content_features(self):
        """Fallback content features if TF-IDF fails"""
        features = []
        
        # Define genre list
        all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
                     'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 
                     'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
        
        for _, movie in self.merged_df.iterrows():
            feature_vector = []
            
            # Genre one-hot encoding
            movie_genres = []
            if pd.notna(movie[self.genre_col]):
                movie_genres = [g.strip() for g in movie[self.genre_col].split(',')]
            
            genre_features = [1 if genre in movie_genres else 0 for genre in all_genres]
            feature_vector.extend(genre_features)
            
            # Director hash feature
            director = str(movie.get('Director', 'unknown'))
            director_hash = hash(director) % 50  # Reduce dimensionality
            feature_vector.append(director_hash)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_collaborative_features(self):
        """Create collaborative features from user ratings"""
        user_ratings_df = load_user_ratings()
        
        if user_ratings_df is None or 'Movie_ID' not in self.merged_df.columns:
            return None
        
        try:
            # Create user-item matrix
            user_item_matrix = user_ratings_df.pivot_table(
                index='User_ID', 
                columns='Movie_ID', 
                values='Rating', 
                fill_value=0
            )
            
            # Align with our movie dataset
            movie_ids = self.merged_df['Movie_ID'].values
            
            # Create features for each movie based on user ratings patterns
            collab_features = []
            
            for movie_id in movie_ids:
                if movie_id in user_item_matrix.columns:
                    # Get rating pattern for this movie
                    movie_ratings = user_item_matrix[movie_id].values
                    
                    # Extract statistical features
                    features = [
                        np.mean(movie_ratings[movie_ratings > 0]) if np.any(movie_ratings > 0) else 0,  # avg rating
                        np.sum(movie_ratings > 0),  # number of ratings
                        np.std(movie_ratings[movie_ratings > 0]) if np.sum(movie_ratings > 0) > 1 else 0,  # rating std
                        np.sum(movie_ratings >= 4) / max(1, np.sum(movie_ratings > 0)),  # high rating ratio
                        np.sum(movie_ratings <= 2) / max(1, np.sum(movie_ratings > 0)),  # low rating ratio
                    ]
                else:
                    # Movie not in user ratings - use default features
                    features = [0, 0, 0, 0, 0]
                
                collab_features.append(features)
            
            return np.array(collab_features)
            
        except Exception as e:
            return None
    
    def _create_metadata_features(self):
        """Create features from movie metadata"""
        features = []
        
        for _, movie in self.merged_df.iterrows():
            feature_vector = []
            
            # IMDB Rating (normalized)
            rating = movie.get(self.rating_col, 7.0)
            if pd.notna(rating):
                feature_vector.append(rating / 10.0)
            else:
                feature_vector.append(0.7)
            
            # Number of votes (log normalized)
            votes = movie.get('No_of_Votes', 100000)
            if pd.notna(votes) and votes > 0:
                feature_vector.append(min(np.log10(votes) / 7.0, 1.0))
            else:
                feature_vector.append(0.5)
            
            # Year (normalized)
            year_col = 'Released_Year' if 'Released_Year' in movie.index else 'Year'
            year = movie.get(year_col, 2000)
            if pd.notna(year) and isinstance(year, (int, float)) and year > 1900:
                feature_vector.append((year - 1920) / (2025 - 1920))
            else:
                feature_vector.append(0.5)
            
            # Runtime (normalized)
            runtime = movie.get('Runtime', 120)
            if pd.notna(runtime):
                if isinstance(runtime, str):
                    # Extract number from string like "120 min"
                    import re
                    runtime_match = re.search(r'(\d+)', str(runtime))
                    if runtime_match:
                        runtime = int(runtime_match.group(1))
                    else:
                        runtime = 120
                
                feature_vector.append(min(runtime / 200.0, 1.0))
            else:
                feature_vector.append(0.6)
            
            # Certificate encoding
            cert_mapping = {'G': 0.2, 'PG': 0.4, 'PG-13': 0.6, 'R': 0.8, 'NC-17': 1.0}
            cert = movie.get('Certificate', 'PG-13')
            feature_vector.append(cert_mapping.get(cert, 0.6))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit_svd_model(self, n_components=50):
        """Fit SVD model on hybrid feature matrix"""
        
        # Create hybrid feature matrix
        self.hybrid_features = self.create_hybrid_feature_matrix()
        
        # Fit SVD model
        self.svd_model = TruncatedSVD(
            n_components=min(n_components, self.hybrid_features.shape[1] - 1),
            random_state=42
        )
        
        # Transform features to latent space
        latent_features = self.svd_model.fit_transform(self.hybrid_features)
        
        return latent_features
    
    def find_similar_movies_svd(self, target_movie, top_n=10):
        """Find similar movies using SVD latent features"""
        
        if self.svd_model is None:
            self.fit_svd_model()
        
        # Find target movie index
        target_idx = None
        for idx, movie in self.merged_df.iterrows():
            if movie['Series_Title'].lower() == target_movie.lower():
                target_idx = self.merged_df.index.get_loc(idx)
                break
        
        if target_idx is None:
            return None
        
        # Get latent features
        latent_features = self.svd_model.transform(self.hybrid_features)
        
        # Calculate similarities in latent space
        target_latent = latent_features[target_idx].reshape(1, -1)
        similarities = cosine_similarity(target_latent, latent_features).flatten()
        
        # Get top similar movies (excluding target)
        similar_indices = np.argsort(-similarities)[1:top_n+1]
        
        result_df = self.merged_df.iloc[similar_indices]
        return result_df[['Series_Title', self.genre_col, self.rating_col]]
    
    def hybrid_recommend_svd(self, target_movie=None, genre=None, top_n=8):
        """Main SVD-based hybrid recommendation method"""
        
        if not target_movie and not genre:
            return None
        
        # Get traditional recommendations for comparison/fallback
        content_results = content_based_filtering_enhanced(
            self.merged_df, target_movie, genre, top_n * 2
        )
        
        collab_results = None
        if target_movie:
            collab_results = collaborative_filtering_enhanced(
                self.merged_df, target_movie, top_n * 2
            )
        
        # If we have target movie, use SVD-based similarity
        if target_movie:
            svd_results = self.find_similar_movies_svd(target_movie, top_n * 2)
            
            if svd_results is not None:
                # Combine SVD results with traditional methods
                return self._ensemble_combine(svd_results, content_results, collab_results, top_n)
        
        # Fallback to genre-based recommendations with SVD enhancement
        if genre and content_results is not None:
            return self._enhance_genre_recommendations(content_results, genre, top_n)
        
        # Final fallback
        if content_results is not None:
            return content_results.head(top_n)
        elif collab_results is not None:
            return collab_results.head(top_n)
        
        return None
    
    def _ensemble_combine(self, svd_results, content_results, collab_results, top_n):
        """Combine SVD results with traditional methods using weighted voting"""
        
        movie_scores = {}
        
        # SVD results (highest weight - 50%)
        if svd_results is not None:
            for idx, (_, row) in enumerate(svd_results.iterrows()):
                title = row['Series_Title']
                score = (len(svd_results) - idx) / len(svd_results)  # Position-based scoring
                rating_boost = row[self.rating_col] / 10.0  # Quality boost
                movie_scores[title] = score * 0.5 * rating_boost
        
        # Content-based results (30%)
        if content_results is not None:
            for idx, (_, row) in enumerate(content_results.iterrows()):
                title = row['Series_Title']
                score = (len(content_results) - idx) / len(content_results)
                rating_boost = row[self.rating_col] / 10.0
                
                if title in movie_scores:
                    movie_scores[title] += score * 0.3 * rating_boost
                else:
                    movie_scores[title] = score * 0.3 * rating_boost
        
        # Collaborative results (20%)
        if collab_results is not None:
            for idx, (_, row) in enumerate(collab_results.iterrows()):
                title = row['Series_Title']
                score = (len(collab_results) - idx) / len(collab_results)
                rating_boost = row[self.rating_col] / 10.0
                
                if title in movie_scores:
                    movie_scores[title] += score * 0.2 * rating_boost
                else:
                    movie_scores[title] = score * 0.2 * rating_boost
        
        # Sort by combined score
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        top_titles = [title for title, _ in sorted_movies[:top_n]]
        
        # Create result dataframe
        result_df = self.merged_df[self.merged_df['Series_Title'].isin(top_titles)]
        
        # Preserve order
        title_to_rank = {title: i for i, (title, _) in enumerate(sorted_movies[:top_n])}
        result_df = result_df.copy()
        result_df['rank_order'] = result_df['Series_Title'].map(title_to_rank)
        result_df = result_df.sort_values('rank_order').drop('rank_order', axis=1)
        
        return result_df[['Series_Title', self.genre_col, self.rating_col]].head(top_n)
    
    def _enhance_genre_recommendations(self, content_results, genre, top_n):
        """Enhance genre-based recommendations using SVD latent features"""
        
        if self.svd_model is None:
            self.fit_svd_model()
        
        # Get latent features for all movies
        latent_features = self.svd_model.transform(self.hybrid_features)
        
        # Find movies of the target genre
        genre_movies = []
        for idx, (_, movie) in enumerate(self.merged_df.iterrows()):
            if pd.notna(movie[self.genre_col]) and genre.lower() in movie[self.genre_col].lower():
                genre_movies.append((idx, latent_features[idx]))
        
        if not genre_movies:
            return content_results.head(top_n)
        
        # Calculate centroid of genre in latent space
        genre_latent_features = np.array([features for _, features in genre_movies])
        genre_centroid = np.mean(genre_latent_features, axis=0)
        
        # Find movies closest to genre centroid
        similarities = cosine_similarity([genre_centroid], latent_features).flatten()
        
        # Combine with content-based scores
        enhanced_scores = {}
        
        # Start with content results
        for _, row in content_results.iterrows():
            title = row['Series_Title']
            movie_idx = self.merged_df[self.merged_df['Series_Title'] == title].index[0]
            dataset_idx = self.merged_df.index.get_loc(movie_idx)
            
            content_score = 0.6  # Base content score
            svd_score = similarities[dataset_idx] * 0.4  # SVD similarity score
            quality_score = (row[self.rating_col] / 10.0) * 0.2  # Quality bonus
            
            enhanced_scores[title] = content_score + svd_score + quality_score
        
        # Sort and return top N
        sorted_enhanced = sorted(enhanced_scores.items(), key=lambda x: x[1], reverse=True)
        top_titles = [title for title, _ in sorted_enhanced[:top_n]]
        
        result_df = self.merged_df[self.merged_df['Series_Title'].isin(top_titles)]
        
        # Preserve order
        title_to_rank = {title: i for i, (title, _) in enumerate(sorted_enhanced[:top_n])}
        result_df = result_df.copy()
        result_df['rank_order'] = result_df['Series_Title'].map(title_to_rank)
        result_df = result_df.sort_values('rank_order').drop('rank_order', axis=1)
        
        return result_df[['Series_Title', self.genre_col, self.rating_col]].head(top_n)

# Main interface functions
@st.cache_data
def svd_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """SVD-based hybrid recommendation system"""
    recommender = SVDHybridRecommender(merged_df)
    return recommender.hybrid_recommend_svd(target_movie, genre, top_n)

# Updated main hybrid function
@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """Updated smart hybrid using SVD-based approach"""
    return svd_hybrid_recommendation(merged_df, target_movie, genre, top_n)