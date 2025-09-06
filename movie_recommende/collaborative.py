import pandas as pd
import numpy as np
import streamlit as st
from content_based import content_tfidf_advanced, find_rating_column, find_genre_column
from collaborative import collaborative_knn, load_user_ratings
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SingleHybridRecommender:
    def __init__(self, merged_df):
        self.merged_df = merged_df
        self.rating_col = find_rating_column(merged_df)
        self.genre_col = find_genre_column(merged_df)
        self.user_ratings_df = load_user_ratings()
        
        # Hybrid weights for: FinalScore = α×Content + β×CF + γ×Popularity + δ×Recency
        self.alpha = 0.4  # Content weight
        self.beta = 0.3   # Collaborative weight
        self.gamma = 0.2  # Popularity weight
        self.delta = 0.1  # Recency weight
    
    def get_content_scores(self, target_movie, genre, top_n):
        """Get normalized content-based scores"""
        content_results = content_tfidf_advanced(self.merged_df, target_movie, genre, top_n * 3)
        content_scores = {}
        
        if content_results is not None and not content_results.empty:
            max_rating = content_results[self.rating_col].max()
            for _, movie in content_results.iterrows():
                title = movie['Series_Title']
                content_scores[title] = movie[self.rating_col] / max_rating
        
        return content_scores
    
    def get_collaborative_scores(self, target_movie, top_n):
        """Get normalized collaborative filtering scores"""
        cf_scores = {}
        
        if target_movie and self.user_ratings_df is not None:
            cf_results = collaborative_knn(self.merged_df, target_movie, top_n=top_n * 3)
            if cf_results is not None and not cf_results.empty:
                max_rating = cf_results[self.rating_col].max()
                for _, movie in cf_results.iterrows():
                    title = movie['Series_Title']
                    cf_scores[title] = movie[self.rating_col] / max_rating
        
        return cf_scores
    
    def get_popularity_scores(self):
        """Calculate popularity scores using rating and interaction frequency"""
        popularity_scores = {}
        
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            rating = movie.get(self.rating_col, 7.0)
            votes = movie.get('No_of_Votes', movie.get('Votes', 1000))
            
            if pd.isna(votes):
                votes = 1000
            if pd.isna(rating):
                rating = 7.0
            
            # Popularity formula: rating weighted by vote count
            popularity = rating * np.log10(votes + 1) / 10.0
            popularity_scores[title] = min(popularity, 1.0)
        
        # Boost with user interaction frequency if available
        if self.user_ratings_df is not None:
            for movie_id in self.user_ratings_df['Movie_ID'].unique():
                interaction_count = len(self.user_ratings_df[self.user_ratings_df['Movie_ID'] == movie_id])
                
                movie_match = self.merged_df[self.merged_df['Movie_ID'] == movie_id]
                if not movie_match.empty:
                    title = movie_match.iloc[0]['Series_Title']
                    
                    # Normalize interaction count (max 100 interactions = 1.0)
                    interaction_score = min(interaction_count / 100.0, 1.0)
                    
                    if title in popularity_scores:
                        # Combine: 60% rating popularity + 40% interaction popularity
                        popularity_scores[title] = 0.6 * popularity_scores[title] + 0.4 * interaction_score
        
        return popularity_scores
    
    def get_recency_scores(self):
        """Calculate recency scores - newer movies get higher weight"""
        current_year = datetime.now().year
        recency_scores = {}
        
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            year = movie.get('Released_Year', movie.get('Year', 2000))
            
            if pd.isna(year):
                year = 2000
            
            # Exponential decay: newer = higher score
            year_diff = current_year - year
            recency_factor = np.exp(-year_diff / 20.0)  # 20-year half-life
            recency_scores[title] = max(min(recency_factor, 1.0), 0.1)
        
        return recency_scores
    
    def hybrid_recommend(self, target_movie=None, genre=None, top_n=8):
        """
        Main hybrid recommendation using:
        FinalScore = α×ContentScore + β×CFScore + γ×PopularityScore + δ×RecencyScore
        """
        
        # Get all component scores
        content_scores = self.get_content_scores(target_movie, genre, top_n)
        cf_scores = self.get_collaborative_scores(target_movie, top_n)
        popularity_scores = self.get_popularity_scores()
        recency_scores = self.get_recency_scores()
        
        # Collect all candidate movies
        all_candidates = set(content_scores.keys()) | set(cf_scores.keys())
        
        # Add popular movies if we don't have enough candidates
        if len(all_candidates) < top_n * 2:
            popular_movies = sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)
            for title, _ in popular_movies[:top_n * 2]:
                all_candidates.add(title)
        
        # Calculate final hybrid scores
        final_scores = {}
        
        for title in all_candidates:
            # Get individual scores (default to 0 if not available)
            content_score = content_scores.get(title, 0)
            cf_score = cf_scores.get(title, 0)
            popularity_score = popularity_scores.get(title, 0.5)
            recency_score = recency_scores.get(title, 0.5)
            
            # Apply hybrid formula
            final_score = (self.alpha * content_score + 
                          self.beta * cf_score + 
                          self.gamma * popularity_score + 
                          self.delta * recency_score)
            
            # Genre boost: if target movie provided, boost similar genres
            if target_movie:
                genre_boost = self.calculate_genre_boost(title, target_movie)
                final_score *= genre_boost
            
            final_scores[title] = final_score
        
        # Sort by final score and get top N
        sorted_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_titles = [title for title, _ in sorted_movies]
        
        # Create result DataFrame
        result_df = self.merged_df[self.merged_df['Series_Title'].isin(top_titles)]
        
        if result_df.empty:
            return None
        
        # Preserve ranking order
        title_to_rank = {title: i for i, (title, _) in enumerate(sorted_movies)}
        result_df = result_df.copy()
        result_df['rank_order'] = result_df['Series_Title'].map(title_to_rank)
        result_df = result_df.sort_values('rank_order').drop('rank_order', axis=1)
        
        return result_df[['Series_Title', self.genre_col, self.rating_col]]
    
    def calculate_genre_boost(self, movie_title, target_movie):
        """Calculate genre similarity boost"""
        try:
            target_row = self.merged_df[self.merged_df['Series_Title'].str.lower() == target_movie.lower()]
            movie_row = self.merged_df[self.merged_df['Series_Title'] == movie_title]
            
            if target_row.empty or movie_row.empty:
                return 1.0
            
            target_genres = str(target_row.iloc[0][self.genre_col]).lower().split(', ')
            movie_genres = str(movie_row.iloc[0][self.genre_col]).lower().split(', ')
            
            # Count matching genres
            matches = len(set(target_genres) & set(movie_genres))
            
            # 15% boost per matching genre
            return 1.0 + (0.15 * matches)
            
        except Exception:
            return 1.0

# Main interface function
@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """Single hybrid recommendation system with all advanced features"""
    recommender = SingleHybridRecommender(merged_df)
    return recommender.hybrid_recommend(target_movie, genre, top_n)

# Alternative interface for backwards compatibility
@st.cache_data 
def advanced_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """Alternative interface name"""
    return smart_hybrid_recommendation(merged_df, target_movie, genre, top_n)