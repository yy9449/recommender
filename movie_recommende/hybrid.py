import pandas as pd
import numpy as np
import streamlit as st
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced, load_user_ratings
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

class ProductionHybridRecommender:
    def __init__(self, merged_df):
        self.merged_df = merged_df
        self.rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        self.genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        self.scaler = MinMaxScaler()
        
    def normalize_scores(self, scores_dict):
        """Normalize scores to 0-1 range for fair combination"""
        if not scores_dict:
            return {}
        
        scores_array = np.array(list(scores_dict.values())).reshape(-1, 1)
        normalized_array = self.scaler.fit_transform(scores_array).flatten()
        
        return dict(zip(scores_dict.keys(), normalized_array))
    
    def calculate_algorithm_confidence(self, results, algorithm_type):
        """Calculate confidence score for each algorithm's recommendations"""
        if results is None or results.empty:
            return 0.0
        
        # Base confidence factors
        size_factor = min(len(results) / 10.0, 1.0)  # More results = higher confidence
        quality_factor = results[self.rating_col].mean() / 10.0  # Higher rated movies = higher confidence
        
        # Algorithm-specific confidence adjustments
        if algorithm_type == 'collaborative':
            user_ratings_df = load_user_ratings()
            has_real_data = user_ratings_df is not None and 'user_ratings_df' in st.session_state
            data_factor = 1.0 if has_real_data else 0.6  # Lower confidence for synthetic data
        else:
            data_factor = 0.9  # Content-based is generally reliable
        
        return size_factor * quality_factor * data_factor
    
    def ensemble_weighted_average(self, content_results, collaborative_results, content_weight=0.6):
        """Netflix-style weighted ensemble approach"""
        movie_scores = defaultdict(float)
        movie_weights = defaultdict(float)
        
        # Process content-based results
        if content_results is not None and not content_results.empty:
            content_confidence = self.calculate_algorithm_confidence(content_results, 'content')
            content_scores = {}
            
            for idx, (_, row) in enumerate(content_results.iterrows()):
                # Decay factor for ranking position (higher ranks get lower scores)
                position_weight = 1.0 / (1.0 + idx * 0.1)
                score = (row[self.rating_col] / 10.0) * position_weight * content_confidence
                content_scores[row['Series_Title']] = score
            
            # Normalize content scores
            content_scores = self.normalize_scores(content_scores)
            
            # Add to ensemble
            for title, score in content_scores.items():
                movie_scores[title] += score * content_weight
                movie_weights[title] += content_weight
        
        # Process collaborative results
        if collaborative_results is not None and not collaborative_results.empty:
            collaborative_confidence = self.calculate_algorithm_confidence(collaborative_results, 'collaborative')
            collab_scores = {}
            
            for idx, (_, row) in enumerate(collaborative_results.iterrows()):
                position_weight = 1.0 / (1.0 + idx * 0.1)
                score = (row[self.rating_col] / 10.0) * position_weight * collaborative_confidence
                collab_scores[row['Series_Title']] = score
            
            # Normalize collaborative scores
            collab_scores = self.normalize_scores(collab_scores)
            
            # Add to ensemble
            collaborative_weight = 1.0 - content_weight
            for title, score in collab_scores.items():
                movie_scores[title] += score * collaborative_weight
                movie_weights[title] += collaborative_weight
        
        # Calculate final weighted averages
        final_scores = {}
        for title in movie_scores:
            if movie_weights[title] > 0:
                final_scores[title] = movie_scores[title] / movie_weights[title]
        
        return final_scores
    
    def apply_business_rules(self, movie_scores, target_movie=None, genre=None):
        """Apply business rules for diversity and quality"""
        enhanced_scores = {}
        
        for title, score in movie_scores.items():
            movie_info = self.merged_df[self.merged_df['Series_Title'] == title]
            if movie_info.empty:
                continue
            
            movie_row = movie_info.iloc[0]
            enhanced_score = score
            
            # Quality boost for high IMDB ratings
            imdb_rating = movie_row[self.rating_col]
            if pd.notna(imdb_rating) and imdb_rating >= 8.0:
                enhanced_score *= 1.15  # 15% boost for high-quality movies
            
            # Genre relevance boost
            if genre:
                movie_genres = []
                if pd.notna(movie_row[self.genre_col]):
                    movie_genres = [g.strip().lower() for g in movie_row[self.genre_col].split(',')]
                
                if genre.lower() in movie_genres:
                    enhanced_score *= 1.25  # 25% boost for genre match
                elif any(g in genre.lower() or genre.lower() in g for g in movie_genres):
                    enhanced_score *= 1.10  # 10% boost for partial genre match
            
            # Recency boost (newer movies get slight advantage)
            year_col = 'Released_Year' if 'Released_Year' in movie_row.index else 'Year'
            if year_col in movie_row.index and pd.notna(movie_row[year_col]):
                year = int(movie_row[year_col]) if movie_row[year_col] != 'Unknown' else 2000
                if year >= 2015:
                    enhanced_score *= 1.05  # 5% boost for recent movies
            
            enhanced_scores[title] = enhanced_score
        
        return enhanced_scores
    
    def ensure_diversity(self, final_scores, top_n):
        """Ensure diverse recommendations across genres, years, directors"""
        sorted_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        diverse_recommendations = []
        genre_counts = defaultdict(int)
        director_counts = defaultdict(int)
        year_counts = defaultdict(int)
        
        # Diversity constraints
        max_same_genre = max(2, top_n // 3)  # At most 1/3 from same genre
        max_same_director = max(1, top_n // 5)  # At most 1/5 from same director
        max_same_year = max(2, top_n // 4)  # At most 1/4 from same year
        
        for title, score in sorted_movies:
            if len(diverse_recommendations) >= top_n:
                break
            
            movie_info = self.merged_df[self.merged_df['Series_Title'] == title]
            if movie_info.empty:
                continue
            
            movie_row = movie_info.iloc[0]
            
            # Extract movie attributes
            director = str(movie_row.get('Director', 'Unknown'))
            year_col = 'Released_Year' if 'Released_Year' in movie_row.index else 'Year'
            year = str(movie_row.get(year_col, 'Unknown'))
            
            # Get primary genre
            primary_genre = 'Unknown'
            if pd.notna(movie_row[self.genre_col]):
                genres = [g.strip() for g in movie_row[self.genre_col].split(',')]
                primary_genre = genres[0] if genres else 'Unknown'
            
            # Check diversity constraints
            genre_ok = genre_counts[primary_genre] < max_same_genre
            director_ok = director_counts[director] < max_same_director
            year_ok = year_counts[year] < max_same_year
            
            # Accept if within diversity limits OR if we need to fill remaining slots
            if (genre_ok and director_ok and year_ok) or len(diverse_recommendations) < top_n // 2:
                diverse_recommendations.append((title, score))
                genre_counts[primary_genre] += 1
                director_counts[director] += 1
                year_counts[year] += 1
        
        # Fill remaining slots if needed (relaxing constraints)
        if len(diverse_recommendations) < top_n:
            remaining_slots = top_n - len(diverse_recommendations)
            existing_titles = {title for title, _ in diverse_recommendations}
            
            for title, score in sorted_movies:
                if title not in existing_titles and remaining_slots > 0:
                    diverse_recommendations.append((title, score))
                    remaining_slots -= 1
        
        return diverse_recommendations
    
    def recommend(self, target_movie=None, genre=None, top_n=8):
        """Main recommendation method using production-quality hybrid approach - silent version"""
        if not target_movie and not genre:
            return None
        
        # Determine optimal weights based on input and data quality
        user_ratings_df = load_user_ratings()
        has_real_user_data = user_ratings_df is not None and 'user_ratings_df' in st.session_state
        
        if target_movie and genre:
            # Both inputs - balanced approach
            content_weight = 0.65 if has_real_user_data else 0.75
        elif target_movie:
            # Movie only - balanced
            content_weight = 0.60 if has_real_user_data else 0.70
        else:
            # Genre only - content-heavy
            content_weight = 0.85
        
        # Get recommendations from both algorithms
        content_results = None
        collaborative_results = None
        
        # Content-based recommendations
        if target_movie or genre:
            content_results = content_based_filtering_enhanced(
                self.merged_df, target_movie, genre, min(top_n * 2, 20)
            )
        
        # Collaborative recommendations (only if we have a target movie)
        if target_movie:
            collaborative_results = collaborative_filtering_enhanced(
                self.merged_df, target_movie, min(top_n * 2, 20)
            )
        
        # Handle genre-only case
        if genre and not target_movie:
            if content_results is not None and not content_results.empty:
                # Apply enhanced scoring for genre-only recommendations
                enhanced_scores = {}
                for _, row in content_results.iterrows():
                    title = row['Series_Title']
                    base_score = row[self.rating_col] / 10.0
                    
                    # Genre matching bonus
                    movie_genres = []
                    if pd.notna(row[self.genre_col]):
                        movie_genres = [g.strip().lower() for g in row[self.genre_col].split(',')]
                    
                    genre_bonus = 1.5 if genre.lower() in movie_genres else 1.0
                    quality_bonus = (base_score * 0.3) + 0.7
                    
                    enhanced_scores[title] = base_score * genre_bonus * quality_bonus
                
                # Apply business rules and ensure diversity
                final_scores = self.apply_business_rules(enhanced_scores, target_movie, genre)
                diverse_recommendations = self.ensure_diversity(final_scores, top_n)
                
                # Create result dataframe
                final_titles = [title for title, _ in diverse_recommendations]
                result_df = self.merged_df[self.merged_df['Series_Title'].isin(final_titles)]
                
                # Preserve recommendation order
                title_to_score = dict(diverse_recommendations)
                result_df = result_df.copy()
                result_df['rec_score'] = result_df['Series_Title'].map(title_to_score)
                result_df = result_df.sort_values('rec_score', ascending=False)
                result_df = result_df.drop('rec_score', axis=1)
                
                return result_df[['Series_Title', self.genre_col, self.rating_col]].head(top_n)
            return None
        
        # Ensemble combination for movie-based recommendations
        if content_results is None and collaborative_results is None:
            return None
        
        # Weighted ensemble
        movie_scores = self.ensemble_weighted_average(
            content_results, collaborative_results, content_weight
        )
        
        if not movie_scores:
            return None
        
        # Apply business rules
        enhanced_scores = self.apply_business_rules(movie_scores, target_movie, genre)
        
        # Ensure diversity
        diverse_recommendations = self.ensure_diversity(enhanced_scores, top_n)
        
        if not diverse_recommendations:
            return None
        
        # Create final result dataframe
        final_titles = [title for title, _ in diverse_recommendations]
        result_df = self.merged_df[self.merged_df['Series_Title'].isin(final_titles)]
        
        # Preserve recommendation order
        title_to_score = dict(diverse_recommendations)
        result_df = result_df.copy()
        result_df['rec_score'] = result_df['Series_Title'].map(title_to_score)
        result_df = result_df.sort_values('rec_score', ascending=False)
        result_df = result_df.drop('rec_score', axis=1)
        
        return result_df[['Series_Title', self.genre_col, self.rating_col]].head(top_n)

# Main interface functions
@st.cache_data
def production_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """Production-quality hybrid recommendation system - silent version"""
    recommender = ProductionHybridRecommender(merged_df)
    return recommender.recommend(target_movie, genre, top_n)

@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """Updated smart hybrid using production approach - silent version"""
    return production_hybrid_recommendation(merged_df, target_movie, genre, top_n)

@st.cache_data
def hybrid_recommendation_system(merged_df, target_movie=None, genre=None, top_n=8):
    """Main hybrid system for backward compatibility - silent version"""
    return production_hybrid_recommendation(merged_df, target_movie, genre, top_n)
