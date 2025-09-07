import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from content_based import content_based_filtering_enhanced, create_content_features, find_rating_column, find_genre_column, find_director_column
from collaborative import collaborative_filtering_enhanced, load_user_ratings
import warnings
warnings.filterwarnings('ignore')

# Optimized recommendation weights based on evaluation results
ALPHA = 0.4  # Content-based weight
BETA = 0.4   # Collaborative weight
GAMMA = 0.1  # Popularity weight
DELTA = 0.1  # Recency weight


def _calculate_popularity(df: pd.DataFrame) -> pd.Series:
    """Calculate popularity score based on votes and ratings"""
    rating_col = find_rating_column(df)
    votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else None
    
    if votes_col is None:
        # Fallback to just ratings
        return df[rating_col].fillna(df[rating_col].mean())
    
    log_votes = np.log1p(df[votes_col].fillna(0))
    rating = df[rating_col].fillna(df[rating_col].mean())
    return rating * log_votes


def _calculate_recency(df: pd.DataFrame) -> pd.Series:
    """Calculate recency score with exponential decay"""
    current_year = pd.to_datetime('today').year
    decay_rate = 0.98
    
    year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
    if year_col not in df.columns:
        return pd.Series([0.5] * len(df), index=df.index)
    
    years = df[year_col].fillna(df[year_col].mode()[0] if not df[year_col].mode().empty else 2000)
    age = current_year - years
    return decay_rate ** age


def _create_optimized_soup(df: pd.DataFrame) -> pd.Series:
    """Create optimized soup for content-based similarity"""
    # Optimized weights from evaluation
    w_overview = 4.0
    w_genre = 3.0
    w_title = 2.0
    w_director = 1.0

    genre_col = find_genre_column(df)
    director_col = find_director_column(df)
    
    overview = df.get('Overview', pd.Series([''] * len(df))).fillna('').astype(str)
    genre = df[genre_col].fillna('').astype(str)
    title = df['Series_Title'].fillna('').astype(str)
    director = df[director_col].fillna('').astype(str)

    soup = (
        (overview + ' ') * int(w_overview) +
        (genre + ' ') * int(w_genre) +
        (title + ' ') * int(w_title) +
        (director + ' ') * int(w_director)
    )
    return soup


@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """
    Enhanced hybrid recommendation system combining content-based, collaborative,
    popularity, and recency signals with optimized weights.
    """
    if not target_movie and not genre:
        return None
    
    # Get individual recommendation results for comparison and fallback
    content_results = content_based_filtering_enhanced(
        merged_df, target_movie, genre, top_n * 2
    )
    
    collab_results = None
    if target_movie:
        collab_results = collaborative_filtering_enhanced(
            merged_df, target_movie, top_n * 2
        )
    
    # If we have a target movie, create hybrid scores
    if target_movie and target_movie in merged_df['Series_Title'].values:
        idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]
        
        # 1. Content-based similarity
        try:
            soup = _create_optimized_soup(merged_df)
            tfidf = TfidfVectorizer(
                stop_words='english', 
                ngram_range=(1, 2), 
                min_df=5, 
                max_df=0.8
            )
            tfidf_matrix = tfidf.fit_transform(soup)
            content_sim_matrix = cosine_similarity(tfidf_matrix)
            content_scores = content_sim_matrix[idx]
        except Exception:
            # Fallback to basic content similarity
            simple_soup = merged_df[find_genre_column(merged_df)].fillna('')
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(simple_soup)
            content_sim_matrix = cosine_similarity(tfidf_matrix)
            content_scores = content_sim_matrix[idx]
        
        # 2. Collaborative scores (simplified)
        collab_scores = np.zeros(len(merged_df))
        user_ratings_df = load_user_ratings()
        if user_ratings_df is not None and 'Movie_ID' in merged_df.columns:
            try:
                # Simple collaborative scoring based on user rating patterns
                user_item_matrix = user_ratings_df.pivot_table(
                    index='Movie_ID', columns='User_ID', values='Rating'
                ).fillna(0)
                
                aligned_movies = merged_df['Movie_ID'].values
                for i, movie_id in enumerate(aligned_movies):
                    if movie_id in user_item_matrix.index:
                        # Simple similarity based on rating correlation
                        target_movie_id = merged_df.iloc[idx]['Movie_ID']
                        if target_movie_id in user_item_matrix.index:
                            target_ratings = user_item_matrix.loc[target_movie_id].values
                            movie_ratings = user_item_matrix.loc[movie_id].values
                            if np.sum(target_ratings > 0) > 0 and np.sum(movie_ratings > 0) > 0:
                                # Calculate correlation between non-zero ratings
                                common_users = (target_ratings > 0) & (movie_ratings > 0)
                                if np.sum(common_users) > 1:
                                    corr = np.corrcoef(
                                        target_ratings[common_users], 
                                        movie_ratings[common_users]
                                    )[0, 1]
                                    if not np.isnan(corr):
                                        collab_scores[i] = max(0, corr)
            except Exception:
                pass  # Keep zeros if collaborative scoring fails
        
        # 3. Popularity and recency scores
        popularity_scores = _calculate_popularity(merged_df)
        recency_scores = _calculate_recency(merged_df)
        
        # 4. Scale and combine all scores
        scaler = MinMaxScaler()
        
        try:
            scaled_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
        except:
            scaled_content = np.zeros(len(merged_df))
            
        try:
            scaled_collab = scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()
        except:
            scaled_collab = np.zeros(len(merged_df))
            
        try:
            scaled_popularity = scaler.fit_transform(popularity_scores.values.reshape(-1, 1)).flatten()
        except:
            scaled_popularity = np.zeros(len(merged_df))
            
        try:
            scaled_recency = scaler.fit_transform(recency_scores.values.reshape(-1, 1)).flatten()
        except:
            scaled_recency = np.zeros(len(merged_df))
        
        # Combine with optimized weights
        final_scores = (
            ALPHA * scaled_content +
            BETA * scaled_collab +
            GAMMA * scaled_popularity +
            DELTA * scaled_recency
        )
        
        # Get top recommendations (excluding the target movie)
        sim_scores = sorted(
            list(enumerate(final_scores)), 
            key=lambda x: x[1], 
            reverse=True
        )
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
        movie_indices = [i[0] for i in sim_scores]
        
        result_df = merged_df.iloc[movie_indices]
        rating_col = find_rating_column(merged_df)
        genre_col = find_genre_column(merged_df)
        return result_df[['Series_Title', genre_col, rating_col]]
    
    # Genre-based recommendations with hybrid enhancement
    elif genre and content_results is not None:
        # Enhance genre recommendations with popularity and recency
        popularity_scores = _calculate_popularity(merged_df)
        recency_scores = _calculate_recency(merged_df)
        
        # Get genre-matched movies
        genre_col = find_genre_column(merged_df)
        genre_movies = merged_df[
            merged_df[genre_col].str.contains(genre, case=False, na=False)
        ]
        
        if not genre_movies.empty:
            # Calculate hybrid scores for genre movies
            scaler = MinMaxScaler()
            
            pop_scores = popularity_scores[genre_movies.index]
            rec_scores = recency_scores[genre_movies.index]
            
            try:
                scaled_pop = scaler.fit_transform(pop_scores.values.reshape(-1, 1)).flatten()
                scaled_rec = scaler.fit_transform(rec_scores.values.reshape(-1, 1)).flatten()
                
                # Combine popularity and recency for genre recommendations
                hybrid_scores = 0.7 * scaled_pop + 0.3 * scaled_rec
                
                # Sort by hybrid score
                scored_indices = sorted(
                    enumerate(hybrid_scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                
                result_indices = [genre_movies.index[i] for i, _ in scored_indices]
                result_df = merged_df.loc[result_indices]
                
                rating_col = find_rating_column(merged_df)
                return result_df[['Series_Title', genre_col, rating_col]]
                
            except Exception:
                pass
    
    # Fallback to individual algorithm results
    if content_results is not None and not content_results.empty:
        return content_results.head(top_n)
    elif collab_results is not None and not collab_results.empty:
        return collab_results.head(top_n)
    
    return None


# Maintain compatibility with original API
@st.cache_data
def svd_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    """Wrapper for compatibility with original API"""
    return smart_hybrid_recommendation(merged_df, target_movie, genre, top_n)
