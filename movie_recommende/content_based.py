import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from difflib import get_close_matches
import streamlit as st

def safe_convert_to_numeric(value, default=None):
    """Safely convert a value to numeric, handling strings and NaN"""
    if pd.isna(value):
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove any non-numeric characters except decimal point
        clean_value = re.sub(r'[^\d.-]', '', str(value))
        try:
            return float(clean_value) if clean_value else default
        except (ValueError, TypeError):
            return default
    
    return default

def find_similar_titles(input_title, titles_list, cutoff=0.6):
    """Enhanced fuzzy matching for movie titles"""
    if not input_title or not titles_list:
        return []
    
    input_lower = input_title.lower().strip()
    
    # Direct match
    exact_matches = [title for title in titles_list if title.lower() == input_lower]
    if exact_matches:
        return exact_matches
    
    # Partial match
    partial_matches = []
    for title in titles_list:
        title_lower = title.lower()
        if input_lower in title_lower:
            partial_matches.append((title, len(input_lower) / len(title_lower)))
        elif title_lower in input_lower:
            partial_matches.append((title, len(title_lower) / len(input_lower)))
    
    if partial_matches:
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in partial_matches[:3]]
    
    # Fuzzy match
    matches = get_close_matches(input_title, titles_list, n=5, cutoff=cutoff)
    return matches

@st.cache_data
def create_content_features(merged_df):
    """Create enhanced content-based features matrix"""
    features = []
    
    for _, movie in merged_df.iterrows():
        feature_vector = []
        
        # Genre features (one-hot encoded)
        all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
                     'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 
                     'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
        
        movie_genres = []
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        if pd.notna(movie[genre_col]):
            movie_genres = [g.strip() for g in movie[genre_col].split(',')]
        
        # One-hot encode genres
        genre_features = [1 if genre in movie_genres else 0 for genre in all_genres]
        feature_vector.extend(genre_features)
        
        # Director feature (simplified)
        director_col = 'Director' if 'Director' in merged_df.columns else 'Director'
        director_hash = hash(str(movie.get(director_col, 'unknown'))) % 100
        feature_vector.append(director_hash)
        
        # Year feature (normalized)
        year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
        year = safe_convert_to_numeric(movie.get(year_col), 2000)
        if year and 1900 <= year <= 2025:
            normalized_year = (year - 1920) / (2025 - 1920)
        else:
            normalized_year = 0.5
        feature_vector.append(normalized_year)
        
        # Runtime feature (normalized)
        runtime_col = 'Runtime' if 'Runtime' in merged_df.columns else 'Runtime'
        runtime = safe_convert_to_numeric(movie.get(runtime_col), 120)
        if runtime and runtime > 0:
            normalized_runtime = min(runtime / 200.0, 1.0)
        else:
            normalized_runtime = 0.6
        feature_vector.append(normalized_runtime)
        
        # Rating feature
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        rating = safe_convert_to_numeric(movie.get(rating_col), 7.0)
        if rating and 0 <= rating <= 10:
            normalized_rating = rating / 10.0
        else:
            normalized_rating = 0.7
        feature_vector.append(normalized_rating)
        
        features.append(feature_vector)
    
    return np.array(features)

@st.cache_data
def content_based_filtering_enhanced(merged_df, target_movie=None, genre=None, top_n=5):
    """Enhanced content-based filtering"""
    if target_movie:
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if not similar_titles:
            return None
        
        target_title = similar_titles[0]
        target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
        
        content_features = create_content_features(merged_df)
        target_features = content_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        similarities = cosine_similarity(target_features, content_features).flatten()
        similar_indices = np.argsort(-similarities)[1:top_n+1]
        
        result_df = merged_df.iloc[similar_indices]
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        return result_df[['Series_Title', genre_col, rating_col]]
    
    elif genre:
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        
        genre_corpus = merged_df[genre_col].fillna('').tolist()
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(genre_corpus)
        query_vector = tfidf.transform([genre])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = np.argsort(-similarities)[:top_n]
        
        result_df = merged_df.iloc[top_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
    
    return None
