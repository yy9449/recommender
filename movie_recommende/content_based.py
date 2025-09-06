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
        # Sort by match ratio
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in partial_matches]

    # Close matches
    return get_close_matches(input_title, titles_list, n=5, cutoff=cutoff)

@st.cache_data
def create_content_features(merged_df):
    """Create enhanced numeric content feature matrix based on genres and metadata."""
    features = []
    # Fixed genre vocabulary
    all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
                 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    # Robust director column resolution
    director_col = 'Director_y' if 'Director_y' in merged_df.columns else ('Director_x' if 'Director_x' in merged_df.columns else 'Director')
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
    runtime_col = 'Runtime'

    for _, movie in merged_df.iterrows():
        vector = []

        # Genres one-hot
        movie_genres = []
        if genre_col in movie and pd.notna(movie[genre_col]):
            movie_genres = [g.strip() for g in str(movie[genre_col]).split(',')]
        vector.extend([1 if g in movie_genres else 0 for g in all_genres])

        # Director hashed
        director_val = str(movie.get(director_col, 'unknown'))
        director_hash = hash(director_val) % 100
        vector.append(director_hash)

        # Year normalized
        year_val = safe_convert_to_numeric(movie.get(year_col), 2000)
        if year_val and 1900 <= year_val <= 2025:
            norm_year = (year_val - 1920) / (2025 - 1920)
        else:
            norm_year = 0.5
        vector.append(norm_year)

        # Runtime normalized
        runtime_val = safe_convert_to_numeric(movie.get(runtime_col), 120)
        if runtime_val and runtime_val > 0:
            norm_runtime = min(runtime_val / 200.0, 1.0)
        else:
            norm_runtime = 0.6
        vector.append(norm_runtime)

        # Rating normalized
        rating_val = safe_convert_to_numeric(movie.get(rating_col), 7.0)
        if rating_val is not None and 0 <= rating_val <= 10:
            norm_rating = rating_val / 10.0
        else:
            norm_rating = 0.7
        vector.append(norm_rating)

        features.append(vector)

    return np.array(features)

@st.cache_data
def create_genre_features(merged_df):
    """Create TF-IDF features using only genres (comma-separated)."""
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    def genre_tokenizer(text):
        return [g.strip().lower().replace('-', '') for g in str(text).split(',') if g and g.strip()]
    vectorizer = TfidfVectorizer(
        tokenizer=genre_tokenizer,
        preprocessor=None,
        token_pattern=None,
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 1)
    )
    tfidf_matrix = vectorizer.fit_transform(merged_df[genre_col].fillna(''))
    return tfidf_matrix, genre_col

@st.cache_data
def content_based_filtering_enhanced(merged_df, target_movie=None, genre=None, top_n=8):
    """Enhanced Content-Based filtering with improved feature engineering and fuzzy matching"""
    if target_movie:
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if not similar_titles:
            return None
        
        target_title = similar_titles[0]
        
        # Ensure the target movie exists in the dataframe
        if target_title not in merged_df['Series_Title'].values:
            return None
            
        target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
        
        # Content similarity on numeric feature space
        content_features = create_content_features(merged_df)
        target_vec = content_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        sims = cosine_similarity(target_vec, content_features).flatten()
        similar_indices = np.argsort(-sims)
        similar_indices = [idx for idx in similar_indices if idx != merged_df.index.get_loc(target_idx)]
        top_indices = similar_indices[:top_n]
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        result_df = merged_df.iloc[top_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
    
    elif genre:
        genre_features, genre_col = create_genre_features(merged_df)
        vectorizer_ready = genre_features  # already fitted to corpus
        # Build query vector by fitting tokenizer to the single genre string via the same vectorizer
        # We reuse the same tokenizer by creating a tiny vectorizer with identical tokenizer is hard; instead use the fitted vectorizer's vocabulary
        # Create a zero vector and fill indices for tokens present in vocabulary
        from scipy.sparse import csr_matrix
        tokens = [g.strip().lower().replace('-', '') for g in str(genre).split(',') if g and g.strip()]
        vocab = {v: k for k, v in enumerate(genre_features.T.nonzero()[0])}  # fallback if needed
        # More robust approach: rebuild using the existing fitted vectorizer via its analyzer is not accessible; fallback to transform via re-fitting a compatible vectorizer
        # Simpler: match titles whose genre contains the token(s)
        mask = merged_df[genre_col].fillna('').str.contains(genre, case=False)
        candidates = merged_df[mask].index.tolist()
        if not candidates:
            candidates = list(range(len(merged_df)))
        sims = cosine_similarity(genre_features[candidates], genre_features[candidates]).A[0] if candidates else np.array([])
        order = np.argsort(-sims) if sims.size else []
        ranked = [candidates[i] for i in order[:top_n]] if sims.size else merged_df.index[:top_n].tolist()
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        result_df = merged_df.loc[ranked]
        return result_df[['Series_Title', genre_col, rating_col]]
        
    return None