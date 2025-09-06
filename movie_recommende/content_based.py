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


def find_rating_column(df: pd.DataFrame) -> str:
    return 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'


def find_genre_column(df: pd.DataFrame) -> str:
    return 'Genre_y' if 'Genre_y' in df.columns else 'Genre'


def find_director_column(df: pd.DataFrame) -> str:
    if 'Director_y' in df.columns:
        return 'Director_y'
    if 'Director_x' in df.columns:
        return 'Director_x'
    return 'Director'


def find_votes_column(df: pd.DataFrame) -> str:
    return 'No_of_Votes' if 'No_of_Votes' in df.columns else 'Votes'


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


def _compute_quality_vector(merged_df: pd.DataFrame) -> np.ndarray:
    """Normalized quality signal from rating and votes, in [0,1] per item order."""
    rating_col = find_rating_column(merged_df)
    votes_col = find_votes_column(merged_df)
    qualities: list[float] = []
    for _, row in merged_df.iterrows():
        rating_val = safe_convert_to_numeric(row.get(rating_col, np.nan), default=np.nan)
        if pd.isna(rating_val):
            rating_val = 7.0
        votes_val = row.get(votes_col, 1000)
        try:
            votes_val = float(str(votes_val).replace(',', ''))
        except Exception:
            votes_val = 1000.0
        # Popularity-style normalization: rating * log10(votes+1) scaled to ~[0,1]
        popularity = (rating_val * np.log10(votes_val + 1.0)) / 10.0
        qualities.append(float(np.clip(popularity, 0.0, 1.0)))
    return np.array(qualities)


@st.cache_data
def create_content_features(merged_df):
    """Create TF-IDF features using only title, genre, director, and rating tokens with weights"""

    genre_col = find_genre_column(merged_df)
    rating_col = find_rating_column(merged_df)
    director_col = find_director_column(merged_df)

    # Emphasize genre strongly; drop director; keep title minimal
    WEIGHTS = {
        'title': 1,
        'genre': 8,
        'director': 0,
        'rating': 2,
    }

    def build_row_text(row: pd.Series) -> str:
        title = str(row.get('Series_Title', '')).strip()
        genre = str(row.get(genre_col, '')).strip()
        director = str(row.get(director_col, '')).strip()

        # Rating bucket token (1..10) to inject numeric signal into TF-IDF
        rating_val = safe_convert_to_numeric(row.get(rating_col, np.nan), default=np.nan)
        if pd.isna(rating_val):
            rating_val = 7.0
        rating_bucket = int(max(1, min(10, round(rating_val))))
        rating_token = f"rating_{rating_bucket}"

        return ' '.join(
            [title] * WEIGHTS['title'] +
            [genre] * WEIGHTS['genre'] +
            [director] * WEIGHTS['director'] +
            [rating_token] * WEIGHTS['rating']
        )

    merged_df = merged_df.copy()
    merged_df['cb_text'] = merged_df.apply(build_row_text, axis=1)

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    return tfidf.fit_transform(merged_df['cb_text'])


@st.cache_data
def content_based_filtering_enhanced(merged_df, target_movie=None, genre=None, top_n=8):
    """Content-Based filtering using title/genre/rating tokens + cosine, blended with quality (rating & votes)."""
    if target_movie:
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if not similar_titles:
            return None
        
        target_title = similar_titles[0]
        
        # Ensure the target movie exists in the dataframe
        if target_title not in merged_df['Series_Title'].values:
            return None
            
        target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
        
        content_features = create_content_features(merged_df)
        target_features = content_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        similarities = cosine_similarity(target_features, content_features).flatten()
        
        # Blend cosine with normalized quality
        quality_vec = _compute_quality_vector(merged_df)
        sim_min, sim_max = float(similarities.min()), float(similarities.max())
        norm_sim = (similarities - sim_min) / (sim_max - sim_min + 1e-9)
        final_scores = 0.85 * norm_sim + 0.15 * quality_vec
        
        # Exclude the target itself and rank
        target_pos = merged_df.index.get_loc(target_idx)
        final_scores[target_pos] = -1.0
        ranked_indices = np.argsort(-final_scores)[:top_n]
        
        result_df = merged_df.iloc[ranked_indices]
        rating_col = find_rating_column(merged_df)
        genre_col = find_genre_column(merged_df)
        return result_df[['Series_Title', genre_col, rating_col]]
    
    elif genre:
        genre_col = find_genre_column(merged_df)
        rating_col = find_rating_column(merged_df)
        
        # Build genre-only TF-IDF for query matching
        genre_corpus = merged_df[genre_col].fillna('').tolist()
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(genre_corpus)
        query_vector = tfidf.transform([genre])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Blend cosine with normalized quality
        quality_vec = _compute_quality_vector(merged_df)
        sim_min, sim_max = float(similarities.min()), float(similarities.max())
        norm_sim = (similarities - sim_min) / (sim_max - sim_min + 1e-9)
        final_scores = 0.85 * norm_sim + 0.15 * quality_vec
        
        ranked_indices = np.argsort(-final_scores)[:top_n]
        result_df = merged_df.iloc[ranked_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
        
    return None