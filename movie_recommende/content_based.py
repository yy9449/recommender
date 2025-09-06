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
    """Create weighted TF-IDF features from multiple text fields"""
    
    # Clean and combine features with weights
    def combine_features(row):
        # Intelligently select the best available column after the merge
        overview = str(row.get('Overview_y', row.get('Overview_x', '')))
        genre = str(row.get('Genre_y', row.get('Genre_x', '')))
        director = str(row.get('Director_y', row.get('Director_x', '')))
        certificate = str(row.get('Certificate', ''))
        year = row.get('Released_Year', row.get('Year', None))
        runtime = row.get('Runtime', None)
        
        # Combine all available star information
        stars = str(row.get('Stars', '')) # from movies.csv
        star1 = str(row.get('Star1', '')) # from imdb_top_1000.csv
        star2 = str(row.get('Star2', ''))
        star3 = str(row.get('Star3', ''))
        star4 = str(row.get('Star4', ''))
        all_stars = ' '.join(filter(None, [stars, star1, star2, star3, star4]))
        
        # Ensure values are not NaN before joining
        overview = overview if pd.notna(overview) else ''
        genre = genre if pd.notna(genre) else ''
        director = director if pd.notna(director) else ''
        certificate = certificate if pd.notna(certificate) else ''

        # Derive decade token
        decade_token = ''
        try:
            y = int(year) if pd.notna(year) else None
            if y and y > 1900:
                decade = (y // 10) * 10
                decade_token = f"decade_{decade}s"
        except Exception:
            decade_token = ''

        # Derive runtime bucket token (if runtime provided as "123 min" or number)
        runtime_token = ''
        try:
            if isinstance(runtime, str):
                m = re.search(r'(\d+)', runtime)
                runtime_val = int(m.group(1)) if m else None
            else:
                runtime_val = int(runtime) if pd.notna(runtime) else None
            if runtime_val:
                if runtime_val < 90:
                    runtime_token = 'runtime_<90'
                elif runtime_val <= 120:
                    runtime_token = 'runtime_90_120'
                else:
                    runtime_token = 'runtime_>120'
        except Exception:
            runtime_token = ''

        # Apply weights
        tokens = []
        tokens.extend([overview] * 3)
        tokens.extend([genre] * 3)
        tokens.extend([director] * 2)
        tokens.extend([all_stars] * 2)
        if certificate:
            tokens.extend([f"cert_{certificate}"])
        if decade_token:
            tokens.append(decade_token)
        if runtime_token:
            tokens.append(runtime_token)
        return ' '.join(tokens)

    # Create a new column with combined features
    merged_df['combined_features'] = merged_df.apply(combine_features, axis=1)
    
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.80,
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True
    )
    return tfidf.fit_transform(merged_df['combined_features'])

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
        
        # Genre-only similarity
        genre_features, genre_col = create_genre_features(merged_df)
        target_vec = genre_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        sims = cosine_similarity(target_vec, genre_features).flatten()
        similar_indices = np.argsort(-sims)
        similar_indices = [idx for idx in similar_indices if idx != merged_df.index.get_loc(target_idx)]
        top_indices = similar_indices[:top_n]
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
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