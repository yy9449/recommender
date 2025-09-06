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
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True
    )
    return tfidf.fit_transform(merged_df['combined_features'])

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
        
        content_features = create_content_features(merged_df)
        target_features = content_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        similarities = cosine_similarity(target_features, content_features).flatten()
        
        # Take a larger candidate pool for re-ranking
        candidate_k = min(top_n * 5 + 1, len(similarities))
        candidate_indices = np.argsort(-similarities)[:candidate_k]
        candidate_indices = [idx for idx in candidate_indices if idx != merged_df.index.get_loc(target_idx)]
        
        # Prepare target movie attributes for boosting
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        director_col = 'Director_y' if 'Director_y' in merged_df.columns else 'Director'
        target_row = merged_df.loc[target_idx]
        target_genres = set([g.strip().lower() for g in str(target_row.get(genre_col, '')).split(',') if g])
        target_director = str(target_row.get(director_col, '')).strip().lower()
        # Collect all star tokens for the target
        def gather_stars(row):
            names = []
            for key in ['Stars', 'Star1', 'Star2', 'Star3', 'Star4']:
                val = row.get(key, '')
                if pd.notna(val) and str(val).strip():
                    names.extend([n.strip().lower() for n in str(val).split(',') if n.strip()])
            return set(names)
        target_stars = gather_stars(target_row)
        
        # Re-rank candidates with simple content-aware boosts
        scored = []
        for idx in candidate_indices:
            row = merged_df.iloc[idx]
            base_sim = float(similarities[idx])
            # Genre overlap ratio
            cand_genres = set([g.strip().lower() for g in str(row.get(genre_col, '')).split(',') if g])
            genre_overlap = len(target_genres & cand_genres)
            genre_ratio = genre_overlap / max(1, len(target_genres)) if target_genres else 0.0
            # Director match
            cand_director = str(row.get(director_col, '')).strip().lower()
            director_score = 1.0 if target_director and cand_director == target_director else 0.0
            # Star overlap ratio
            cand_stars = gather_stars(row)
            star_ratio = len(target_stars & cand_stars) / max(1, len(target_stars)) if target_stars else 0.0
            # Final boosted score (heavily weight cosine, light boosts for matches)
            final_score = 0.75 * base_sim + 0.15 * genre_ratio + 0.05 * director_score + 0.05 * star_ratio
            scored.append((final_score, idx))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scored[:top_n]]
        result_df = merged_df.iloc[top_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
    
    elif genre:
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        
        genre_corpus = merged_df[genre_col].fillna('').tolist()
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(genre_corpus)
        query_vector = tfidf.transform([genre])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        similar_indices = np.argsort(-similarities)[1:top_n+1]
        
        result_df = merged_df.iloc[similar_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
        
    return None