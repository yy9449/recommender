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
    """Create TF-IDF features from tag tokens (genres, director, stars, certificate, time buckets)."""
    
    def normalize_name(text: str) -> str:
        return re.sub(r"\s+", "_", str(text).strip().lower())

    def build_tokens(row) -> str:
        tokens = []
        # Genres
        genre_val = row.get('Genre_y', row.get('Genre_x', row.get('Genre', '')))
        for g in normalize_genre_tokens(genre_val):
            tokens.append(f"genre_{g.replace(' ', '_')}")
        # Director
        director_val = row.get('Director_y', row.get('Director_x', row.get('Director', '')))
        if pd.notna(director_val) and str(director_val).strip():
            tokens.append(f"dir_{normalize_name(director_val)}")
        # Stars (collect unique)
        star_fields = ['Stars', 'Star1', 'Star2', 'Star3', 'Star4']
        star_set = set()
        for key in star_fields:
            val = row.get(key, '')
            if pd.notna(val) and str(val).strip():
                parts = [p.strip() for p in str(val).split(',') if p.strip()]
                for p in parts:
                    star_set.add(normalize_name(p))
        for s in list(star_set)[:6]:
            tokens.append(f"star_{s}")
        # Certificate
        cert = row.get('Certificate', '')
        if pd.notna(cert) and str(cert).strip():
            tokens.append(f"cert_{normalize_name(cert)}")
        # Decade
        y = row.get('Released_Year', row.get('Year', None))
        try:
            y = int(y) if pd.notna(y) else None
            if y and y > 1900:
                tokens.append(f"decade_{(y // 10) * 10}s")
        except Exception:
            pass
        # Runtime bucket
        runtime = row.get('Runtime', None)
        try:
            if isinstance(runtime, str):
                m = re.search(r'(\d+)', runtime)
                runtime_val = int(m.group(1)) if m else None
            else:
                runtime_val = int(runtime) if pd.notna(runtime) else None
            if runtime_val:
                if runtime_val < 90:
                    tokens.append('runtime_<90')
                elif runtime_val <= 120:
                    tokens.append('runtime_90_120')
                else:
                    tokens.append('runtime_>120')
        except Exception:
            pass
        return ' '.join(tokens)

    # Create a new column with combined features
    merged_df['combined_features'] = merged_df.apply(build_tokens, axis=1)
    
    tfidf = TfidfVectorizer(
        stop_words=None,
        ngram_range=(1, 1),
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents=None,
        lowercase=True
    )
    return tfidf.fit_transform(merged_df['combined_features'])


def normalize_genre_tokens(text):
    """Normalize genre labels to a consistent vocabulary."""
    raw = [g.strip().lower() for g in str(text).split(',') if g and g.strip()]
    norm = []
    for g in raw:
        s = g.replace('-', ' ').replace('_', ' ').strip()
        if s in ("sci fi", "science fiction", "scifi", "sci-fi"):
            s = "sci fi"
        if s in ("film noir",):
            s = "noir"
        if s in ("musical",):
            s = "music"
        if s in ("romcom", "rom com", "rom-com"):
            s = "rom com"
        if s in ("super hero", "super-hero", "superhero"):
            s = "superhero"
        if s in ("biopic",):
            s = "biography"
        norm.append(s)
    return norm


@st.cache_data
def create_genre_features(merged_df):
    """Create TF-IDF features using only normalized genres and return the fitted vectorizer too."""
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    vectorizer = TfidfVectorizer(
        tokenizer=normalize_genre_tokens,
        preprocessor=None,
        token_pattern=None,
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 2),
        use_idf=True,
        sublinear_tf=True,
        norm='l2',
        min_df=1,
        max_df=0.75
    )
    tfidf_matrix = vectorizer.fit_transform(merged_df[genre_col].fillna(''))
    return tfidf_matrix, vectorizer, genre_col


@st.cache_data
def create_genre_binary_features(merged_df):
    """Create binary one-hot matrix over normalized genres for basic content-based scoring."""
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    tokens_per_row = merged_df[genre_col].fillna('').apply(normalize_genre_tokens).tolist()
    # Build vocabulary
    vocab = {}
    for toks in tokens_per_row:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    if not vocab:
        # Fallback single dummy column
        mat = np.zeros((len(tokens_per_row), 1), dtype=float)
        return mat, tokens_per_row, genre_col, vocab
    mat = np.zeros((len(tokens_per_row), len(vocab)), dtype=float)
    for i, toks in enumerate(tokens_per_row):
        for t in toks:
            j = vocab.get(t)
            if j is not None:
                mat[i, j] = 1.0
    return mat, tokens_per_row, genre_col, vocab


@st.cache_data
def create_core_content_features(merged_df):
    """Create TF-IDF features from core tags: genres, director, and title tokens.

    This focuses on the essential content attributes requested to keep the model simple
    and accurate without adding extra feature families.
    """

    def normalize_name(text: str) -> str:
        return re.sub(r"\s+", "_", str(text).strip().lower())

    def title_tokens(title: str):
        txt = re.sub(r"[^a-z0-9\s]", " ", str(title).lower())
        toks = [t for t in txt.split() if t and len(t) >= 3]
        return [f"title_{t}" for t in toks[:8]]

    def build_core(row) -> str:
        tokens = []
        # Genres
        genre_val = row.get('Genre_y', row.get('Genre', ''))
        for g in normalize_genre_tokens(genre_val):
            tokens.append(f"genre_{g.replace(' ', '_')}")
        # Director
        director_val = row.get('Director_y', row.get('Director', ''))
        if pd.notna(director_val) and str(director_val).strip():
            tokens.append(f"dir_{normalize_name(director_val)}")
        # Title tokens
        title_val = row.get('Series_Title', '')
        tokens.extend(title_tokens(title_val))
        return ' '.join(tokens)

    docs = merged_df.apply(build_core, axis=1)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
        lowercase=True
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    return tfidf_matrix, vectorizer


@st.cache_data
def content_based_filtering_enhanced(merged_df, target_movie=None, genre=None, top_n=8):
    """Content-Based filtering using cosine similarity over core TF-IDF features.

    Core features: genres, director, title tokens. Light rating-aware re-ranking is applied.
    """
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'

    try:
        tfidf_matrix, vectorizer = create_core_content_features(merged_df)
    except Exception:
        return None

    # Normalized rating for re-ranking
    try:
        ratings = merged_df[rating_col].astype(float).fillna(0.0)
        max_r = float(ratings.max()) if len(ratings) else 10.0
        max_r = max(max_r, 1.0)
        norm_rating = (ratings / max_r).to_numpy()
    except Exception:
        norm_rating = np.zeros((len(merged_df),), dtype=float)

    # Also compute simple genre overlap to keep recommendations on-topic
    try:
        _, tokens_per_row, _, _ = create_genre_binary_features(merged_df)
    except Exception:
        tokens_per_row = [[] for _ in range(len(merged_df))]

    if target_movie:
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if not similar_titles:
            return None

        target_title = similar_titles[0]
        if target_title not in merged_df['Series_Title'].values:
            return None

        target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
        target_loc = merged_df.index.get_loc(target_idx)

        sims = cosine_similarity(tfidf_matrix[target_loc], tfidf_matrix).flatten()

        # Combine cosine with small genre Jaccard and rating
        target_set = set(tokens_per_row[target_loc]) if tokens_per_row else set()
        scored = []
        for idx in range(len(merged_df)):
            if idx == target_loc:
                continue
            cand_set = set(tokens_per_row[idx]) if tokens_per_row else set()
            if target_set and cand_set:
                inter = len(target_set & cand_set)
                union = len(target_set | cand_set)
                jacc = (inter / union) if union > 0 else 0.0
            else:
                jacc = 0.0
            # Require minimal overlap to avoid off-genre items
            if jacc <= 0.0:
                continue
            final_score = 0.70 * float(sims[idx]) + 0.20 * jacc + 0.10 * float(norm_rating[idx])
            scored.append((final_score, idx))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scored[:top_n]]
        result_df = merged_df.iloc[top_indices]
        cols = [c for c in ['Series_Title', genre_col, rating_col] if c in result_df.columns]
        return result_df[cols]

    elif genre:
        # Build a simple query vector from normalized genre tokens
        g_tokens = [f"genre_{g.replace(' ', '_')}" for g in normalize_genre_tokens(genre)]
        if not g_tokens:
            return None
        try:
            q_vec = vectorizer.transform([' '.join(g_tokens)])
        except Exception:
            return None
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

        # Combine with rating to prefer higher-rated items
        scored = []
        for idx in range(len(merged_df)):
            # Ensure at least one shared genre token
            cand_set = set(tokens_per_row[idx]) if tokens_per_row else set()
            jacc = 1.0 if set(normalize_genre_tokens(genre)) & cand_set else 0.0
            if jacc <= 0.0:
                continue
            final_score = 0.85 * float(sims[idx]) + 0.15 * float(norm_rating[idx])
            scored.append((final_score, idx))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scored[:top_n]]
        result_df = merged_df.iloc[top_indices]
        cols = [c for c in ['Series_Title', genre_col, rating_col] if c in result_df.columns]
        return result_df[cols]

    return None

