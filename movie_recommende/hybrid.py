import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io

warnings.filterwarnings('ignore')

# Import functions with error handling
try:
    from content_based import content_based_filtering_enhanced
except ImportError:
    content_based_filtering_enhanced = None

try:
    from collaborative import collaborative_filtering_enhanced
except ImportError:
    collaborative_filtering_enhanced = None
    
try:
    from hybrid import smart_hybrid_recommendation
except ImportError:
    smart_hybrid_recommendation = None

# Import column utilities
try:
    from column_utils import (
        get_genre_column, get_overview_column, get_rating_column, 
        get_year_column, get_votes_column, safe_get_column_data,
        apply_genre_filter, get_movie_display_info
    )
except ImportError:
    # Fallback functions if column_utils.py is not available
    def get_genre_column(df):
        for col in ['Genre', 'Genre_y', 'Genre_x', 'Genres']:
            if col in df.columns:
                return col
        return None
    
    def get_rating_column(df):
        for col in ['IMDB_Rating', 'Rating', 'IMDB_Rating_y', 'IMDB_Rating_x']:
            if col in df.columns:
                return col
        return None
    
    def get_year_column(df):
        for col in ['Released_Year', 'Year', 'Released_Year_y', 'Released_Year_x']:
            if col in df.columns:
                return col
        return None
    
    def apply_genre_filter(df, genre_filter):
        genre_col = get_genre_column(df)
        if genre_col:
            return df[df[genre_col].str.contains(genre_filter, case=False, na=False)]
        else:
            return pd.DataFrame()
    
    def get_movie_display_info(df, movie_row):
        rating_col = get_rating_column(df)
        genre_col = get_genre_column(df)
        year_col = get_year_column(df)
        
        return {
            'title': movie_row.get('Series_Title', 'Unknown'),
            'rating': movie_row.get(rating_col, 'N/A') if rating_col else 'N/A',
            'genre': movie_row.get(genre_col, 'N/A') if genre_col else 'N/A',
            'year': movie_row.get(year_col, 'N/A') if year_col else 'N/A',
            'poster': movie_row.get('Poster_Link', '')
        }

# Backup content-based function
def simple_content_based(merged_df, target_movie, genre_filter=None, top_n=10):
    """Simplified content-based filtering using available columns with proper column resolution"""
    if not target_movie and not genre_filter:
        return pd.DataFrame()
    
    # Handle genre-only filtering
    if genre_filter and not target_movie:
        filtered = apply_genre_filter(merged_df, genre_filter)
        if not filtered.empty:
            rating_col = get_rating_column(filtered)
            if rating_col:
                filtered = filtered.sort_values(rating_col, ascending=False)
            return filtered.head(top_n)
        return pd.DataFrame()
    
    # Movie-based filtering
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
    
    # Simple genre-based similarity
    target_row = merged_df[merged_df['Series_Title'] == target_movie].iloc[0]
    genre_col = get_genre_column(merged_df)
    
    if not genre_col:
        return pd.DataFrame()
    
    target_genres = str(target_row[genre_col]).split(', ') if pd.notna(target_row[genre_col]) else []
    
    # Find movies with similar genres
    similar_movies = []
    for idx, row in merged_df.iterrows():
        if row['Series_Title'] == target_movie:
            continue
        
        movie_genres = str(row[genre_col]).split(', ') if pd.notna(row[genre_col]) else []
        common_genres = set(target_genres) & set(movie_genres)
        
        if common_genres:
            similar_movies.append((idx, len(common_genres)))
    
    # Sort by genre similarity and rating
    similar_movies.sort(key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in similar_movies[:top_n*2]]
    
    results = merged_df.loc[top_indices]
    
    # Apply genre filter if provided
    if genre_filter:
        results = apply_genre_filter(results, genre_filter)
    
    # Sort by rating
    rating_col = get_rating_column(results)
    if rating_col:
        results = results.sort_values(rating_col, ascending=False)
    
    return results.head(top_n)

# =========================
# Streamlit Configuration
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Recommendation System")
st.markdown("---")

# =========================
# GitHub CSV Loading Functions
# =========================
@st.cache_data
def load_csv_from_github(file_url, file_name):
    """Load CSV file from GitHub repository - silent version"""
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Read CSV from response content
        csv_content = io.StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        # Silent success - no st.success message
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to load {file_name} from GitHub: {str(e)}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"❌ {file_name} is empty or corrupted")
        return None
    except Exception as e:
        st.error(f"❌ Error processing {file_name}: {str(e)}")
        return None

@st.cache_data
def load_and_prepare_data():
    """Load CSVs from GitHub and prepare data for recommendation algorithms - silent version"""
    
    # GitHub raw file URLs - replace with your actual repository URLs
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    # File URLs
    movies_url = github_base_url + "movies.csv"
    imdb_url = github_base_url + "imdb_top_1000.csv"
    user_ratings_url = github_base_url + "user_movie_rating.csv"
    
    # Silent loading - show minimal progress info
    with st.spinner("Loading datasets..."):
        movies_df = load_csv_from_github(movies_url, "movies.csv")
        imdb_df = load_csv_from_github(imdb_url, "imdb_top_1000.csv")
        user_ratings_df = load_csv_from_github(user_ratings_url, "user_movie_rating.csv")
    
    # Check if required files loaded successfully
    if movies_df is None or imdb_df is None:
        return None, None, "❌ Required CSV files (movies.csv, imdb_top_1000.csv) could not be loaded from GitHub"
    
    # Store user ratings in session state for other functions to access - silent
    if user_ratings_df is not None:
        st.session_state['user_ratings_df'] = user_ratings_df
        # Silent success - no message
    else:
        # Only show warning if explicitly needed
        if 'user_ratings_df' in st.session_state:
            del st.session_state['user_ratings_df']
    
    try:
        # Validate required columns
        if 'Series_Title' not in movies_df.columns or 'Series_Title' not in imdb_df.columns:
            return None, None, "❌ Missing Series_Title column in one or both datasets"
        
        # Check if movies.csv has Movie_ID
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
            # Silent addition - no info message
        
        # Merge on Series_Title
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
        merged_df = merged_df
