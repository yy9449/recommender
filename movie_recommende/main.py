import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced, load_user_ratings, diagnose_data_linking
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

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
        
        # Merge datasets with proper column handling
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how
