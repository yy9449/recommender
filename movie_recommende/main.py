# main.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io

# --- Algorithm Imports ---
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================
# GitHub CSV Loading Functions (Your Code)
# ==================================
@st.cache_data
def load_csv_from_github(file_url, file_name):
    """Load CSV file from GitHub repository."""
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        csv_content = io.StringIO(response.text)
        df = pd.read_csv(csv_content)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to load {file_name} from GitHub: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing {file_name}: {str(e)}")
        return None

@st.cache_data
def load_and_prepare_data():
    """Load CSVs from GitHub and prepare data for recommendation algorithms."""
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    # File URLs
    movies_url = github_base_url + "movies.csv"
    imdb_url = github_base_url + "imdb_top_1000.csv"
    user_ratings_url = github_base_url + "user_movie_rating.csv"
    
    with st.spinner("Loading datasets from GitHub..."):
        movies_df = load_csv_from_github(movies_url, "movies.csv")
        imdb_df = load_csv_from_github(imdb_url, "imdb_top_1000.csv")
        user_ratings_df = load_csv_from_github(user_ratings_url, "user_movie_rating.csv")
    
    if movies_df is None or imdb_df is None:
        return None, None, "❌ Required CSV files could not be loaded. Please check the GitHub URLs and repository permissions."
    
    # Store user ratings in session state for other functions to access
    if user_ratings_df is not None:
        st.session_state['user_ratings_df'] = user_ratings_df
    
    try:
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
        
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
        merged_df = merged_df.drop_duplicates(subset="
