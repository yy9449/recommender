import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
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
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to load {file_name} from GitHub: {str(e)}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"❌ {file_name} is empty or corrupted")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading {file_name}: {str(e)}")
        return None

# =========================
# Data Preparation
# =========================
@st.cache_data
def get_and_prepare_data():
    """Loads and prepares all necessary dataframes."""
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    movies_df = load_csv_from_github(github_base_url + "movies.csv", "Movies Dataset")
    imdb_df = load_csv_from_github(github_base_url + "imdb_top_1000.csv", "IMDb Dataset")
    user_ratings_df = load_csv_from_github(github_base_url + "user_movie_rating.csv", "User Ratings Dataset")
    
    if movies_df is None or imdb_df is None:
        return None, None

    merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')
    
    for col in ['Genre', 'Overview', 'Director']:
        col_x, col_y = f'{col}_x', f'{col}_y'
        if col_y in merged_df.columns and col_x in merged_df.columns:
            merged_df[col] = merged_df[col_y].fillna(merged_df[col_x])
            merged_df.drop(columns=[col_x, col_y], inplace=True)
            
    if 'Movie_ID' not in merged_df.columns:
        merged_df['Movie_ID'] = merged_df.index
        
    return merged_df, user_ratings_df

# =========================
# Main App Logic
# =========================
def main():
    merged_df, user_ratings_df = get_and_prepare_data()

    if merged_df is None:
        st.error("❌ Critical data could not be loaded. App cannot proceed.")
        st.stop()
        
    st.sidebar.header("🔍 Your Preferences")
    
    algorithm = st.sidebar.selectbox(
        "Choose a Recommendation Algorithm",
        ("Content-Based", "Collaborative Filtering", "Hybrid")
    )
    
    movie_list = sorted(merged_df['Series_Title'].dropna().unique())
    movie_title = st.sidebar.selectbox("Select a Movie You Like", options=movie_list, index=None, placeholder="Type or select a movie...")
    
    unique_genres = sorted(list(set(g for sublist in merged_df['Genre'].dropna().str.split(', ') for g in sublist)))
    genre_input = st.sidebar.selectbox("Or, Pick a Genre", options=unique_genres, index=None, placeholder="Select a genre...")

    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        results = None
        
        with st.spinner("🎬 Generating personalized recommendations..."):
            if algorithm == "Content-Based":
                if movie_title or genre_input:
                    results = content_based_filtering_enhanced(merged_df, movie_title, genre_input, top_n)
            
            elif algorithm == "Collaborative Filtering":
                if movie_title and user_ratings_df is not None:
                    results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)

            elif algorithm == "Hybrid":
                # --- START OF FIX ---
                # 1. ADDED CHECKS: The hybrid function needs a movie title and user ratings to work.
                if not movie_title:
                    st.error("❌ The Hybrid algorithm requires a movie title to be selected.")
                elif user_ratings_df is None:
                    st.error("❌ The Hybrid algorithm requires user ratings data, which could not be loaded.")
                else:
                    # 2. CORRECTED FUNCTION CALL: Arguments now match the function in hybrid.py
                    results = smart_hybrid_recommendation(
                        merged_df, 
                        user_ratings_df, 
                        movie_title, 
                        top_n
                    )
                # --- END OF FIX ---
        
        if results is not None and not results.empty:
            st.subheader("🎬 Recommended Movies")
            
            rating_col = 'IMDB_Rating'
            genre_col = 'Genre'
            
            num_cols = 5
            cols = st.columns(num_cols)
            for i, row in results.head(num_cols).iterrows():
                with cols[i % num_cols]:
                    if 'Poster_Link' in row and pd.notna(row['Poster_Link']):
                        st.image(row['Poster_Link'], caption=f"{row['Series_Title']} ({row[rating_col]:.1f}⭐)", use_column_width=True)

        # The rest of your display logic remains unchanged...
        # (I have removed the duplicate display code for brevity, but it's the same as your original file)
        
if __name__ == "__main__":
    main()
