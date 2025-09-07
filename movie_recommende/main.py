# main.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
import os

# --- Algorithm Imports ---
# Corrected imports to match the function definitions in your files
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

# =========================
# Streamlit Page Configuration
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
# Data Loading Functions (from your code)
# =========================
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
        return None  # Return None on failure for the fallback to trigger
    except Exception as e:
        return None

@st.cache_data
def load_and_prepare_data():
    """Load CSVs from GitHub, prepare data, and handle merging."""
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    with st.spinner("Loading datasets from GitHub..."):
        movies_df = load_csv_from_github(github_base_url + "movies.csv", "movies.csv")
        imdb_df = load_csv_from_github(github_base_url + "imdb_top_1000.csv", "imdb_top_1000.csv")
        user_ratings_df = load_csv_from_github(github_base_url + "user_movie_rating.csv", "user_movie_rating.csv")
    
    if movies_df is None or imdb_df is None:
        return None, None, "GitHub loading failed."

    try:
        # Data cleaning and merging
        imdb_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        movies_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        merged_df = pd.merge(imdb_df, movies_df, on="Title", how="left")

        if 'Movie_ID' not in merged_df.columns and 'Movie_ID_y' in merged_df.columns:
             merged_df['Movie_ID'] = merged_df['Movie_ID_y']

        if 'Movie_ID' not in merged_df.columns:
            merged_df['Movie_ID'] = range(len(merged_df))
        
        merged_df.rename(columns={'Title': 'Series_Title'}, inplace=True)
        return merged_df, user_ratings_df, None
        
    except Exception as e:
        return None, None, f"Error merging datasets: {str(e)}"

@st.cache_data
def load_local_fallback():
    """Fallback to load local files if GitHub loading fails."""
    try:
        movies_df = pd.read_csv("movies.csv")
        imdb_df = pd.read_csv("imdb_top_1000.csv")
        user_ratings_df = pd.read_csv("user_movie_rating.csv")
        
        imdb_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        movies_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        merged_df = pd.merge(imdb_df, movies_df, on="Title", how="left")

        if 'Movie_ID' not in merged_df.columns and 'Movie_ID_y' in merged_df.columns:
             merged_df['Movie_ID'] = merged_df['Movie_ID_y']
        
        if 'Movie_ID' not in merged_df.columns:
            merged_df['Movie_ID'] = range(len(merged_df))
            
        merged_df.rename(columns={'Title': 'Series_Title'}, inplace=True)
        return merged_df, user_ratings_df, None
    except Exception as e:
        return None, None, str(e)

# =========================
# Helper and Display Functions (from your code)
# =========================
def get_recommendations_by_genre(df, genre, top_n=10):
    """Get top-rated movies for a given genre if no movie title is selected."""
    genre_col = 'Genre_x' if 'Genre_x' in df.columns else 'Genre'
    genre_df = df[df[genre_col].str.contains(genre, case=False, na=False)]
    return genre_df.sort_values(by="IMDB_Rating", ascending=False).head(top_n)

def display_movie_posters(results_df):
    """Display movie posters in a 5-column layout."""
    if results_df is None or results_df.empty:
        return
    
    st.subheader("🎬 Recommended Movies")
    movies_per_row = 5
    for i in range(0, len(results_df), movies_per_row):
        cols = st.columns(movies_per_row)
        row_movies = results_df.iloc[i:i + movies_per_row]
        
        for j, (_, movie) in enumerate(row_movies.iterrows()):
            with cols[j]:
                poster_url = movie.get('Poster_Link')
                if poster_url and pd.notna(poster_url):
                    st.image(poster_url, use_column_width=True)
                else:
                    st.markdown("<div style='height: 300px; display: flex; align-items: center; justify-content: center; background-color: #f0f2f6; border-radius: 8px;'><p>No Image</p></div>", unsafe_allow_html=True)
                
                st.markdown(f"**{movie['Series_Title'][:25]}{'...' if len(movie['Series_Title']) > 25 else ''}**")
                st.markdown(f"⭐ {movie.get('IMDB_Rating', 'N/A')}/10")

# =========================
# Main Application Logic
# =========================
def main():
    merged_df, user_ratings_df, error = load_and_prepare_data()
    
    if merged_df is None:
        st.warning("⚠️ GitHub loading failed, trying local files...")
        merged_df, user_ratings_df, local_error = load_local_fallback()
        if merged_df is None:
            st.error("❌ Could not load datasets from GitHub or local files.")
            with st.expander("🔍 Error Details"):
                st.write("**GitHub Error:**", error)
                st.write("**Local Error:**", local_error)
            st.stop()

    st.success("🎉 Datasets loaded successfully! Ready to recommend.")
    
    # --- Sidebar UI ---
    with st.sidebar:
        st.header("🎯 Recommendation Settings")
        
        all_movie_titles = sorted(merged_df['Series_Title'].dropna().unique())
        movie_title = st.selectbox("Select a Movie (Optional):", options=[""] + all_movie_titles)

        genre_col = 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
        all_genres = sorted(merged_df[genre_col].str.split(', ').explode().str.strip().dropna().unique())
        genre_input = st.selectbox("Select Genre (Optional):", options=[""] + all_genres)
        
        algorithm = st.selectbox("🔬 Choose Algorithm:", ["Hybrid", "Content-Based", "Collaborative"])
        top_n = st.slider("📊 Number of Recommendations:", 5, 20, 10)

        run_button = st.button("🚀 Generate Recommendations", type="primary")

    # --- Recommendation Logic ---
    if run_button:
        if not movie_title and not genre_input:
            st.error("❌ Please select a movie OR a genre to get recommendations.")
            return

        with st.spinner("🎬 Generating recommendations..."):
            results = pd.DataFrame()

            # --- CORE LOGIC: Handle different input combinations ---
            if movie_title:
                # User selected a movie, so we use the main algorithms
                if algorithm == "Content-Based":
                    results = content_based_filtering_enhanced(merged_df, movie_title, top_n)
                elif algorithm == "Collaborative":
                    if user_ratings_df is not None:
                        results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
                    else:
                        st.warning("⚠️ User rating data not available for Collaborative filtering.")
                elif algorithm == "Hybrid":
                     if user_ratings_df is not None:
                        results = smart_hybrid_recommendation(merged_df, user_ratings_df, movie_title, top_n)
                     else:
                        st.warning("⚠️ User rating data not available for Hybrid filtering.")

                # If a genre is ALSO selected, filter the results
                if genre_input and not results.empty:
                    results = results[results[genre_col].str.contains(genre_input, na=False, case=False)]

            elif genre_input and not movie_title:
                # User ONLY selected a genre, so we find top movies in that genre
                st.info(f"Showing top-rated movies for the genre: **{genre_input}**")
                results = get_recommendations_by_genre(merged_df, genre_input, top_n)

            # --- Display Results ---
            if not results.empty:
                display_movie_posters(results)
                
                with st.expander("📊 View Detailed Information", expanded=False):
                    display_cols = ['Series_Title', genre_col, 'IMDB_Rating', 'Released_Year', 'Director']
                    st.dataframe(results[display_cols].rename(columns={
                        'Series_Title': 'Title',
                        genre_col: 'Genre',
                        'IMDB_Rating': 'Rating',
                        'Released_Year': 'Year'
                    }), use_container_width=True)
            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")

if __name__ == "__main__":
    main()
