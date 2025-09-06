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
# Streamlit Page Configuration
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Recommendation System")
st.markdown("Discover your next favorite movie with our smart recommendation engine!")
st.markdown("---")


# =========================
# GitHub CSV Loading Function (Your Original Method)
# =========================
@st.cache_data
def load_csv_from_github(file_url, file_name):
    """Load a single CSV file from a GitHub raw URL."""
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()  # This will raise an error for bad responses (404, 500, etc.)
        return pd.read_csv(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to load {file_name} from GitHub: {e}")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading {file_name}: {e}")
        return None

# =========================
# Main Application Logic
# =========================
def main():
    """Main function to run the Streamlit app."""

    # --- Define GitHub URLs ---
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    movies_url = github_base_url + "movies.csv"
    imdb_url = github_base_url + "imdb_top_1000.csv"
    user_ratings_url = github_base_url + "user_movie_rating.csv"
    
    # --- Load all datasets from GitHub ---
    with st.spinner("Loading datasets from GitHub..."):
        movies_df = load_csv_from_github(movies_url, 'Movies Dataset')
        imdb_df = load_csv_from_github(imdb_url, 'IMDb Dataset')
        user_ratings_df = load_csv_from_github(user_ratings_url, 'User Ratings Dataset')

    if movies_df is None or imdb_df is None:
        st.error("❌ Critical data files could not be loaded. The application cannot proceed.")
        st.stop() # Stop the app if essential data is missing

    # --- Merge and Prepare Data ---
    merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')
    
    if 'Movie_ID' not in merged_df.columns:
        merged_df['Movie_ID'] = merged_df.index
    
    # Clean up merged columns to create a single, reliable source for Genre, Overview, etc.
    for col in ['Genre', 'Overview', 'Director']:
        col_x, col_y = f'{col}_x', f'{col}_y'
        if col_y in merged_df.columns and col_x in merged_df.columns:
            merged_df[col] = merged_df[col_y].fillna(merged_df[col_x])
            merged_df.drop(columns=[col_x, col_y], inplace=True)

    # --- Sidebar for User Inputs ---
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

    # =========================
    # Recommendation Generation
    # =========================
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        if not movie_title and not genre_input:
            st.error("❌ Please provide either a movie title or select a genre!")
            return
        
        with st.spinner("🎬 Generating personalized recommendations..."):
            results = None
            
            if algorithm == "Content-Based":
                results = content_based_filtering_enhanced(merged_df, movie_title, genre_input, top_n)
            
            elif algorithm == "Collaborative Filtering":
                if not movie_title:
                    st.warning("⚠️ Collaborative filtering requires a movie title input.")
                elif user_ratings_df is None:
                    st.error("❌ User ratings data is required but could not be loaded.")
                else:
                    results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)

            elif algorithm == "Hybrid":
                if not movie_title:
                    st.error("❌ The Hybrid algorithm requires a movie title to be selected.")
                elif user_ratings_df is None:
                    st.error("❌ The Hybrid algorithm requires user ratings data, which could not be loaded.")
                else:
                    results = smart_hybrid_recommendation(merged_df, user_ratings_df, movie_title, top_n)
            
            # --- Display Results ---
            if results is not None and not results.empty:
                st.subheader("🎬 Recommended Movies")
                
                rating_col = 'IMDB_Rating'
                title_col = 'Series_Title'
                poster_col = 'Poster_Link'

                num_cols = 5
                cols = st.columns(num_cols)
                for i, row in results.head(num_cols).iterrows():
                    with cols[i % num_cols]:
                        if poster_col in row and pd.notna(row[poster_col]):
                            st.image(row[poster_col], caption=f"{row[title_col]} ({row[rating_col]:.1f}⭐)", use_column_width=True)
            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")

if __name__ == "__main__":
    main()
