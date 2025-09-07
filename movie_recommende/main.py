# main.py

import streamlit as st
import pandas as pd
import warnings
import requests
import io

# --- A synchronized set of imports ---
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

# =========================
# Streamlit Page Configuration
# =========================
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
st.title("🎬 Movie Recommendation System")

# =========================
# Data Loading and Preparation
# =========================
@st.cache_data
def load_and_prepare_data():
    """
    Loads data from GitHub with a local fallback.
    This function now uses a definitive merge strategy to ensure column names are always predictable.
    """
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    try:
        # Load from GitHub
        imdb_df = pd.read_csv(github_base_url + "imdb_top_1000.csv")
        movies_df = pd.read_csv(github_base_url + "movies.csv")
        user_ratings_df = pd.read_csv(github_base_url + "user_movie_rating.csv")
        st.session_state['data_source'] = "GitHub"
    except Exception:
        st.warning("⚠️ GitHub loading failed. Attempting to load local files...")
        try:
            # Fallback to local
            imdb_df = pd.read_csv("imdb_top_1000.csv")
            movies_df = pd.read_csv("movies.csv")
            user_ratings_df = pd.read_csv("user_movie_rating.csv")
            st.session_state['data_source'] = "Local Files"
        except FileNotFoundError:
            st.error("❌ CRITICAL ERROR: Required CSV files not found. The app cannot run.")
            return None, None
            
    # --- The Definitive Merge Strategy ---
    # Merge imdb_df (left) with movies_df (right). This ensures the richer data from imdb_df 
    # consistently gets the '_x' suffix on overlapping columns (like Genre, Overview).
    # The other algorithm files are now built to expect exactly this.
    merged_df = pd.merge(imdb_df, movies_df, on="Series_Title", how="left", suffixes=('_x', '_y'))

    # Ensure a consistent Movie_ID for linking
    if 'Movie_ID' not in merged_df.columns:
        if 'Movie_ID_y' in merged_df.columns:
            # Prioritize the Movie_ID from movies.csv if it exists
            merged_df['Movie_ID'] = merged_df['Movie_ID_y']
        else:
             merged_df['Movie_ID'] = merged_df.index
    
    # Fill any missing Movie_IDs just in case
    merged_df['Movie_ID'] = merged_df['Movie_ID'].fillna(pd.Series(merged_df.index)).astype(int)

    return merged_df.drop_duplicates(subset=['Series_Title']).reset_index(drop=True), user_ratings_df

# =========================
# Main Application Logic
# =========================
def main():
    merged_df, user_ratings_df = load_and_prepare_data()

    if merged_df is None:
        st.stop()

    st.success(f"🎉 Datasets loaded successfully from {st.session_state.get('data_source', 'a source')}.")

    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("🎯 Recommendation Settings")

        movie_titles = sorted(merged_df['Series_Title'].dropna().unique())
        movie_title = st.selectbox("1. Select a Movie:", [""] + movie_titles)
        
        # This is now the definitive genre column the app will use
        genre_col = 'Genre_x' 
        all_genres = sorted(merged_df[genre_col].dropna().str.split(', ').explode().unique())
        genre_input = st.selectbox("2. Filter by Genre (Optional):", [""] + all_genres)
        
        algorithm = st.selectbox("3. Choose Algorithm:", ["Hybrid", "Content-Based", "Collaborative"])
        top_n = st.slider("4. Number of Recommendations:", 5, 20, 10)
        
        generate_button = st.button("🚀 Generate Recommendations", type="primary")

    # --- Recommendation and Display Logic ---
    if generate_button:
        if not movie_title:
            st.error("❌ Please select a movie from the dropdown to start.")
            return

        with st.spinner("Finding movies you'll love..."):
            results = pd.DataFrame()

            if algorithm == "Content-Based":
                results = content_based_filtering_enhanced(merged_df, movie_title, top_n)
            elif algorithm == "Collaborative":
                results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
            elif algorithm == "Hybrid":
                results = smart_hybrid_recommendation(merged_df, user_ratings_df, movie_title, top_n)

            # --- Display Results ---
            if results is not None and not results.empty:
                if genre_input:
                    # Filter results by the selected genre
                    results = results[results[genre_col].str.contains(genre_input, na=False, case=False)]

                if not results.empty:
                    st.subheader(f"Top {len(results)} Recommendations for '{movie_title}'")
                    # Display movie posters in a grid
                    for i in range(0, len(results), 5):
                        cols = st.columns(5)
                        for j, (_, row) in enumerate(results.iloc[i:i+5].iterrows()):
                            with cols[j]:
                                if pd.notna(row.get('Poster_Link')):
                                    st.image(row['Poster_Link'], use_column_width=True, caption=row['Series_Title'])
                                else:
                                    st.markdown(f"**{row['Series_Title']}**")
                                st.write(f"⭐ {row.get('IMDB_Rating', 'N/A')}")
                else:
                    st.warning(f"No recommendations for '{movie_title}' matched the genre '{genre_input}'. Try removing the genre filter.")
            else:
                st.error(f"Could not generate recommendations for '{movie_title}' with the {algorithm} algorithm. The movie might lack sufficient data.")

if __name__ == "__main__":
    main()
