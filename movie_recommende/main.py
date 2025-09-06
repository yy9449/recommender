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
# Data Loading & Preparation from GitHub
# =========================
@st.cache_data
def load_and_prepare_data():
    """Load CSVs from GitHub and prepare data for recommendation algorithms."""
    
    # Base URL for the raw GitHub content
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    # Full URLs for each file
    files_to_load = {
        "movies_df": (github_base_url + "movies.csv", "Movies Dataset"),
        "imdb_df": (github_base_url + "imdb_top_1000.csv", "IMDb Dataset"),
        "user_ratings_df": (github_base_url + "user_movie_rating.csv", "User Ratings Dataset")
    }
    
    dataframes = {}
    
    # Function to fetch and load a single file
    def load_csv_from_url(url, name):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            return pd.read_csv(io.StringIO(response.text))
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to load {name} from GitHub: {e}")
            return None
        except Exception as e:
            st.error(f"❌ An error occurred while processing {name}: {e}")
            return None

    # Load all dataframes
    for key, (url, name) in files_to_load.items():
        dataframes[key] = load_csv_from_url(url, name)

    movies_df = dataframes.get("movies_df")
    imdb_df = dataframes.get("imdb_df")
    user_ratings_df = dataframes.get("user_ratings_df")

    # --- Data Merging and Cleaning ---
    if movies_df is None or imdb_df is None:
        return None, None, None # Return None if critical files fail to load
        
    merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')
    
    if 'Movie_ID' not in merged_df.columns:
        merged_df['Movie_ID'] = merged_df.index
    
    for col in ['Genre_x', 'Genre_y', 'Overview_x', 'Overview_y', 'Director_x', 'Director_y']:
        if col not in merged_df.columns:
            merged_df[col] = ''
            
    merged_df['Genre'] = merged_df['Genre_y'].fillna(merged_df['Genre_x'])
    merged_df['Overview'] = merged_df['Overview_y'].fillna(merged_df['Overview_x'])
    merged_df['Director'] = merged_df['Director_y'].fillna(merged_df['Director_x'])
    
    merged_df.drop(columns=['Genre_x', 'Genre_y', 'Overview_x', 'Overview_y', 'Director_x', 'Director_y'], inplace=True, errors='ignore')
    
    return merged_df, user_ratings_df, imdb_df


# =========================
# Main Application Logic
# =========================
def main():
    """Main function to run the Streamlit app."""
    
    # Load all necessary datasets from GitHub
    with st.spinner("Loading datasets from GitHub..."):
        merged_df, user_ratings_df, imdb_df = load_and_prepare_data()

    if merged_df is None:
        st.error("❌ Critical data files could not be loaded. The application cannot proceed.")
        return

    # Sidebar for user inputs
    st.sidebar.header("🔍 Your Preferences")
    algorithm = st.sidebar.selectbox(
        "Choose a Recommendation Algorithm",
        ("Content-Based", "Collaborative Filtering", "Hybrid")
    )
    movie_list = merged_df['Series_Title'].unique()
    movie_title = st.sidebar.selectbox("Select a Movie You Like", options=movie_list, index=None, placeholder="Type or select a movie...")
    unique_genres = sorted(list(set(genre for sublist in merged_df['Genre'].str.split(', ') for genre in sublist)))
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
                if movie_title:
                    if user_ratings_df is not None:
                        results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
                    else:
                        st.error("❌ User ratings data is required for this algorithm but could not be loaded.")
                else:
                    st.warning("⚠️ Collaborative filtering requires a movie title input.")

            elif algorithm == "Hybrid":
                if not movie_title:
                    st.error("❌ The Hybrid algorithm requires a movie title to be selected.")
                elif user_ratings_df is None:
                    st.error("❌ The Hybrid algorithm requires user ratings data, which could not be loaded.")
                else:
                    results = smart_hybrid_recommendation(merged_df, user_ratings_df, movie_title, top_n)
            
            # =========================
            # Display Results
            # =========================
            if results is not None and not results.empty:
                st.subheader("🎬 Recommended Movies")
                
                rating_col = 'IMDB_Rating' if 'IMDB_Rating' in results.columns else 'Rating'
                genre_col = 'Genre'
                title_col = 'Series_Title'
                poster_col = 'Poster_Link'

                num_cols = 5
                cols = st.columns(num_cols)
                for i, row in results.head(num_cols).iterrows():
                    with cols[i % num_cols]:
                        if poster_col in row and pd.notna(row[poster_col]):
                            st.image(row[poster_col], caption=f"{row[title_col]} ({row[rating_col]:.1f}⭐)", use_column_width=True)
                        else:
                            st.write(f"{row[title_col]} ({row[rating_col]:.1f}⭐)")
                
                with st.expander("📊 View Detailed Information", expanded=False):
                    display_df = results.rename(columns={
                        title_col: 'Movie Title', genre_col: 'Genre', rating_col: 'IMDB Rating'
                    })
                    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
                    st.dataframe(display_df[['Rank', 'Movie Title', 'Genre', 'IMDB Rating']], use_container_width=True, hide_index=True)
                
            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")

if __name__ == "__main__":
    main()
