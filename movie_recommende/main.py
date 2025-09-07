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
        merged_df = merged_df.drop_duplicates(subset="Series_Title").reset_index(drop=True)
        
        return merged_df, user_ratings_df, None
        
    except Exception as e:
        return None, None, f"❌ Error merging datasets: {str(e)}"

# =========================
# Main Application
# =========================
def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")

    merged_df, user_ratings_df, error_message = load_and_prepare_data()

    if error_message:
        st.error(error_message)
        return

    if merged_df is not None:
        with st.sidebar:
            st.header("🔍 Select Your Preferences")

            algorithm = st.radio(
                "Choose a Recommendation Algorithm",
                ("Content-Based", "Collaborative", "Hybrid")
            )

            movie_titles = sorted(merged_df['Series_Title'].unique())
            selected_movie = st.selectbox(
                "Select a Movie You Like",
                options=[""] + movie_titles
            )
            
            # Use 'Genre_x' as it is the column from the primary (movies_df) file after the merge
            genre_col_name = 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
            all_genres = sorted(merged_df[genre_col_name].astype(str).str.split(', ').explode().unique())
            selected_genre = st.selectbox(
                "Filter by Genre (Optional)",
                options=[""] + all_genres
            )

        if st.sidebar.button("Get Recommendations"):
            if selected_movie:
                with st.spinner('🤖 Generating recommendations...'):
                    results_df = pd.DataFrame() 

                    if algorithm == "Content-Based":
                        results_df = content_based_filtering_enhanced(merged_df, selected_movie)
                    
                    elif algorithm == "Collaborative":
                        results_df = collaborative_filtering_enhanced(merged_df, user_ratings_df, selected_movie)
                    
                    elif algorithm == "Hybrid":
                        results_df = smart_hybrid_recommendation(merged_df, user_ratings_df, selected_movie)
                    
                    # --- Display Results ---
                    if not results_df.empty:
                        if selected_genre:
                            display_genre_col = 'Genre_x' if 'Genre_x' in results_df.columns else 'Genre'
                            results_df = results_df[results_df[display_genre_col].str.contains(selected_genre, case=False, na=False)]

                        if not results_df.empty:
                            st.success(f"🎉 Found {len(results_df)} recommendations!")
                            for index, row in results_df.iterrows():
                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    st.image(row.get('Poster_Link', ''), use_column_width=True)
                                with col2:
                                    st.subheader(row['Series_Title'])
                                    st.write(f"**Genre:** {row.get('Genre_x', 'N/A')}")
                                    st.write(f"**IMDb Rating:** {row.get('IMDB_Rating', 'N/A')} ⭐")
                                    st.caption(f"**Overview:** {row.get('Overview_x', 'N/A')}")
                                st.markdown("---")
                        else:
                            st.warning(f"⚠️ No recommendations for '{selected_movie}' match the genre '{selected_genre}'.")
                    else:
                        st.error(f"❌ Could not generate recommendations for '{selected_movie}' with the '{algorithm}' algorithm.")
            else:
                st.warning("👈 Please select a movie to get recommendations.")
        else:
             st.info("👋 Select your preferences in the sidebar and click 'Get Recommendations'.")

if __name__ == "__main__":
    main()
