# main.py

import streamlit as st
import pandas as pd
import warnings
import requests
import io

# --- Attempt to Import All Recommendation Functions ---
# This allows the app to run even if some libraries are not installed on the server.

try:
    from content_based import content_based_filtering_enhanced
    CONTENT_AVAILABLE = True
except ImportError:
    CONTENT_AVAILABLE = False

try:
    from collaborative import collaborative_filtering_enhanced
    COLLABORATIVE_AVAILABLE = True
except ImportError:
    COLLABORATIVE_AVAILABLE = False

try:
    from hybrid import smart_hybrid_recommendation
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False


warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# --- Fallback Functions ---
# These are used if the main recommendation functions can't be imported.

def fallback_content(merged_df, target_movie, top_n=10):
    st.error("Content-Based model could not be loaded. Please check the file 'content_based.py'.")
    return pd.DataFrame()

def fallback_collaborative(merged_df, user_ratings_df, target_movie, top_n=10):
    st.warning("Collaborative model is running in a simplified fallback mode (based on popularity).", icon="⚠️")
    return merged_df.sort_values(by='IMDB_Rating', ascending=False).head(top_n)

def fallback_hybrid(merged_df, user_ratings_df, target_movie, top_n=10):
    st.warning("Hybrid model is running in a simplified fallback mode (content + popularity).", icon="⚠️")
    if CONTENT_AVAILABLE:
        content_recs = content_based_filtering_enhanced(merged_df, target_movie, top_n=50)
        return content_recs.sort_values(by='IMDB_Rating', ascending=False).head(top_n)
    return fallback_collaborative(merged_df, user_ratings_df, target_movie, top_n) # Fallback further if content is also missing

# --- Assign the correct functions to use ---
if not CONTENT_AVAILABLE: content_based_filtering_enhanced = fallback_content
if not COLLABORATIVE_AVAILABLE: collaborative_filtering_enhanced = fallback_collaborative
if not HYBRID_AVAILABLE: smart_hybrid_recommendation = fallback_hybrid


# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads and cleans data from your specific GitHub repository."""
    # This URL must point to the raw version of your CSV files on GitHub
    base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    try:
        movies_df = pd.read_csv(base_url + "movies.csv")
        imdb_df = pd.read_csv(base_url + "imdb_top_1000.csv")
        user_ratings_df = pd.read_csv(base_url + "user_movie_rating.csv")
        
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
            
        merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')

        # Robustly handle conflicting columns
        for col in ['Genre', 'Overview', 'Director']:
            col_x, col_y = f'{col}_x', f'{col}_y'
            if col_y in merged_df.columns and col_x in merged_df.columns:
                merged_df[col] = merged_df[col_y].fillna(merged_df[col_x])
        
        return merged_df, user_ratings_df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Fatal Error: Could not load data from GitHub. Please check the URL and ensure the repository is public. Details: {e}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None, None

# --- Main App Logic ---
merged_df, user_ratings_df = load_data()

if merged_df is not None:
    st.sidebar.header("Select Your Preferences")
    
    movie_list = sorted(merged_df['Series_Title'].dropna().unique())
    selected_movie = st.sidebar.selectbox("Choose a movie you like:", movie_list)

    # Only show algorithms that were successfully imported
    available_algorithms = []
    if HYBRID_AVAILABLE: available_algorithms.append('Hybrid')
    if CONTENT_AVAILABLE: available_algorithms.append('Content-Based')
    if COLLABORATIVE_AVAILABLE: available_algorithms.append('Collaborative')
    
    if not available_algorithms:
        st.error("No recommendation models could be loaded. Please check your Python files.")
    else:
        algorithm = st.sidebar.radio("Choose a recommendation algorithm:", available_algorithms)

        if st.sidebar.button("Get Recommendations"):
            with st.spinner("Finding recommendations..."):
                if algorithm == 'Content-Based':
                    recommendations = content_based_filtering_enhanced(merged_df, selected_movie)
                elif algorithm == 'Collaborative':
                    recommendations = collaborative_filtering_enhanced(merged_df, user_ratings_df, selected_movie)
                else: # Hybrid
                    recommendations = smart_hybrid_recommendation(merged_df, user_ratings_df, selected_movie)

                st.subheader(f"Recommendations based on '{selected_movie}' using {algorithm} Filtering:")
                
                if recommendations is not None and not recommendations.empty:
                    display_cols = ['Series_Title', 'Genre', 'IMDB_Rating']
                    existing_cols = [col for col in display_cols if col in recommendations.columns]
                    display_df = recommendations[existing_cols].reset_index(drop=True)
                    st.table(display_df)
                else:
                    st.warning("Could not find recommendations for this movie/algorithm combination.")