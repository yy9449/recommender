# main.py
import streamlit as st
import pandas as pd
import warnings

# --- Corrected Imports ---
# These now match the function definitions in your algorithm files.
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

# --- Data Loading ---
@st.cache_data
def load_data():
    """
    Loads, merges, and preprocesses the movie data from local CSV files.
    """
    try:
        # Load the datasets
        imdb_df = pd.read_csv("imdb_top_1000.csv")
        movies_df = pd.read_csv("movies.csv")
        user_ratings_df = pd.read_csv("user_movie_rating.csv")

        # Merge IMDb and movies data
        # Renaming to avoid column conflicts and ensure clarity
        imdb_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        movies_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        
        # A simple merge based on title
        merged_df = pd.merge(imdb_df, movies_df, on="Title", how="left")

        # Add a unique Movie_ID if it doesn't exist from the merge
        if 'Movie_ID' not in merged_df.columns:
            merged_df['Movie_ID'] = range(len(merged_df))
            
        # Rename 'Title' back to 'Series_Title' to match algorithm functions
        merged_df.rename(columns={'Title': 'Series_Title'}, inplace=True)

        return merged_df, user_ratings_df
    except FileNotFoundError as e:
        st.error(f"❌ Error loading data file: {e}. Please make sure the CSV files are in the correct directory.")
        return None, None

# --- Main Application ---
def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")

    merged_df, user_ratings_df = load_data()

    if merged_df is not None and user_ratings_df is not None:
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

            all_genres = sorted(merged_df['Genre_x'].astype(str).str.split(', ').explode().unique())
            selected_genre = st.selectbox(
                "Filter by Genre (Optional)",
                options=[""] + all_genres
            )

        # --- Corrected Recommendation Logic ---
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
                            genre_col = 'Genre_x' if 'Genre_x' in results_df.columns else 'Genre'
                            results_df = results_df[results_df[genre_col].str.contains(selected_genre, case=False, na=False)]

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
             st.info("👋 Select a movie and algorithm, then click 'Get Recommendations'.")

if __name__ == "__main__":
    main()
