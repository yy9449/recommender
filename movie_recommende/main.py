# main.py
import streamlit as st
import pandas as pd
import warnings

# Import your recommendation algorithms
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
        
        # A simple merge based on title, assuming they are consistent
        # For a more robust system, a unique movie ID would be ideal
        merged_df = pd.merge(imdb_df, movies_df, on="Title", how="left")

        # Add a unique Movie_ID if it doesn't exist from the merge
        if 'Movie_ID' not in merged_df.columns:
            merged_df['Movie_ID'] = range(len(merged_df))
            
        # For simplicity, renaming 'Title' back to 'Series_Title' to match algorithm functions
        merged_df.rename(columns={'Title': 'Series_Title'}, inplace=True)


        return merged_df, user_ratings_df
    except FileNotFoundError as e:
        st.error(f"❌ Error loading data file: {e}. Please make sure the CSV files are in the correct directory.")
        return None, None

# --- Main Application ---
def main():
    """
    Renders the Streamlit UI and handles the recommendation logic.
    """
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")

    # Load the data
    merged_df, user_ratings_df = load_data()

    if merged_df is not None and user_ratings_df is not None:
        # --- Sidebar for User Input ---
        with st.sidebar:
            st.header("🔍 Select Your Preferences")

            # Algorithm selection
            algorithm = st.radio(
                "Choose a Recommendation Algorithm",
                ("Content-Based", "Collaborative", "Hybrid"),
                help="""
                - **Content-Based:** Recommends movies similar to the one you select.
                - **Collaborative:** Recommends movies based on user ratings.
                - **Hybrid:** A smart blend of multiple recommendation strategies.
                """
            )

            # Movie title selection
            movie_titles = sorted(merged_df['Series_Title'].unique())
            selected_movie = st.selectbox(
                "Select a Movie You Like",
                options=[""] + movie_titles,
                index=0
            )

            # Genre selection
            # Extracting unique genres can be complex if a movie has multiple genres
            # Here's a simple approach:
            all_genres = merged_df['Genre_x'].astype(str).str.split(', ').explode().unique()
            selected_genre = st.selectbox(
                "Select a Genre (Optional)",
                options=[""] + sorted(all_genres),
                index=0
            )

        # --- Recommendation Logic ---
        if selected_movie:
            st.subheader(f"Recommendations based on '{selected_movie}'")
            
            results_df = pd.DataFrame() # Initialize an empty dataframe for results

            if algorithm == "Content-Based":
                results_df = content_based_filtering_enhanced(merged_df, selected_movie)
            elif algorithm == "Collaborative":
                # Collaborative filtering requires both merged_df and user_ratings_df
                results_df = collaborative_filtering_enhanced(merged_df, user_ratings_df, selected_movie)
            elif algorithm == "Hybrid":
                # Hybrid filtering also uses both dataframes
                results_df = smart_hybrid_recommendation(merged_df, user_ratings_df, selected_movie)
            
            # --- Display Results ---
            if not results_df.empty:
                # Filter by genre if one is selected
                if selected_genre:
                    results_df = results_df[results_df['Genre'].str.contains(selected_genre, case=False, na=False)]

                if not results_df.empty:
                    st.success(f"🎉 Found {len(results_df)} recommendations for you!")
                    
                    # Display results in a more appealing way
                    for index, row in results_df.iterrows():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            # Using Poster_Link from the imdb_df part of the merge
                            st.image(row.get('Poster_Link', ''), use_column_width=True)
                        with col2:
                            st.subheader(row['Series_Title'])
                            st.write(f"**Genre:** {row.get('Genre', 'N/A')}")
                            st.write(f"**IMDb Rating:** {row.get('IMDB_Rating', 'N/A')} ⭐")
                            st.write(f"**Overview:** {row.get('Overview_x', 'N/A')}")
                            st.markdown("---")
                else:
                    st.warning("⚠️ No recommendations found matching your selected genre. Try another genre or leave it blank.")

            else:
                st.error("❌ Could not generate recommendations. The selected movie might not have enough data for this algorithm.")
        
        else:
            st.info("👋 Welcome! Please select a movie from the sidebar to get started.")

if __name__ == "__main__":
    main()
