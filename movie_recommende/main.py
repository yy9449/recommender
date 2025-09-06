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
# Data Loading Functions
# =========================
@st.cache_data
def load_data(file_path, file_name):
    """Load a CSV file locally and display status."""
    try:
        df = pd.read_csv(file_path)
        # st.success(f"✅ Loaded {file_name} successfully!")
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: {file_name} not found at {file_path}. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading {file_name}: {e}")
        return None

# =========================
# Main Application Logic
# =========================
def main():
    """Main function to run the Streamlit app."""
    
    # Load all necessary datasets
    with st.spinner("Loading datasets..."):
        movies_df = load_data('movies.csv', 'Movies Dataset')
        imdb_df = load_data('imdb_top_1000.csv', 'IMDb Dataset')
        user_ratings_df = load_data('user_movie_rating.csv', 'User Ratings Dataset')

    if movies_df is None or imdb_df is None:
        st.error("❌ Critical data files could not be loaded. The application cannot proceed.")
        return

    # Merge datasets for a comprehensive view
    merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')
    
    # --- FIX: Ensure 'Movie_ID' exists and handle potential merge issues ---
    if 'Movie_ID' not in merged_df.columns:
        merged_df['Movie_ID'] = merged_df.index
    
    # Fill missing values to prevent errors
    for col in ['Genre_x', 'Genre_y', 'Overview_x', 'Overview_y', 'Director_x', 'Director_y']:
        if col not in merged_df.columns:
            merged_df[col] = '' # Add column if missing
            
    merged_df['Genre'] = merged_df['Genre_y'].fillna(merged_df['Genre_x'])
    merged_df['Overview'] = merged_df['Overview_y'].fillna(merged_df['Overview_x'])
    merged_df['Director'] = merged_df['Director_y'].fillna(merged_df['Director_x'])
    
    # Drop redundant columns after merging
    merged_df.drop(columns=['Genre_x', 'Genre_y', 'Overview_x', 'Overview_y', 'Director_x', 'Director_y'], inplace=True)
    
    # Sidebar for user inputs
    st.sidebar.header("🔍 Your Preferences")

    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Choose a Recommendation Algorithm",
        ("Content-Based", "Collaborative Filtering", "Hybrid"),
        help="**Content-Based**: Recommends movies similar to one you like. \n\n**Collaborative Filtering**: Recommends what similar users liked. \n\n**Hybrid**: Combines the best of both worlds!"
    )

    # Movie title input
    movie_list = merged_df['Series_Title'].unique()
    movie_title = st.sidebar.selectbox("Select a Movie You Like", options=movie_list, index=None, placeholder="Type or select a movie...")

    # Genre input
    unique_genres = sorted(list(set(genre for sublist in merged_df['Genre'].str.split(', ') for genre in sublist)))
    genre_input = st.sidebar.selectbox("Or, Pick a Genre", options=unique_genres, index=None, placeholder="Select a genre...")

    # Number of recommendations
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
            
            # --- ALGORITHM LOGIC ---
            if algorithm == "Content-Based":
                # Use either movie or genre for content-based
                if movie_title or genre_input:
                    results = content_based_filtering_enhanced(merged_df, movie_title, genre_input, top_n)
                else:
                    st.warning("⚠️ Please select a movie or genre for Content-Based filtering.")

            elif algorithm == "Collaborative Filtering":
                if movie_title:
                    if user_ratings_df is not None:
                         # Pass both dataframes to the function
                        results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
                    else:
                        st.error("❌ User ratings data is required for Collaborative Filtering but could not be loaded.")
                else:
                    st.warning("⚠️ Collaborative filtering requires a movie title input.")

            elif algorithm == "Hybrid":
                # --- FIX: HYBRID ALGORITHM CALL ---
                # 1. CHECK: Hybrid algorithm needs both a movie title and user ratings data
                if not movie_title:
                    st.error("❌ The Hybrid algorithm requires a movie title to be selected.")
                elif user_ratings_df is None:
                    st.error("❌ The Hybrid algorithm requires user ratings data, which could not be loaded.")
                else:
                    # 2. CORRECTED CALL: Pass the arguments in the correct order
                    results = smart_hybrid_recommendation(
                        merged_df, 
                        user_ratings_df, # Pass the user ratings DataFrame
                        movie_title,     # Pass the movie title
                        top_n
                    )
            
            # =========================
            # Display Results
            # =========================
            if results is not None and not results.empty:
                st.subheader("🎬 Recommended Movies")
                
                # Dynamically determine columns for display
                rating_col = 'IMDB_Rating' if 'IMDB_Rating' in results.columns else 'Rating'
                genre_col = 'Genre'
                title_col = 'Series_Title'
                poster_col = 'Poster_Link'

                # Display posters in columns
                num_cols = 5
                cols = st.columns(num_cols)
                for i, row in results.head(num_cols).iterrows():
                    with cols[i % num_cols]:
                        if poster_col in row and pd.notna(row[poster_col]):
                            st.image(row[poster_col], caption=f"{row[title_col]} ({row[rating_col]}⭐)", use_column_width=True)
                        else:
                            st.write(f"{row[title_col]} ({row[rating_col]}⭐)")
                
                # Expander for detailed view
                with st.expander("📊 View Detailed Information", expanded=False):
                    display_df = results.rename(columns={
                        title_col: 'Movie Title',
                        genre_col: 'Genre',
                        rating_col: 'IMDB Rating'
                    })
                    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
                    st.dataframe(
                        display_df[['Rank', 'Movie Title', 'Genre', 'IMDB Rating']],
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Analytics and insights section
                st.subheader("📈 Recommendation Insights")
                col1, col2, col3 = st.columns(3)
                
                avg_rating = results[rating_col].mean()
                col1.metric("Average Rating", f"{avg_rating:.1f}⭐")
                
                total_movies = len(results)
                col2.metric("Total Recommendations", total_movies)

                # Find the most common genre in the results
                genres_list = [g.strip() for sublist in results[genre_col].dropna().str.split(',') for g in sublist]
                if genres_list:
                    most_common_genre = pd.Series(genres_list).mode()[0]
                    col3.metric("Top Genre", most_common_genre)

            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")
                st.subheader("💡 Suggestions:")
                if movie_title and not genre_input:
                    st.write("- Try adding a genre preference.")
                    st.write("- Try a different algorithm.")
                elif genre_input and not movie_title:
                    st.write("- Try selecting a movie you like.")
                else:
                    st.write("- Check if the movie title is spelled correctly.")

if __name__ == "__main__":
    main()
