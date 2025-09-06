import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
try:
    from content_based import content_based_filtering_enhanced
    from collaborative import collaborative_filtering_enhanced
    from hybrid import smart_hybrid_recommendation
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please make sure all Python files are in the same directory and properly formatted.")
    st.stop()

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
    # Base URL for GitHub raw content
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    # Load data
    movies_df = load_csv_from_github(github_base_url + "movies.csv", "Movies Dataset")
    imdb_df = load_csv_from_github(github_base_url + "imdb_top_1000.csv", "IMDb Dataset")
    user_ratings_df = load_csv_from_github(github_base_url + "user_movie_rating.csv", "User Ratings Dataset")
    
    if movies_df is None or imdb_df is None:
        return None, None

    # Add Movie_ID to movies_df if it doesn't exist
    if 'Movie_ID' not in movies_df.columns:
        movies_df['Movie_ID'] = range(len(movies_df))

    # Merge dataframes
    merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')
    
    # Handle merged columns
    for col in ['Genre', 'Overview', 'Director']:
        col_x, col_y = f'{col}_x', f'{col}_y'
        if col_y in merged_df.columns and col_x in merged_df.columns:
            merged_df[col] = merged_df[col_y].fillna(merged_df[col_x])
            merged_df.drop(columns=[col_x, col_y], inplace=True)
        elif col_x in merged_df.columns:
            merged_df[col] = merged_df[col_x]
        elif col_y in merged_df.columns:
            merged_df[col] = merged_df[col_y]
            
    # Ensure Movie_ID exists
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
        
    # --- Sidebar UI ---
    st.sidebar.header("🔍 Your Preferences")
    
    algorithm = st.sidebar.selectbox(
        "Choose a Recommendation Algorithm",
        ("Content-Based", "Collaborative Filtering", "Hybrid")
    )
    
    movie_list = sorted(merged_df['Series_Title'].dropna().unique())
    movie_title = st.sidebar.selectbox("Select a Movie You Like", options=movie_list, index=None, placeholder="Type or select a movie...")
    
    # Handle genre extraction safely
    genre_list = []
    for genres in merged_df['Genre'].dropna():
        if isinstance(genres, str):
            genre_list.extend([g.strip() for g in genres.split(',')])
    unique_genres = sorted(list(set(genre_list))) if genre_list else []
    
    genre_input = None
    if unique_genres:
        genre_input = st.sidebar.selectbox("Or, Pick a Genre", options=unique_genres, index=None, placeholder="Select a genre...")

    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    # --- Recommendation Logic ---
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        
        # Input validation
        if not movie_title and not genre_input:
            st.error("❌ Please provide either a movie title or select a genre!")
            return
        
        results = None
        
        with st.spinner("🎬 Generating personalized recommendations..."):
            
            if algorithm == "Content-Based":
                if movie_title:
                    results = content_based_filtering_enhanced(merged_df, movie_title, top_n)
                else:
                    st.error("❌ Content-Based filtering requires a movie selection.")
            
            elif algorithm == "Collaborative Filtering":
                if movie_title and user_ratings_df is not None:
                    results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
                elif user_ratings_df is None:
                    st.error("❌ User ratings data could not be loaded.")
                else:
                    st.error("❌ Collaborative Filtering requires a movie selection.")

            elif algorithm == "Hybrid":
                if not movie_title:
                    st.error("❌ The Hybrid algorithm requires a movie title to be selected.")
                elif user_ratings_df is None:
                    st.error("❌ The Hybrid algorithm requires user ratings data, which could not be loaded.")
                else:
                    results = smart_hybrid_recommendation(
                        merged_df, 
                        user_ratings_df, 
                        movie_title, 
                        top_n
                    )
        
        # --- Display Results ---
        if results is not None and not results.empty:
            st.subheader("🎬 Recommended Movies")
            
            # Determine correct column names
            rating_col = 'IMDB_Rating' if 'IMDB_Rating' in results.columns else 'Rating'
            genre_col = 'Genre'
            
            # Display posters
            num_cols = 5
            cols = st.columns(num_cols)
            for i, (idx, row) in enumerate(results.head(num_cols).iterrows()):
                with cols[i % num_cols]:
                    if 'Poster_Link' in row and pd.notna(row['Poster_Link']):
                        try:
                            rating_value = row[rating_col] if pd.notna(row[rating_col]) else 0
                            st.image(row['Poster_Link'], caption=f"{row['Series_Title']} ({rating_value:.1f}⭐)", use_column_width=True)
                        except:
                            st.text(f"{row['Series_Title']}")
                    else:
                        st.text(f"{row['Series_Title']}")

            # Detailed information expander
            with st.expander("📊 View Detailed Information", expanded=False):
                # Create a clean dataframe for display
                display_results = results.copy()
                display_results = display_results.rename(columns={
                    'Series_Title': 'Movie Title',
                    genre_col: 'Genre',
                    rating_col: 'IMDB Rating'
                })
                
                # Add rank
                display_results.insert(0, 'Rank', range(1, len(display_results) + 1))
                
                # Add movie ID if available
                if 'Movie_ID' in results.columns:
                    display_results.insert(1, 'Movie ID', results['Movie_ID'])
                
                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn(format="%d"),
                        "Movie ID": st.column_config.NumberColumn(format="%d") if 'Movie_ID' in results.columns else None,
                        "IMDB Rating": st.column_config.NumberColumn(format="%.1f⭐")
                    }
                )
            
            # Analytics section
            st.subheader("📈 Recommendation Insights")
            col1, col2, col3, col4 = st.columns(4)
            
            # Metrics
            try:
                avg_rating = results[rating_col].mean()
                col1.metric("Average Rating", f"{avg_rating:.1f}⭐")
                col2.metric("Total Recommendations", len(results))
                max_rating = results[rating_col].max()
                col3.metric("Highest Rating", f"{max_rating:.1f}⭐")
                
                # Genre analysis
                if genre_col in results.columns:
                    genres_list = []
                    for genres in results[genre_col].dropna():
                        if isinstance(genres, str):
                            genres_list.extend([g.strip() for g in genres.split(',')])
                    
                    if genres_list:
                        most_common_genre = pd.Series(genres_list).mode().iloc[0] if len(pd.Series(genres_list).mode()) > 0 else "N/A"
                        col4.metric("Top Genre", most_common_genre)
                
                # Charts
                col1, col2 = st.columns(2)
                if genres_list:
                    with col1:
                        st.subheader("🎭 Genre Distribution")
                        genre_counts = pd.Series(genres_list).value_counts().head(8)
                        st.bar_chart(genre_counts)
                        
                with col2:
                    st.subheader("⭐ Rating Distribution")
                    valid_ratings = results[rating_col].dropna()
                    if len(valid_ratings) > 0:
                        rating_bins = pd.cut(valid_ratings, bins=5)
                        rating_dist = rating_bins.value_counts()
                        st.bar_chart(rating_dist)
                
            except Exception as e:
                st.warning(f"Could not generate all analytics: {str(e)}")
        
        else:
            st.error("❌ No recommendations found. Try different inputs or algorithms.")
            
            # Suggestions
            st.subheader("💡 Suggestions:")
            if movie_title and not genre_input:
                st.write("- Try adding a genre preference")
                st.write("- Try a different algorithm (Content-Based might work better)")
            elif genre_input and not movie_title:
                st.write("- Try selecting a movie you like")
                st.write("- Try a more common genre")
            else:
                st.write("- Check if the movie title is spelled correctly")
                st.write("- Try selecting from the dropdown instead of typing")

if __name__ == "__main__":
    main()
