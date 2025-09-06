import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
import os

# Ignore common warnings
warnings.filterwarnings('ignore')

# --- Import custom recommendation functions with error handling ---
# This allows the app to run even if a file is missing, using fallbacks.
try:
    from content_based import content_based_filtering_enhanced
except ImportError:
    st.error("Warning: `content_based.py` not found. Content-Based features will be limited.")
    content_based_filtering_enhanced = None

try:
    from collaborative import collaborative_filtering_enhanced
except ImportError:
    st.warning("Warning: `collaborative.py` not found. Collaborative features will be disabled.")
    collaborative_filtering_enhanced = None
    
try:
    from hybrid import smart_hybrid_recommendation
except ImportError:
    st.error("Warning: `hybrid.py` not found. Hybrid features will be limited.")
    smart_hybrid_recommendation = None

# --- Fallback Recommendation Function ---
def simple_fallback_content_based(merged_df, target_movie=None, genre_filter=None, top_n=10):
    """
    A reliable fallback content-based recommender.
    Uses a simplified TF-IDF on overview and genre if the main functions fail.
    Handles various column names (e.g., 'Genre_y', 'Overview_x').
    """
    if not target_movie and not genre_filter:
        return pd.DataFrame()

    # Determine correct column names after potential merges
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
    overview_col = 'Overview_y' if 'Overview_y' in merged_df.columns else 'Overview_x' if 'Overview_x' in merged_df.columns else 'Overview'
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'

    # Handle genre-only filtering
    if genre_filter and not target_movie:
        if genre_col in merged_df.columns:
            results = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
            if rating_col in results.columns:
                return results.sort_values(rating_col, ascending=False).head(top_n)
            return results.head(top_n)
        return pd.DataFrame() # Return empty if no genre column

    if not target_movie or target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
        
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Create a "soup" of text from available columns
        soup = merged_df[overview_col].fillna('') + ' ' + merged_df[genre_col].fillna('')
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(soup)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N results, allowing for genre filtering
        sim_scores = sim_scores[1:(top_n * 2) + 1]
        movie_indices = [i[0] for i in sim_scores]
        results = merged_df.iloc[movie_indices]
        
        # Apply genre filter if provided
        if genre_filter and genre_col in results.columns:
            results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
            
        return results.head(top_n)
    except Exception:
        # If TF-IDF fails for any reason, return empty
        return pd.DataFrame()

# --- Data Loading ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data():
    """Loads, merges, and preprocesses the movie datasets from local files."""
    try:
        movies_df = pd.read_csv("movies.csv")
        imdb_df = pd.read_csv("imdb_top_1000.csv")
        user_ratings_df = pd.read_csv("user_movie_rating.csv")
        
        # Merge dataframes on the movie title
        merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='inner')
        
        # --- Data Cleaning and Preparation ---
        # Clean runtime column
        if 'Runtime' in merged_df.columns and merged_df['Runtime'].dtype == 'object':
            merged_df['Runtime'] = merged_df['Runtime'].str.replace(' min', '', regex=False).astype(float)
        
        # Create a unified list of unique genres for the dropdown selector
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x'
        if genre_col in merged_df.columns:
            all_genres = merged_df[genre_col].dropna().str.split(', ').explode()
            unique_genres = sorted(all_genres.unique())
        else:
            unique_genres = [] # No genre column found
        
        return merged_df, user_ratings_df, unique_genres
    except FileNotFoundError as e:
        st.error(f"❌ Critical Error: A required data file was not found. Please ensure all CSV files are in the same directory. Details: {e}")
        return None, None, None

# --- Main Application UI ---
def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎬")
    st.title("🎬 Movie Recommendation System")

    merged_df, user_ratings_df, unique_genres = load_data()

    if merged_df is None:
        st.info("Please add `movies.csv`, `imdb_top_1000.csv`, and `user_movie_rating.csv` to the directory and refresh.")
        st.stop()

    # --- Sidebar for User Input ---
    st.sidebar.header("🔍 Configure Recommendations")
    
    # Select algorithm
    algorithm = st.sidebar.selectbox(
        "Choose a Recommendation Algorithm",
        ("Hybrid", "Content-Based", "Collaborative"),
        help="**Hybrid**: A smart blend of all methods. **Content-Based**: Recommends movies similar in plot and genre. **Collaborative**: Recommends movies liked by similar users (requires user data)."
    )

    # Select movie
    movie_title = st.sidebar.selectbox(
        "Select a Movie (Optional)",
        options=[''] + sorted(merged_df['Series_Title'].dropna().unique()),
        help="Pick a movie to base recommendations on."
    )

    # Select genre
    genre_input = st.sidebar.selectbox(
        "Filter by Genre (Optional)",
        options=[''] + unique_genres,
        help="Narrow down recommendations to a specific genre."
    )
    
    # Select number of recommendations
    top_n = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5, max_value=20, value=10
    )

    # --- Recommendation Generation Logic ---
    if st.sidebar.button("✨ Generate Recommendations", type="primary"):
        if not movie_title and not genre_input:
            st.warning("⚠️ Please select a movie or a genre to get recommendations.")
        else:
            results = pd.DataFrame()
            
            with st.spinner(f"Finding recommendations with {algorithm} algorithm..."):
                try:
                    # --- Content-Based Logic ---
                    if algorithm == "Content-Based":
                        if content_based_filtering_enhanced:
                            # CORRECTED: Pass arguments by keyword for clarity and safety.
                            results = content_based_filtering_enhanced(
                                merged_df=merged_df,
                                target_movie=movie_title if movie_title else None,
                                genre_filter=genre_input if genre_input else None,
                                top_n=top_n
                            )
                        else:
                             st.error("Content-based module is unavailable. Using fallback.")
                             results = simple_fallback_content_based(merged_df, movie_title, genre_input, top_n)
                    
                    # --- Collaborative Logic ---
                    elif algorithm == "Collaborative":
                        if not movie_title:
                            st.warning("⚠️ Collaborative filtering requires selecting a movie.")
                        elif collaborative_filtering_enhanced and user_ratings_df is not None:
                             results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
                        else:
                            st.error("Collaborative module or user data is unavailable. Please select another algorithm.")

                    # --- Hybrid Logic (Default) ---
                    else:
                        if smart_hybrid_recommendation and user_ratings_df is not None:
                            # CORRECTED: Pass all arguments by keyword.
                            results = smart_hybrid_recommendation(
                                merged_df=merged_df,
                                user_ratings_df=user_ratings_df,
                                target_movie=movie_title if movie_title else None,
                                genre_filter=genre_input if genre_input else None,
                                top_n=top_n
                            )
                        else:
                            st.info("Hybrid module or user data not available. Falling back to content-based recommendation.")
                            results = simple_fallback_content_based(merged_df, movie_title, genre_input, top_n)

                except Exception as e:
                    st.error(f"An unexpected error occurred with {algorithm} filtering: {e}")
                    st.info("Falling back to a simpler recommendation method.")
                    results = simple_fallback_content_based(merged_df, movie_title, genre_input, top_n)

            # --- Display Results ---
            st.markdown("---")
            if not results.empty:
                st.subheader(f"🏆 Top {len(results)} Recommendations")
                
                # Dynamically find column names for display consistency
                poster_col = 'Poster_Link'
                genre_col = 'Genre_y' if 'Genre_y' in results.columns else 'Genre_x'
                overview_col = 'Overview_y' if 'Overview_y' in results.columns else 'Overview_x'
                rating_col = 'IMDB_Rating'

                # Display top 5 with posters in columns
                cols = st.columns(5)
                for i, (_, row) in enumerate(results.head(5).iterrows()):
                    with cols[i]:
                        if poster_col in row and pd.notna(row[poster_col]):
                            st.image(row[poster_col], use_column_width=True, caption=f"⭐ {row.get(rating_col, 'N/A')}")
                        st.markdown(f"**{row['Series_Title']}**")
                
                # Display all results in an expander for more detail
                with st.expander("See More Details in a Table"):
                    display_cols = ['Series_Title', genre_col, rating_col]
                    st.dataframe(results[display_cols].rename(columns={
                        'Series_Title': 'Title',
                        genre_col: 'Genre',
                        rating_col: 'Rating'
                    }))
            else:
                st.error("❌ No recommendations found. Please try different selections or another algorithm.")

if __name__ == "__main__":
    main()
