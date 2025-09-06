# main.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io

warnings.filterwarnings('ignore')

# Import functions with error handling
try:
    from content_based import content_based_filtering_enhanced
except ImportError:
    content_based_filtering_enhanced = None

try:
    from collaborative import collaborative_filtering_enhanced
except ImportError:
    collaborative_filtering_enhanced = None
    
try:
    from hybrid import smart_hybrid_recommendation
except ImportError:
    smart_hybrid_recommendation = None

# Backup content-based function in case main ones fail
def simple_content_based(merged_df, target_movie, genre_filter=None, top_n=10):
    """Simplified content-based filtering for fallback"""
    if not target_movie and not genre_filter:
        return pd.DataFrame()
    
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
    
    if genre_filter and not target_movie:
        if genre_col in merged_df.columns:
            filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
            rating_col = 'IMDB_Rating' if 'IMDB_Rating' in filtered.columns else 'Rating'
            if rating_col in filtered.columns:
                filtered = filtered.sort_values(rating_col, ascending=False)
            return filtered.head(top_n)
        return pd.DataFrame()
    
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    overview_col = 'Overview_y' if 'Overview_y' in merged_df.columns else 'Overview_x' if 'Overview_x' in merged_df.columns else 'Overview'
    
    text_content = merged_df[overview_col].fillna('') + ' ' + merged_df[genre_col].fillna('')
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(text_content)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return merged_df.iloc[movie_indices]

# --- Data Loading ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data():
    """Loads, merges, and preprocesses the movie datasets."""
    try:
        movies_df = pd.read_csv("movies.csv")
        imdb_df = pd.read_csv("imdb_top_1000.csv")
        user_ratings_df = pd.read_csv("user_movie_rating.csv")
        
        # Merge dataframes
        merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='outer')
        
        # Data Cleaning
        merged_df['Runtime'] = merged_df['Runtime'].str.replace(' min', '').astype(float)
        
        # Create a unified genre list for the dropdown
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x'
        all_genres = merged_df[genre_col].dropna().str.split(', ').explode()
        unique_genres = sorted(all_genres.unique())
        
        return merged_df, user_ratings_df, unique_genres
    except FileNotFoundError as e:
        st.error(f"❌ Error: A required data file was not found. Please ensure all CSV files are present. Details: {e}")
        return None, None, None

# --- Main App ---
def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("🎬 Movie Recommendation System")

    merged_df, user_ratings_df, unique_genres = load_data()

    if merged_df is None:
        st.stop()

    # --- Sidebar for User Input ---
    st.sidebar.header("🔍 Configure Your Recommendations")
    
    algorithm = st.sidebar.selectbox(
        "Choose a Recommendation Algorithm",
        ("Hybrid", "Content-Based", "Collaborative"),
        help="**Hybrid**: A smart blend of all methods. **Content-Based**: Recommends movies similar in plot and genre. **Collaborative**: Recommends movies liked by similar users."
    )

    movie_title = st.sidebar.selectbox(
        "Select a Movie You Like",
        options=[''] + sorted(merged_df['Series_Title'].dropna().unique()),
        help="Start by picking a movie to base the recommendations on."
    )

    genre_input = st.sidebar.selectbox(
        "Filter by Genre (Optional)",
        options=[''] + unique_genres,
        help="Narrow down the recommendations to a specific genre."
    )
    
    top_n = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5, max_value=20, value=10
    )

    # --- Recommendation Generation ---
    if st.sidebar.button("✨ Generate Recommendations"):
        if not movie_title and not genre_input:
            st.warning("⚠️ Please select a movie or a genre to get recommendations.")
        else:
            results = pd.DataFrame()
            try:
                if algorithm == "Content-Based":
                    if content_based_filtering_enhanced:
                        # FIX: Pass all required arguments, including genre_filter
                        results = content_based_filtering_enhanced(
                            merged_df=merged_df,
                            target_movie=movie_title if movie_title else None,
                            genre_filter=genre_input if genre_input else None,
                            top_n=top_n
                        )
                    else:
                        st.error("Content-based filtering module is unavailable.")
                
                elif algorithm == "Collaborative":
                    if not movie_title:
                        st.warning("⚠️ Collaborative filtering requires you to select a movie.")
                    elif collaborative_filtering_enhanced:
                        # FIX: Use keyword arguments for clarity
                        results = collaborative_filtering_enhanced(
                            merged_df=merged_df,
                            user_ratings_df=user_ratings_df,
                            target_movie=movie_title,
                            top_n=top_n
                        )
                    else:
                        st.error("Collaborative filtering module is unavailable.")
                
                else:  # Hybrid
                    if smart_hybrid_recommendation:
                        # FIX: Pass all arguments correctly using keywords
                        results = smart_hybrid_recommendation(
                            merged_df=merged_df,
                            user_ratings_df=user_ratings_df,
                            target_movie=movie_title if movie_title else None,
                            genre_filter=genre_input if genre_input else None,
                            top_n=top_n
                        )
                    else:
                        st.error("Hybrid filtering module is unavailable.")

            except Exception as e:
                st.error(f"An error occurred with {algorithm} filtering: {str(e)}")
                st.info("Falling back to a simpler recommendation method.")
                results = simple_content_based(
                    merged_df,
                    target_movie=movie_title,
                    genre_filter=genre_input,
                    top_n=top_n
                )

            # --- Display Results ---
            st.markdown("---")
            if not results.empty:
                st.subheader(f"🏆 Top {len(results)} Recommendations")

                # Dynamically find column names for display
                poster_col = 'Poster_Link'
                genre_col = 'Genre_y' if 'Genre_y' in results.columns else 'Genre_x'
                overview_col = 'Overview_y' if 'Overview_y' in results.columns else 'Overview_x'
                rating_col = 'IMDB_Rating'
                runtime_col = 'Runtime'
                director_col = 'Director'

                # Display results in columns
                cols = st.columns(5)
                for i, (_, row) in enumerate(results.head(5).iterrows()):
                    with cols[i]:
                        if poster_col in row and pd.notna(row[poster_col]):
                            st.image(row[poster_col], use_column_width=True)
                        st.markdown(f"**{row['Series_Title']}**")
                        if rating_col in row:
                            st.write(f"⭐ {row[rating_col]}/10")

                # Display detailed list
                with st.expander("See More Details"):
                    for _, row in results.iterrows():
                        st.markdown(f"#### {row['Series_Title']}")
                        c1, c2 = st.columns([1, 4])
                        if poster_col in row and pd.notna(row[poster_col]):
                            c1.image(row[poster_col], width=150)
                        
                        details = ""
                        if rating_col in row: details += f"- **Rating**: {row[rating_col]:.1f}/10\n"
                        if runtime_col in row: details += f"- **Runtime**: {int(row[runtime_col])} min\n"
                        if director_col in row: details += f"- **Director**: {row[director_col]}\n"
                        if genre_col in row: details += f"- **Genre**: {row[genre_col]}\n"
                        c2.markdown(details)
                        
                        if overview_col in row:
                            st.markdown(f"> {row[overview_col]}")
                        st.markdown("---")
            
            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")

if __name__ == "__main__":
    main()
