import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
import os

warnings.filterwarnings('ignore')

# --- Import custom recommendation functions with error handling ---
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

# --- Backup content-based function ---
def simple_content_based(merged_df, target_movie=None, genre_filter=None, top_n=10):
    """Simplified content-based filtering using available columns as a fallback."""
    if not target_movie and not genre_filter:
        return pd.DataFrame()
    
    # Dynamically determine the correct column names after merging
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x' if 'Genre_x' in merged_df.columns else 'Genre'
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'

    # Handle genre-only filtering
    if genre_filter and not target_movie:
        if genre_col in merged_df.columns:
            filtered = merged_df[merged_df[genre_col].str.contains(genre_filter, case=False, na=False)]
            if rating_col in filtered.columns:
                filtered = filtered.sort_values(rating_col, ascending=False)
            return filtered.head(top_n)
        return pd.DataFrame()
    
    # Handle movie-based filtering
    if not target_movie or target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
    
    # Simplified genre similarity logic
    target_row = merged_df[merged_df['Series_Title'] == target_movie].iloc[0]
    
    if genre_col not in merged_df.columns:
        return pd.DataFrame()
    
    target_genres = set(str(target_row[genre_col]).split(', ')) if pd.notna(target_row[genre_col]) else set()
    
    if not target_genres:
        return pd.DataFrame()

    def get_genre_similarity(genres_str):
        movie_genres = set(str(genres_str).split(', '))
        return len(target_genres.intersection(movie_genres))

    merged_df['similarity'] = merged_df[genre_col].apply(get_genre_similarity)
    
    results = merged_df[merged_df['Series_Title'] != target_movie]
    results = results.sort_values(by=['similarity', rating_col], ascending=[False, False])
    
    # Apply genre filter if provided
    if genre_filter:
        results = results[results[genre_col].str.contains(genre_filter, case=False, na=False)]
    
    return results.head(top_n).drop(columns=['similarity'])


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
        st.error(f"❌ Error processing {file_name}: {str(e)}")
        return None

@st.cache_data
def load_and_prepare_data():
    """Load CSVs from GitHub and prepare data for recommendation algorithms - silent version"""
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    movies_url = github_base_url + "movies.csv"
    imdb_url = github_base_url + "imdb_top_1000.csv"
    user_ratings_url = github_base_url + "user_movie_rating.csv"
    
    with st.spinner("Loading datasets..."):
        movies_df = load_csv_from_github(movies_url, "movies.csv")
        imdb_df = load_csv_from_github(imdb_url, "imdb_top_1000.csv")
        user_ratings_df = load_csv_from_github(user_ratings_url, "user_movie_rating.csv")
    
    if movies_df is None or imdb_df is None:
        return None, None, "❌ Required CSV files (movies.csv, imdb_top_1000.csv) could not be loaded from GitHub"
    
    if user_ratings_df is not None:
        st.session_state['user_ratings_df'] = user_ratings_df
    
    try:
        if 'Series_Title' not in movies_df.columns or 'Series_Title' not in imdb_df.columns:
            return None, None, "❌ Missing Series_Title column in one or both datasets"
        
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
        
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner").drop_duplicates(subset="Series_Title")
        
        if 'Movie_ID' not in merged_df.columns and 'Movie_ID' in movies_df.columns:
            merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on="Series_Title", how="inner")
        
        return merged_df, user_ratings_df, None
    except Exception as e:
        return None, None, f"❌ Error merging datasets: {str(e)}"

@st.cache_data
def load_local_fallback():
    """Fallback to load local files if GitHub loading fails - silent version"""
    try:
        movies_df = pd.read_csv("movies.csv") if os.path.exists("movies.csv") else None
        imdb_df = pd.read_csv("imdb_top_1000.csv") if os.path.exists("imdb_top_1000.csv") else None
        user_ratings_df = pd.read_csv("user_movie_rating.csv") if os.path.exists("user_movie_rating.csv") else None
        
        if movies_df is None or imdb_df is None:
            return None, None, "Required CSV files not found locally either"
        
        if user_ratings_df is not None:
            st.session_state['user_ratings_df'] = user_ratings_df
        
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
        
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner").drop_duplicates(subset="Series_Title")
        
        if 'Movie_ID' not in merged_df.columns and 'Movie_ID' in movies_df.columns:
            merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on="Series_Title", how="inner")
            
        return merged_df, user_ratings_df, None
    except Exception as e:
        return None, None, str(e)

def display_movie_posters(results_df, merged_df):
    """Display movie posters in cinema-style layout (5 columns per row)"""
    if results_df is None or results_df.empty:
        return
    
    movies_with_posters = []
    for _, row in results_df.iterrows():
        movie_title = row['Series_Title']
        full_movie_info = merged_df[merged_df['Series_Title'] == movie_title].iloc[0]
        
        poster_url = full_movie_info.get('Poster_Link')
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in full_movie_info else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in full_movie_info else 'Genre_x'
        year_col = 'Released_Year' if 'Released_Year' in full_movie_info else 'Year'
        
        movies_with_posters.append({
            'title': movie_title,
            'poster': poster_url if pd.notna(poster_url) else None,
            'rating': full_movie_info.get(rating_col, 'N/A'),
            'genre': full_movie_info.get(genre_col, 'N/A'),
            'year': full_movie_info.get(year_col, 'N/A')
        })
    
    movies_per_row = 5
    for i in range(0, len(movies_with_posters), movies_per_row):
        cols = st.columns(movies_per_row)
        row_movies = movies_with_posters[i:i + movies_per_row]
        
        for j, movie in enumerate(row_movies):
            with cols[j]:
                placeholder = f"""<div style='width: 200px; height: 300px; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; border-radius: 8px;'><p style='text-align: center; color: #666;'>🎬<br>No Image</p></div>"""
                if movie['poster']:
                    st.image(movie['poster'], width=200, use_column_width='auto')
                else:
                    st.markdown(placeholder, unsafe_allow_html=True)
                
                st.markdown(f"**{movie['title'][:25]}{'...' if len(movie['title']) > 25 else ''}**")
                st.markdown(f"⭐ {movie['rating']}/10")
                st.markdown(f"📅 {movie['year']}")
                genre_text = str(movie['genre'])[:30] + "..." if len(str(movie['genre'])) > 30 else str(movie['genre'])
                st.markdown(f"🎭 {genre_text}")
                st.markdown("---")

# =========================
# Main Application
# =========================
def main():
    merged_df, user_ratings_df, error = load_and_prepare_data()
    
    if merged_df is None:
        st.warning("⚠️ GitHub loading failed, trying local files...")
        merged_df, user_ratings_df, local_error = load_local_fallback()
        
        if merged_df is None:
            st.error("❌ Could not load datasets from GitHub or local files.")
            with st.expander("🔍 Error Details"):
                st.write("**GitHub Error:**", error if error else "Unknown")
                st.write("**Local Error:**", local_error if local_error else "Unknown")
            st.stop()
    
    st.success("🎉 Ready to recommend!")
    
    with st.expander("📊 Dataset Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Movies", len(merged_df))
        if user_ratings_df is not None:
            col2.metric("User Ratings", len(user_ratings_df))
            col3.metric("Unique Users", user_ratings_df['User_ID'].nunique())

    user_ratings_available = user_ratings_df is not None

    st.sidebar.header("🎯 Recommendation Settings")
    st.sidebar.subheader("🔍 Input Selection")
    
    st.sidebar.markdown("**🎬 Movie Selection**")
    all_movie_titles = sorted(merged_df['Series_Title'].dropna().unique().tolist())
    movie_title = st.sidebar.selectbox("Select a Movie (Optional):", options=[""] + all_movie_titles)
    
    st.sidebar.markdown("**🎭 Genre Selection**")
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre_x'
    all_genres = sorted(list(set(g.strip() for genre_str in merged_df[genre_col].dropna() if isinstance(genre_str, str) for g in genre_str.split(','))))
    genre_input = st.sidebar.selectbox("Select Genre (Optional):", options=[""] + all_genres)
    
    if movie_title and genre_input: st.sidebar.success("🎯 Using both movie and genre!")
    elif movie_title: st.sidebar.info("🎬 Using movie-based recommendations")
    elif genre_input: st.sidebar.info("🎭 Using genre-based recommendations")
    else: st.sidebar.warning("⚠️ Please select a movie or genre")
    
    if movie_title:
        with st.sidebar.expander("ℹ️ Selected Movie Info", expanded=True):
            movie_info = merged_df[merged_df['Series_Title'] == movie_title].iloc[0]
            rating_col_info = 'IMDB_Rating' if 'IMDB_Rating' in movie_info else 'Rating'
            year_col_info = 'Released_Year' if 'Released_Year' in movie_info else 'Year'
            st.write(f"**🎬 {movie_title}**")
            if 'Movie_ID' in movie_info: st.write(f"**🆔 Movie ID:** {movie_info['Movie_ID']}")
            if genre_col in movie_info: st.write(f"**🎭 Genre:** {movie_info[genre_col]}")
            if rating_col_info in movie_info: st.write(f"**⭐ Rating:** {movie_info[rating_col_info]}/10")
            if year_col_info in movie_info: st.write(f"**📅 Year:** {movie_info[year_col_info]}")

    algorithm = st.sidebar.selectbox("🔬 Choose Algorithm:", ["Hybrid", "Content-Based", "Collaborative Filtering"])
    top_n = st.sidebar.slider("📊 Number of Recommendations:", 3, 15, 8)
    
    if user_ratings_available: st.sidebar.success("💾 Real user data available")
    else: st.sidebar.info("🤖 Using synthetic profiles")
    
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        if not movie_title and not genre_input:
            st.error("❌ Please provide either a movie title or select a genre!")
            return
        
        with st.spinner("🎬 Generating personalized recommendations..."):
            results = None
            try:
                if algorithm == "Content-Based":
                    if content_based_filtering_enhanced:
                        # --- CORRECTED FUNCTION CALL ---
                        results = content_based_filtering_enhanced(
                            merged_df=merged_df, 
                            target_movie=movie_title if movie_title else None,
                            genre_filter=genre_input if genre_input else None,
                            top_n=top_n
                        )
                    else:
                        results = simple_content_based(merged_df, movie_title, genre_input, top_n)
                    
                elif algorithm == "Collaborative Filtering":
                    if movie_title and user_ratings_df is not None and collaborative_filtering_enhanced:
                        results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
                    else:
                        st.warning("Collaborative filtering requires a movie title and user ratings data.")
                        return
                        
                else:  # Hybrid
                    if smart_hybrid_recommendation and user_ratings_df is not None:
                        # --- CORRECTED FUNCTION CALL ---
                        results = smart_hybrid_recommendation(
                            merged_df=merged_df,
                            user_ratings_df=user_ratings_df,
                            target_movie=movie_title if movie_title else None,
                            genre_filter=genre_input if genre_input else None,
                            top_n=top_n
                        )
                    else:
                        st.info("No user data or hybrid module available, falling back to content-based recommendations.")
                        results = simple_content_based(merged_df, movie_title, genre_input, top_n)
                        
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                return
            
            if results is not None and not results.empty:
                st.subheader("🎬 Recommended Movies")
                display_movie_posters(results, merged_df)
                
                with st.expander("📊 View Detailed Information", expanded=False):
                    display_results = results.copy()
                    rating_col_disp = 'IMDB_Rating' if 'IMDB_Rating' in results.columns else 'Rating'
                    display_results = display_results.rename(columns={'Series_Title': 'Movie Title', genre_col: 'Genre', rating_col_disp: 'IMDB Rating'})
                    display_results.insert(0, 'Rank', range(1, len(display_results) + 1))
                    st.dataframe(display_results[['Rank', 'Movie Title', 'Genre', 'IMDB Rating']], use_container_width=True, hide_index=True)
                
                st.subheader("📈 Recommendation Insights")
                col1, col2, col3, col4 = st.columns(4)
                rating_col_metric = 'IMDB_Rating' if 'IMDB_Rating' in results.columns else 'Rating'
                
                with col1: st.metric("Average Rating", f"{results[rating_col_metric].mean():.1f}⭐")
                with col2: st.metric("Total Recommendations", len(results))
                with col3: st.metric("Highest Rating", f"{results[rating_col_metric].max():.1f}⭐")
                
                genres_list = [g.strip() for genre_str in results[genre_col].dropna() for g in str(genre_str).split(',')]
                if genres_list:
                    most_common_genre = pd.Series(genres_list).mode().iloc[0]
                    with col4: st.metric("Top Genre", most_common_genre)
                
                if movie_title and genre_input:
                    st.subheader("🎯 Input Combination Analysis")
                    genre_matches = sum(1 for _, row in results.iterrows() if genre_input.lower() in str(row[genre_col]).lower())
                    match_percentage = (genre_matches / len(results)) * 100
                    st.info(f"📊 {genre_matches}/{len(results)} recommendations ({match_percentage:.1f}%) match your selected genre '{genre_input}'")
            
            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")
                st.subheader("💡 Suggestions:")
                if movie_title and not genre_input: st.write("- Try adding a genre preference or using a different algorithm.")
                elif genre_input and not movie_title: st.write("- Try selecting a specific movie you like.")
                else: st.write("- Check if the movie title is correct or try a more common genre.")

if __name__ == "__main__":
    main()
