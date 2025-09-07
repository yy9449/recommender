import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced, load_user_ratings
from hybrid import smart_hybrid_recommendation

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
# Data Loading with Error Handling
# =========================
@st.cache_data
def load_and_prepare_data():
    """Load CSVs and prepare data for recommendation algorithms"""
    try:
        # Try different possible file paths
        movies_df = None
        imdb_df = None
        
        # Check for movies.csv
        for path in ["movies.csv", "./movies.csv", "data/movies.csv", "../movies.csv"]:
            if os.path.exists(path):
                movies_df = pd.read_csv(path)
                st.success(f"✅ Found movies.csv at: {path}")
                break
        
        # Check for imdb_top_1000.csv
        for path in ["imdb_top_1000.csv", "./imdb_top_1000.csv", "data/imdb_top_1000.csv", "../imdb_top_1000.csv"]:
            if os.path.exists(path):
                imdb_df = pd.read_csv(path)
                st.success(f"✅ Found imdb_top_1000.csv at: {path}")
                break
        
        if movies_df is None or imdb_df is None:
            return None, "CSV files not found"
        
        # Merge on Series_Title
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
        merged_df = merged_df.drop_duplicates(subset="Series_Title")
        
        # Load user ratings data
        user_ratings_df = load_user_ratings()
        if user_ratings_df is not None:
            st.info(f"📊 Dataset Info: Movies: {len(movies_df)}, IMDB: {len(imdb_df)}, Merged: {len(merged_df)}, User Ratings: {len(user_ratings_df)}")
        else:
            st.info(f"📊 Dataset Info: Movies: {len(movies_df)}, IMDB: {len(imdb_df)}, Merged: {len(merged_df)}")
        
        return merged_df, None
        
    except Exception as e:
        return None, str(e)

def load_data_with_uploader():
    """Alternative data loading with file uploader"""
    st.warning("⚠️ CSV files not found in the project directory.")
    st.info("👆 Please upload your CSV files using the file uploaders below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        movies_file = st.file_uploader("Upload movies.csv", type=['csv'], key="movies")
    
    with col2:
        imdb_file = st.file_uploader("Upload imdb_top_1000.csv", type=['csv'], key="imdb")
    
    with col3:
        ratings_file = st.file_uploader("Upload user_movie_rating.csv (Optional)", type=['csv'], key="ratings")
    
    if movies_file is not None and imdb_file is not None:
        try:
            movies_df = pd.read_csv(movies_file)
            imdb_df = pd.read_csv(imdb_file)
            
            # Handle user ratings if provided
            if ratings_file is not None:
                user_ratings_df = pd.read_csv(ratings_file)
                # Store in session state for later use
                st.session_state['user_ratings_df'] = user_ratings_df
                st.success("✅ User ratings file loaded successfully!")
            
            # Merge datasets
            merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
            merged_df = merged_df.drop_duplicates(subset="Series_Title")
            
            st.success(f"✅ Data loaded successfully! Merged dataset: {len(merged_df)} movies")
            return merged_df, None
            
        except Exception as e:
            return None, f"Error processing uploaded files: {str(e)}"
    
    return None, "Please upload both CSV files (movies.csv and imdb_top_1000.csv)"

def display_movie_posters(results_df, merged_df):
    """Display movie posters in cinema-style layout (5 columns per row)"""
    if results_df is None or results_df.empty:
        return
    
    # Get poster links and movie info
    movies_with_posters = []
    for _, row in results_df.iterrows():
        movie_title = row['Series_Title']
        full_movie_info = merged_df[merged_df['Series_Title'] == movie_title].iloc[0]
        
        poster_url = full_movie_info.get('Poster_Link', '')
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
        
        movies_with_posters.append({
            'title': movie_title,
            'poster': poster_url if pd.notna(poster_url) and poster_url.strip() else None,
            'rating': full_movie_info.get(rating_col, 'N/A'),
            'genre': full_movie_info.get(genre_col, 'N/A'),
            'year': full_movie_info.get(year_col, 'N/A')
        })
    
    # Display in rows of 5 columns
    movies_per_row = 5
    
    for i in range(0, len(movies_with_posters), movies_per_row):
        cols = st.columns(movies_per_row)
        row_movies = movies_with_posters[i:i + movies_per_row]
        
        for j, movie in enumerate(row_movies):
            with cols[j]:
                # Movie poster with consistent sizing
                if movie['poster']:
                    try:
                        st.image(
                            movie['poster'], 
                            width=200  # Fixed width for consistency
                        )
                    except:
                        # Fallback if image fails to load
                        st.container()
                        st.markdown(
                            f"""
                            <div style='
                                width: 200px; 
                                height: 300px; 
                                background-color: #f0f0f0; 
                                display: flex; 
                                align-items: center; 
                                justify-content: center;
                                border: 1px solid #ddd;
                                border-radius: 8px;
                            '>
                                <p style='text-align: center; color: #666;'>🎬<br>No Image<br>Available</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    # No poster available - show placeholder
                    st.markdown(
                        f"""
                        <div style='
                            width: 200px; 
                            height: 300px; 
                            background-color: #f0f0f0; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center;
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            margin-bottom: 10px;
                        '>
                            <p style='text-align: center; color: #666;'>🎬<br>No Image<br>Available</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Movie information below poster
                st.markdown(f"**{movie['title'][:25]}{'...' if len(movie['title']) > 25 else ''}**")
                st.markdown(f"⭐ {movie['rating']}/10")
                st.markdown(f"📅 {movie['year']}")
                
                # Genre with text wrapping
                genre_text = str(movie['genre'])[:30] + "..." if len(str(movie['genre'])) > 30 else str(movie['genre'])
                st.markdown(f"🎭 {genre_text}")
                
                # Add some spacing between movies
                st.markdown("---")

# =========================
# Main Application
# =========================
def main():
    # Load data
    merged_df, error = load_and_prepare_data()

    if merged_df is None:
        merged_df, error = load_data_with_uploader()

    # Stop execution if no data is available
    if merged_df is None:
        st.error(f"❌ Error loading data: {error}")
        st.info("🔧 **Quick Fix Instructions:**")
        st.markdown("""
        1. **Upload Files**: Use the file uploaders above
        2. **Check File Names**: Ensure files are named exactly:
           - `movies.csv` and `imdb_top_1000.csv` (required)
           - `user_movie_rating.csv` (optional, for real user data)
        3. **File Structure**: Make sure CSV files have the required columns:
           - movies.csv should have 'Series_Title' column
           - imdb_top_1000.csv should have 'Series_Title', 'Genre_y', 'IMDB_Rating' columns
           - user_movie_rating.csv should have 'User_ID', 'Movie_ID', 'Rating' columns
        """)
        st.stop()

    # Sidebar
    st.sidebar.header("🎯 Recommendation Settings")
    
    # New input method - can select both movie and genre
    st.sidebar.subheader("📝 Input Selection")
    
    # Movie selection
    st.sidebar.markdown("**🎬 Movie Selection**")
    all_movie_titles = sorted(merged_df['Series_Title'].dropna().unique().tolist())
    movie_title = st.sidebar.selectbox(
        "Select a Movie (Optional):",
        options=[""] + all_movie_titles,
        index=0,
        help="Choose a movie to get similar recommendations"
    )
    
    # Genre selection
    st.sidebar.markdown("**🎭 Genre Selection**")
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    all_genres = set()
    for genre_str in merged_df[genre_col].dropna():
        if isinstance(genre_str, str):
            all_genres.update([g.strip() for g in genre_str.split(',')])
    
    sorted_genres = sorted(all_genres)
    genre_input = st.sidebar.selectbox(
        "Select Genre (Optional):", 
        options=[""] + sorted_genres,
        help="Choose a genre to filter recommendations"
    )
    
    # Show input combination info
    if movie_title and genre_input:
        st.sidebar.success("🎯 Using both movie and genre for enhanced recommendations!")
    elif movie_title:
        st.sidebar.info("🎬 Using movie-based recommendations")
    elif genre_input:
        st.sidebar.info("🎭 Using genre-based recommendations")
    else:
        st.sidebar.warning("⚠️ Please select at least a movie or genre")
    
    # Show selected movie info if movie is selected
    if movie_title:
        movie_info = merged_df[merged_df['Series_Title'] == movie_title].iloc[0]
        
        with st.sidebar.expander("ℹ️ Selected Movie Info", expanded=True):
            rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
            year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
            
            st.write(f"**🎬 {movie_title}**")
            if genre_col in movie_info:
                st.write(f"**🎭 Genre:** {movie_info[genre_col]}")
            if rating_col in movie_info:
                st.write(f"**⭐ Rating:** {movie_info[rating_col]}/10")
            if year_col in movie_info:
                st.write(f"**📅 Year:** {movie_info[year_col]}")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "🔬 Choose Algorithm:",
        ["Hybrid", "Content-Based", "Collaborative Filtering"]
    )
    
    # Number of recommendations
    top_n = st.sidebar.slider("📊 Number of Recommendations:", 3, 15, 8)
    
    # Generate button
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        if not movie_title and not genre_input:
            st.error("❌ Please provide either a movie title or select a genre!")
            return
        
        with st.spinner("🎬 Generating recommendations using advanced algorithms..."):
            results = None
            algorithm_info = ""
            
            if algorithm == "Content-Based":
                results = content_based_filtering_enhanced(merged_df, movie_title, genre_input, top_n)
                algorithm_info = "Content-Based Filtering uses Cosine Similarity to analyze movie features like genre, director, year, and rating to find similar movies."
            
            elif algorithm == "Collaborative Filtering":
                if movie_title:
                    results = collaborative_filtering_enhanced(merged_df, movie_title, top_n)
                    algorithm_info = "Collaborative Filtering uses K-Nearest Neighbors (KNN) to analyze user behavior patterns and recommend movies liked by similar users."
                else:
                    st.warning("⚠️ Collaborative filtering requires a movie title input.")
                    return
            
            else:  # Hybrid
                results = smart_hybrid_recommendation(merged_df, movie_title, genre_input, top_n)
                if movie_title and genre_input:
                    algorithm_info = "Hybrid System combines KNN Collaborative Filtering (50%) + Cosine Similarity Content-Based on movie (25%) + Cosine Similarity Content-Based on genre (25%) for maximum accuracy."
                elif movie_title:
                    algorithm_info = "Hybrid System combines KNN Collaborative Filtering (60%) + Cosine Similarity Content-Based Filtering (40%) for enhanced accuracy."
                else:
                    algorithm_info = "Content-Based Filtering with Cosine Similarity and enhanced genre weighting for optimal genre-based recommendations."
            
            # Display results
            if results is not None and not results.empty:
                st.success(f"✅ Found {len(results)} recommendations!")
                
                # Algorithm info
                st.info(f"🔬 **{algorithm}**: {algorithm_info}")
                
                # Results display
                st.subheader("🎬 Recommended Movies")
                
                # Cinema-style poster display
                display_movie_posters(results, merged_df)
                
                # Optional: Show detailed table
                with st.expander("📊 View Detailed Information", expanded=False):
                    # Format the results for better display
                    display_results = results.copy()
                    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in results.columns else 'Rating'
                    genre_col = 'Genre_y' if 'Genre_y' in results.columns else 'Genre'
                    
                    display_results = display_results.rename(columns={
                        'Series_Title': 'Movie Title',
                        genre_col: 'Genre',
                        rating_col: 'IMDB Rating'
                    })
                    
                    # Add ranking
                    display_results.insert(0, 'Rank', range(1, len(display_results) + 1))
                    
                    st.dataframe(
                        display_results,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", width="small"),
                            "Movie Title": st.column_config.TextColumn("Movie Title", width="large"),
                            "Genre": st.column_config.TextColumn("Genre", width="medium"),
                            "IMDB Rating": st.column_config.NumberColumn("IMDB Rating", format="%.1f⭐")
                        }
                    )
                
                # Enhanced insights
                st.subheader("📈 Recommendation Insights")
                
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_rating = results[rating_col].mean()
                    st.metric("Average Rating", f"{avg_rating:.1f}⭐")
                
                with col2:
                    total_movies = len(results)
                    st.metric("Total Recommendations", total_movies)
                
                with col3:
                    # Highest rated movie
                    max_rating = results[rating_col].max()
                    st.metric("Highest Rating", f"{max_rating:.1f}⭐")
                
                with col4:
                    # Most common genre
                    genres_list = []
                    for genre_str in results[genre_col].dropna():
                        genres_list.extend([g.strip() for g in str(genre_str).split(',')])
                    
                    if genres_list:
                        most_common_genre = pd.Series(genres_list).mode().iloc[0] if len(pd.Series(genres_list).mode()) > 0 else "Various"
                        st.metric("Top Genre", most_common_genre)
                
                # Genre and rating distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    if genres_list:
                        st.subheader("🎭 Genre Distribution")
                        genre_counts = pd.Series(genres_list).value_counts().head(8)
                        st.bar_chart(genre_counts)
                
                with col2:
                    st.subheader("⭐ Rating Distribution")
                    rating_bins = pd.cut(results[rating_col], bins=5, labels=['Low', 'Below Avg', 'Average', 'Above Avg', 'High'])
                    rating_dist = rating_bins.value_counts()
                    st.bar_chart(rating_dist)
                
                # Show input combination effect if both were used
                if movie_title and genre_input:
                    st.subheader("🎯 Input Combination Analysis")
                    st.success(f"Using both '{movie_title}' and '{genre_input}' genre for enhanced precision!")
                    
                    # Show genre matching in results
                    genre_matches = 0
                    for _, row in results.iterrows():
                        if genre_input.lower() in str(row[genre_col]).lower():
                            genre_matches += 1
                    
                    match_percentage = (genre_matches / len(results)) * 100
                    st.info(f"📊 {genre_matches}/{len(results)} recommendations ({match_percentage:.1f}%) match your selected genre '{genre_input}'")
            
            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")
                
                # Provide suggestions
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
    
    # Enhanced dataset info
    with st.expander("📊 Enhanced Dataset Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Movies:** {len(merged_df)}")
            
            rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
            if rating_col in merged_df.columns:
                avg_rating = merged_df[rating_col].mean()
                max_rating = merged_df[rating_col].max()
                min_rating = merged_df[rating_col].min()
                st.write(f"**Average Rating:** {avg_rating:.1f}⭐")
                st.write(f"**Rating Range:** {min_rating:.1f} - {max_rating:.1f}")
            
            # Check for user ratings
            user_ratings_df = load_user_ratings()
            if user_ratings_df is not None:
                st.write(f"**User Ratings Available:** ✅")
                st.write(f"**Total User Ratings:** {len(user_ratings_df)}")
                st.write(f"**Unique Users:** {user_ratings_df['User_ID'].nunique()}")
            else:
                st.write(f"**User Ratings Available:** ❌ (Using synthetic data)")
        
        with col2:
            # Top genres
            all_genres = []
            for genre_str in merged_df[genre_col].dropna():
                if isinstance(genre_str, str):
                    all_genres.extend([g.strip() for g in genre_str.split(',')])
            
            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts()
                st.write("**Top Genres:**")
                st.bar_chart(genre_counts.head(10))

if __name__ == "__main__":
    main()
