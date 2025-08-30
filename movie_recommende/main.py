import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
from difflib import get_close_matches
import warnings
import os

# Try to import advanced search component
try:
    from streamlit_searchbox import st_searchbox
    ADVANCED_SEARCH = True
except ImportError:
    ADVANCED_SEARCH = False

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

st.title("🎬 Enhanced Movie Recommendation System")
st.markdown("---")

# =========================
# Data Loading with Error Handling
# =========================
@st.cache_data
def load_and_prepare_data():
    """Load CSVs and prepare data for recommendation algorithms"""
    try:
        # Try different possible file paths
        possible_paths = [
            "movies.csv", "imdb_top_1000.csv",
            "./movies.csv", "./imdb_top_1000.csv",
            "data/movies.csv", "data/imdb_top_1000.csv",
            "../movies.csv", "../imdb_top_1000.csv"
        ]
        
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
        
        st.info(f"📊 Dataset Info: Movies: {len(movies_df)}, IMDB: {len(imdb_df)}, Merged: {len(merged_df)}")
        
        return merged_df, None
        
    except Exception as e:
        return None, str(e)

# File uploader as backup
def load_data_with_uploader():
    """Alternative data loading with file uploader"""
    st.warning("⚠️ CSV files not found in the project directory.")
    st.info("👆 Please upload your CSV files using the file uploaders below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        movies_file = st.file_uploader("Upload movies.csv", type=['csv'], key="movies")
    
    with col2:
        imdb_file = st.file_uploader("Upload imdb_top_1000.csv", type=['csv'], key="imdb")
    
    if movies_file is not None and imdb_file is not None:
        try:
            movies_df = pd.read_csv(movies_file)
            imdb_df = pd.read_csv(imdb_file)
            
            # Merge datasets
            merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
            merged_df = merged_df.drop_duplicates(subset="Series_Title")
            
            st.success(f"✅ Data loaded successfully! Merged dataset: {len(merged_df)} movies")
            return merged_df, None
            
        except Exception as e:
            return None, f"Error processing uploaded files: {str(e)}"
    
    return None, "Please upload both CSV files"

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
    2. **Check File Names**: Ensure files are named exactly `movies.csv` and `imdb_top_1000.csv`
    3. **File Structure**: Make sure CSV files have the required columns:
       - movies.csv should have 'Series_Title' column
       - imdb_top_1000.csv should have 'Series_Title', 'Genre_y', 'IMDB_Rating' columns
    """)
    st.stop()

# =========================
# Helper Functions
# =========================
def safe_convert_to_numeric(value, default=None):
    """Safely convert a value to numeric, handling strings and NaN"""
    if pd.isna(value):
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove any non-numeric characters except decimal point
        clean_value = re.sub(r'[^\d.-]', '', str(value))
        try:
            return float(clean_value) if clean_value else default
        except (ValueError, TypeError):
            return default
    
    return default

@st.cache_data
def create_content_features():
    """Create enhanced content-based features matrix"""
    features = []
    
    for _, movie in merged_df.iterrows():
        feature_vector = []
        
        # Genre features (one-hot encoded)
        all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
                     'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 
                     'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
        
        movie_genres = []
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        if pd.notna(movie[genre_col]):
            movie_genres = [g.strip() for g in movie[genre_col].split(',')]
        
        # One-hot encode genres
        genre_features = [1 if genre in movie_genres else 0 for genre in all_genres]
        feature_vector.extend(genre_features)
        
        # Director feature (simplified)
        director_col = 'Director' if 'Director' in merged_df.columns else 'Director'
        director_hash = hash(str(movie.get(director_col, 'unknown'))) % 100
        feature_vector.append(director_hash)
        
        # Year feature (normalized)
        year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
        year = safe_convert_to_numeric(movie.get(year_col), 2000)
        if year and 1900 <= year <= 2025:
            normalized_year = (year - 1920) / (2025 - 1920)
        else:
            normalized_year = 0.5
        feature_vector.append(normalized_year)
        
        # Runtime feature (normalized)
        runtime_col = 'Runtime' if 'Runtime' in merged_df.columns else 'Runtime'
        runtime = safe_convert_to_numeric(movie.get(runtime_col), 120)
        if runtime and runtime > 0:
            normalized_runtime = min(runtime / 200.0, 1.0)
        else:
            normalized_runtime = 0.6
        feature_vector.append(normalized_runtime)
        
        # Rating feature
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        rating = safe_convert_to_numeric(movie.get(rating_col), 7.0)
        if rating and 0 <= rating <= 10:
            normalized_rating = rating / 10.0
        else:
            normalized_rating = 0.7
        feature_vector.append(normalized_rating)
        
        features.append(feature_vector)
    
    return np.array(features)

@st.cache_data
def create_user_item_matrix():
    """Create a synthetic user-item matrix based on movie characteristics"""
    np.random.seed(42)
    
    user_types = {
        'action_lover': {'Action': 5, 'Adventure': 4, 'Thriller': 4, 'Drama': 2, 'Comedy': 2, 'Romance': 1},
        'drama_fan': {'Drama': 5, 'Romance': 4, 'Biography': 4, 'Action': 2, 'Comedy': 3, 'Thriller': 2},
        'comedy_fan': {'Comedy': 5, 'Romance': 4, 'Family': 4, 'Action': 2, 'Drama': 3, 'Horror': 1},
        'thriller_fan': {'Thriller': 5, 'Mystery': 4, 'Crime': 4, 'Horror': 3, 'Action': 4, 'Comedy': 2},
        'classic_lover': {'Drama': 4, 'Romance': 4, 'Biography': 5, 'History': 5, 'War': 4, 'Comedy': 3}
    }
    
    user_movie_ratings = {}
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    for user_type, preferences in user_types.items():
        user_ratings = []
        for _, movie in merged_df.iterrows():
            rating = 0
            if pd.notna(movie[genre_col]):
                genres = [g.strip() for g in movie[genre_col].split(',')]
                genre_scores = [preferences.get(genre, 0) for genre in genres]
                if genre_scores:
                    base_rating = np.mean(genre_scores)
                    rating = max(1, min(5, base_rating + np.random.normal(0, 0.5)))
                    if np.random.random() < 0.3:
                        rating = 0
            user_ratings.append(rating)
        user_movie_ratings[user_type] = user_ratings
    
    rating_matrix = np.array(list(user_movie_ratings.values()))
    user_names = list(user_movie_ratings.keys())
    
    return rating_matrix, user_names

def search_movies(search_term):
    """Search function for movie titles"""
    if not search_term:
        return []
    
    all_titles = merged_df['Series_Title'].dropna().unique().tolist()
    
    # Find matches using fuzzy matching
    matches = find_similar_titles(search_term, all_titles, cutoff=0.3)
    
    # Also include partial matches
    partial_matches = [title for title in all_titles 
                      if search_term.lower() in title.lower()]
    
    # Combine and deduplicate
    all_matches = list(dict.fromkeys(matches + partial_matches))
    
    # Limit to top 20 results
    return all_matches[:20]
    """Enhanced fuzzy matching for movie titles"""
    input_lower = input_title.lower().strip()
    
    # Direct match
    exact_matches = [title for title in titles_list if title.lower() == input_lower]
    if exact_matches:
        return exact_matches
    
    # Partial match
    partial_matches = []
    for title in titles_list:
        title_lower = title.lower()
        if input_lower in title_lower:
            partial_matches.append((title, len(input_lower) / len(title_lower)))
        elif title_lower in input_lower:
            partial_matches.append((title, len(title_lower) / len(input_lower)))
    
    if partial_matches:
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in partial_matches[:3]]
    
    # Fuzzy match
    matches = get_close_matches(input_title, titles_list, n=5, cutoff=cutoff)
    return matches

# =========================
# Recommendation Functions
# =========================
@st.cache_data
def content_based_filtering_enhanced(target_movie=None, genre=None, top_n=5):
    """Enhanced content-based filtering"""
    if target_movie:
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if not similar_titles:
            return None
        
        target_title = similar_titles[0]
        target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
        
        content_features = create_content_features()
        target_features = content_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        similarities = cosine_similarity(target_features, content_features).flatten()
        similar_indices = np.argsort(-similarities)[1:top_n+1]
        
        result_df = merged_df.iloc[similar_indices]
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        return result_df[['Series_Title', genre_col, rating_col]]
    
    elif genre:
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        
        genre_corpus = merged_df[genre_col].fillna('').tolist()
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(genre_corpus)
        query_vector = tfidf.transform([genre])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = np.argsort(-similarities)[:top_n]
        
        result_df = merged_df.iloc[top_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
    
    return None

@st.cache_data
def collaborative_filtering_enhanced(target_movie, top_n=5):
    """Enhanced collaborative filtering"""
    if not target_movie:
        return None
    
    similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
    if not similar_titles:
        return None
    
    target_title = similar_titles[0]
    target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
    
    rating_matrix, user_names = create_user_item_matrix()
    user_similarity = cosine_similarity(rating_matrix)
    
    target_movie_idx = merged_df.index.get_loc(target_idx)
    target_ratings = rating_matrix[:, target_movie_idx]
    
    user_scores = []
    for user_idx, rating in enumerate(target_ratings):
        if rating > 3:
            avg_similarity = np.mean([user_similarity[user_idx][other_idx] 
                                    for other_idx in range(len(user_names)) 
                                    if other_idx != user_idx])
            user_scores.append((user_idx, rating * avg_similarity))
    
    if not user_scores:
        return None
    
    user_scores.sort(key=lambda x: x[1], reverse=True)
    top_users = user_scores[:3]
    
    movie_scores = {}
    for user_idx, user_weight in top_users:
        user_ratings = rating_matrix[user_idx]
        for movie_idx, rating in enumerate(user_ratings):
            if rating > 3 and movie_idx != target_movie_idx:
                movie_title = merged_df.iloc[movie_idx]['Series_Title']
                if movie_title not in movie_scores:
                    movie_scores[movie_title] = 0
                movie_scores[movie_title] += rating * user_weight
    
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not recommendations:
        return None
    
    rec_titles = [rec[0] for rec in recommendations]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

@st.cache_data
def hybrid_recommendation_enhanced(target_movie=None, genre=None, top_n=5):
    """Enhanced hybrid recommendation"""
    if not target_movie and not genre:
        return None
    
    collab_recs = collaborative_filtering_enhanced(target_movie, top_n * 2) if target_movie else None
    content_recs = content_based_filtering_enhanced(target_movie, genre, top_n * 2)
    
    if collab_recs is None and content_recs is None:
        return None
    
    movie_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    
    if collab_recs is not None:
        for idx, row in collab_recs.iterrows():
            title = row['Series_Title']
            score = row[rating_col] * 0.6
            movie_scores[title] = movie_scores.get(title, 0) + score
    
    if content_recs is not None:
        for idx, row in content_recs.iterrows():
            title = row['Series_Title']
            score = row[rating_col] * 0.4
            movie_scores[title] = movie_scores.get(title, 0) + score
    
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_movies:
        return None
    
    rec_titles = [movie[0] for movie in sorted_movies]
    result_df = merged_df[merged_df['Series_Title'].isin(rec_titles)]
    result_df = result_df.set_index('Series_Title').loc[rec_titles].reset_index()
    
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

# =========================
# Streamlit UI
# =========================
def main():
    # Sidebar
    st.sidebar.header("🎯 Recommendation Settings")
    
    # Input methods
    input_method = st.sidebar.radio("Choose Input Method:", ["Movie Title", "Genre"])
    
    if input_method == "Movie Title":
        st.sidebar.subheader("🎬 Movie Selection")
        
        # Get all movie titles for dropdown
        all_movie_titles = sorted(merged_df['Series_Title'].dropna().unique().tolist())
        
        # Choice between different input methods
        if ADVANCED_SEARCH:
            title_input_method = st.sidebar.radio(
                "Select movie by:", 
                ["🔍 Advanced Search", "📋 Dropdown List", "✍️ Type manually"],
                horizontal=True
            )
        else:
            title_input_method = st.sidebar.radio(
                "Select movie by:", 
                ["📋 Dropdown List", "✍️ Type manually"],
                horizontal=True
            )
        
        movie_title = None
        
        if ADVANCED_SEARCH and title_input_method == "🔍 Advanced Search":
            # Advanced searchable component
            movie_title = st_searchbox(
                search_movies,
                placeholder="🔍 Type to search movies...",
                label="Search Movies:",
                default="",
                clear_on_submit=False,
                key="movie_search"
            )
            
        elif title_input_method == "📋 Dropdown List":
            # Regular Streamlit selectbox with all movies
            movie_title = st.sidebar.selectbox(
                "🎬 Select Movie from List:",
                options=[""] + all_movie_titles,
                index=0,
                help="Scroll or start typing to find movies"
            )
            
            # Show random suggestions to help users discover movies
            if not movie_title:
                st.sidebar.write("🎲 **Random Suggestions:**")
                random_movies = np.random.choice(all_movie_titles, 5, replace=False)
                for i, movie in enumerate(random_movies, 1):
                    if st.sidebar.button(f"{i}. {movie}", key=f"random_{i}", use_container_width=True):
                        movie_title = movie
        
        elif title_input_method == "✍️ Type manually":
            # Manual text input with live suggestions
            movie_input = st.sidebar.text_input(
                "🎬 Enter Movie Title:", 
                placeholder="e.g., Avengers, Titanic",
                help="Type the movie title manually"
            )
            
            movie_title = movie_input
            
            # Show live suggestions as user types
            if movie_input and len(movie_input) > 2:
                suggestions = find_similar_titles(movie_input, all_movie_titles, cutoff=0.4)
                if suggestions:
                    st.sidebar.write("💡 **Suggestions:**")
                    for i, suggestion in enumerate(suggestions[:5], 1):
                        similarity_score = len(set(movie_input.lower().split()) & set(suggestion.lower().split()))
                        if st.sidebar.button(
                            f"{suggestion} {'⭐' * min(similarity_score, 3)}", 
                            key=f"suggest_{i}",
                            use_container_width=True,
                            help=f"Similarity match for '{movie_input}'"
                        ):
                            movie_title = suggestion
                            # Force rerun to update the input
                            st.session_state['movie_input'] = suggestion
        
        # Show selected movie info
        if movie_title and movie_title in all_movie_titles:
            movie_info = merged_df[merged_df['Series_Title'] == movie_title].iloc[0]
            
            with st.sidebar.expander("ℹ️ Selected Movie Info", expanded=True):
                rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
                genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
                year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
                
                st.write(f"**Title:** {movie_title}")
                if genre_col in movie_info:
                    st.write(f"**Genre:** {movie_info[genre_col]}")
                if rating_col in movie_info:
                    st.write(f"**Rating:** {movie_info[rating_col]}⭐")
                if year_col in movie_info:
                    st.write(f"**Year:** {movie_info[year_col]}")
        
        genre_input = None
        
    else:  # Genre input
        # Show available genres
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        all_genres = set()
        for genre_str in merged_df[genre_col].dropna():
            if isinstance(genre_str, str):
                all_genres.update([g.strip() for g in genre_str.split(',')])
        
        sorted_genres = sorted(all_genres)
        genre_input = st.sidebar.selectbox("🎭 Select Genre:", [""] + sorted_genres)
        movie_title = None
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "🔬 Choose Algorithm:",
        ["Hybrid (Recommended)", "Content-Based", "Collaborative Filtering"]
    )
    
    # Number of recommendations
    top_n = st.sidebar.slider("📊 Number of Recommendations:", 3, 10, 5)
    
    # Generate button
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        if not movie_title and not genre_input:
            st.error("❌ Please provide either a movie title or select a genre!")
            return
        
        with st.spinner("🎬 Generating recommendations..."):
            results = None
            
            if algorithm == "Content-Based":
                results = content_based_filtering_enhanced(movie_title, genre_input, top_n)
                algorithm_info = "Content-Based Filtering uses movie features like genre, director, year, and rating to find similar movies."
            
            elif algorithm == "Collaborative Filtering":
                if movie_title:
                    results = collaborative_filtering_enhanced(movie_title, top_n)
                    algorithm_info = "Collaborative Filtering analyzes user behavior patterns to recommend movies liked by similar users."
                else:
                    st.warning("⚠️ Collaborative filtering requires a movie title input.")
                    return
            
            else:  # Hybrid
                results = hybrid_recommendation_enhanced(movie_title, genre_input, top_n)
                algorithm_info = "Hybrid combines both Content-Based (40%) and Collaborative Filtering (60%) for optimal recommendations."
            
            # Display results
            if results is not None and not results.empty:
                st.success(f"✅ Found {len(results)} recommendations!")
                
                # Algorithm info
                st.info(f"🔬 **{algorithm}**: {algorithm_info}")
                
                # Results table
                st.subheader("🎬 Recommended Movies")
                
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
                
                # Additional insights
                if movie_title:
                    st.subheader("📈 Recommendation Insights")
                    avg_rating = display_results['IMDB Rating'].mean()
                    st.metric("Average Recommended Rating", f"{avg_rating:.1f}⭐")
                    
                    # Genre distribution
                    genres_list = []
                    for genre_str in display_results['Genre'].dropna():
                        genres_list.extend([g.strip() for g in str(genre_str).split(',')])
                    
                    if genres_list:
                        genre_counts = pd.Series(genres_list).value_counts()
                        st.bar_chart(genre_counts.head(5))
            
            else:
                st.error("❌ No recommendations found. Try a different movie title or genre.")
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        st.write(f"**Total Movies:** {len(merged_df)}")
        
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        
        if rating_col in merged_df.columns:
            avg_rating = merged_df[rating_col].mean()
            st.write(f"**Average Rating:** {avg_rating:.1f}⭐")
        
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
