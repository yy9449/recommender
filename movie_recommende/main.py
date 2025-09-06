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

# Backup content-based function
def simple_content_based(merged_df, target_movie, genre_filter=None, top_n=10):
    """Simplified content-based filtering using available columns"""
    if not target_movie and not genre_filter:
        return pd.DataFrame()
    
    # Handle genre-only filtering
    if genre_filter and not target_movie:
        # CORRECTED: Assumes 'Genre' column exists after coalesce
        if 'Genre' in merged_df.columns:
            filtered = merged_df[merged_df['Genre'].str.contains(genre_filter, case=False, na=False)]
            rating_col = 'IMDB_Rating' if 'IMDB_Rating' in filtered.columns else 'Rating'
            if rating_col in filtered.columns:
                filtered = filtered.sort_values(rating_col, ascending=False)
            return filtered.head(top_n)
        return pd.DataFrame()
    
    # Movie-based filtering
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()

    # Create a simple "soup" from available text columns
    # CORRECTED: Simplified list since columns are coalesced in load_data
    text_features = []
    for col in ['Genre', 'Overview', 'Director', 'Stars']:
        if col in merged_df.columns:
            text_features.append(merged_df[col].fillna(''))
    
    if not text_features:
        return pd.DataFrame()

    soup = pd.concat(text_features, axis=1).apply(lambda x: ' '.join(x), axis=1)

    # Basic TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(soup)
    
    idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return merged_df.iloc[movie_indices]

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads and merges datasets, handling potential merge issues."""
    try:
        # Main dataset with detailed info
        imdb_df = pd.read_csv("imdb_top_1000.csv")
        # Simplified dataset for content-based 'soup'
        movies_df = pd.read_csv("movies.csv")
        # User ratings for collaborative filtering
        user_ratings_df = pd.read_csv("user_movie_rating.csv")

        # Merge imdb and movies, handling potential column conflicts
        merged_df = pd.merge(
            imdb_df,
            movies_df,
            on="Series_Title",
            how="left",
            suffixes=('_x', '_y')
        )
        
        # Coalesce conflicting columns - prioritize the more detailed one (e.g., from imdb_df)
        # This is the key part that solves the problem.
        for col in ['Genre', 'Overview', 'Director']:
            col_x, col_y = f'{col}_x', f'{col}_y'
            if col_x in merged_df.columns and col_y in merged_df.columns:
                # Fill missing values in the primary column (_x) with values from the secondary (_y)
                merged_df[col] = merged_df[col_x].fillna(merged_df[col_y])
                # Drop the now redundant _x and _y columns
                merged_df.drop(columns=[col_x, col_y], inplace=True)
            # Handle cases where only one of the conflicting columns might exist after a merge
            elif col_x in merged_df.columns:
                 merged_df.rename(columns={col_x: col}, inplace=True)
            elif col_y in merged_df.columns:
                 merged_df.rename(columns={col_y: col}, inplace=True)


        return merged_df, user_ratings_df
    except FileNotFoundError:
        st.error("One or more data files are missing. Please ensure they are in the correct directory.")
        return None, None

def main():
    st.set_page_config(layout="wide", page_title="Movie Recommender", page_icon="🎬")
    
    # --- Custom CSS for Styling ---
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 0;
    }
    h1, h2, h3 {
        color: #333;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #ff6a6a;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Advanced Movie Recommender")

    # Load data
    merged_df, user_ratings_df = load_data()
    if merged_df is None:
        return

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("🔍 Filter & Select")
        
        # Algorithm selection
        algo_options = []
        if smart_hybrid_recommendation: algo_options.append("Smart Hybrid")
        if content_based_filtering_enhanced: algo_options.append("Content-Based")
        if collaborative_filtering_enhanced: algo_options.append("Collaborative")
        if not algo_options:
            algo_options.append("Simple Content-Based (Backup)")
            
        algorithm = st.selectbox("Choose a recommendation algorithm:", algo_options)
        
        # Movie selection dropdown
        movie_list = [""] + sorted(merged_df['Series_Title'].unique().tolist())
        movie_title = st.selectbox("Select a movie you like:", movie_list)

        # Genre selection dropdown
        # CORRECTED: Simplified to use the single 'Genre' column
        genre_col = 'Genre' 
        all_genres = set()
        if genre_col in merged_df.columns:
            # The original code had a slight error here, sum() is not for lists. Corrected logic:
            genres_list = [genre.strip() for sublist in merged_df[genre_col].dropna().str.split(', ').tolist() for genre in sublist]
            all_genres = sorted(list(set(genres_list)))
        
        genre_input = st.selectbox("Filter by genre (optional):", [""] + all_genres)
        
        # Number of recommendations
        top_n = st.slider("Number of recommendations:", 5, 20, 10)
        
        # Get recommendations button
        recommend_button = st.button("🚀 Get Recommendations")

    # --- Main Panel for Displaying Results ---
    if recommend_button:
        if not movie_title and not genre_input:
            st.warning("⚠️ Please select a movie or a genre to get recommendations.")
        else:
            results = pd.DataFrame()
            with st.spinner("🧠 Analyzing your preferences..."):
                if algorithm == "Smart Hybrid":
                    results = smart_hybrid_recommendation(
                        merged_df, user_ratings_df,
                        target_movie=movie_title,
                        genre_filter=genre_input,
                        top_n=top_n
                    )
                elif algorithm == "Content-Based":
                    results = content_based_filtering_enhanced(
                        merged_df,
                        target_movie=movie_title,
                        genre_filter=genre_input,
                        top_n=top_n
                    )
                elif algorithm == "Collaborative":
                    if movie_title:
                        results = collaborative_filtering_enhanced(
                            merged_df, user_ratings_df, movie_title, top_n
                        )
                    else:
                        st.error("Collaborative filtering requires a movie selection.")
                else: # Backup
                     results = simple_content_based(
                        merged_df, movie_title, genre_input, top_n
                     )

            if not results.empty:
                st.header(f"🌟 Top {len(results)} Recommendations")
                
                # --- Display in a grid-like format ---
                cols = st.columns(5)
                for i, row in enumerate(results.iterrows()):
                    idx, data = row
                    with cols[i % 5]:
                        st.image(data['Poster_Link'], use_column_width=True)
                        st.markdown(f"**{data['Series_Title']}** ({data.get('Released_Year', 'N/A')})")
                        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in data else 'Rating'
                        st.markdown(f"⭐ {data.get(rating_col, 'N/A')}")
                        # CORRECTED: Now safely gets the clean 'Genre' and 'Overview' columns
                        st.expander("Details").write(f"""
                        - **Genre:** {data.get('Genre', 'N/A')}
                        - **Director:** {data.get('Director', 'N/A')}
                        - **Overview:** {data.get('Overview', 'N/A')}
                        """)
                
                # Add some analytics on which inputs were used
                if movie_title and genre_input:
                    st.subheader("🎯 Input Combination Analysis")
                    
                    # Show genre matching in results
                    genre_matches = 0
                    for _, row in results.iterrows():
                        # CORRECTED: Uses the clean 'Genre' column directly
                        if genre_input.lower() in str(row.get('Genre', '')).lower():
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

if __name__ == "__main__":
    main()
