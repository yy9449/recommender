import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import re
from difflib import get_close_matches

# =========================
# Load CSVs
# =========================
movies_df = pd.read_csv("movies.csv")
imdb_df = pd.read_csv("imdb_top_1000.csv")

# Merge on Series_Title
merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")

# Clean duplicates
merged_df = merged_df.drop_duplicates(subset="Series_Title")

print("Movies.csv:", len(movies_df))
print("IMDB Top 1000:", len(imdb_df))
print("Merged dataset:", len(merged_df))

# =========================
# Helper Functions
# =========================
def find_similar_titles(input_title, titles_list, cutoff=0.6):
    """Find similar titles using fuzzy matching"""
    input_lower = input_title.lower()
    
    # Direct match
    if input_title in titles_list:
        return [input_title]
    
    # Partial match (for series like "avenger" -> "Avengers")
    partial_matches = []
    for title in titles_list:
        title_lower = title.lower()
        if input_lower in title_lower or title_lower in input_lower:
            partial_matches.append(title)
    
    if partial_matches:
        return partial_matches
    
    # Fuzzy match using difflib
    matches = get_close_matches(input_title, titles_list, n=5, cutoff=cutoff)
    return matches

def get_series_movies(base_title, all_titles):
    """Find movies that belong to the same series"""
    base_lower = base_title.lower()
    series_movies = []
    
    # Extract series name (e.g., "Avengers" from "Avengers: Endgame")
    series_name = base_lower
    if ':' in base_title:
        series_name = base_title.split(':')[0].strip().lower()
    elif ' ' in base_title:
        # Try to find common prefixes
        words = base_title.split()
        if len(words) > 1:
            series_name = words[0].lower()
    
    # Find all movies with similar series name
    for title in all_titles:
        title_lower = title.lower()
        # Check if the series name appears in the title
        if series_name in title_lower and title_lower != base_lower:
            series_movies.append(title)
    
    # Also check for "The" prefix variations
    if series_name.startswith('the '):
        series_name_without_the = series_name[4:]  # Remove "the "
        for title in all_titles:
            title_lower = title.lower()
            if series_name_without_the in title_lower and title_lower != base_lower:
                series_movies.append(title)
    else:
        # Check if adding "the" would match
        series_name_with_the = f"the {series_name}"
        for title in all_titles:
            title_lower = title.lower()
            if series_name_with_the in title_lower and title_lower != base_lower:
                series_movies.append(title)
    
    return list(set(series_movies))  # Remove duplicates

def get_user_input():
    """Get user input for title and genre"""
    print("\n" + "="*60)
    print("🎬 MOVIE RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Get title input
    title = input("\nEnter movie title (or press Enter to skip): ").strip()
    
    # Get genre input
    genre = input("Enter genre (or press Enter to skip): ").strip()
    
    return title, genre

def display_available_genres():
    """Display available genres from the dataset"""
    genres = set()
    for genre_str in merged_df['Genre_y'].dropna():
        if isinstance(genre_str, str):
            genres.update([g.strip() for g in genre_str.split(',')])
    
    print("\nAvailable genres:")
    for i, genre in enumerate(sorted(genres), 1):
        print(f"{i:2d}. {genre}")
    print()

def print_pretty_table(data, title, max_width=80):
    """Print a pretty formatted table"""
    if data is None or data.empty:
        return
    
    print(f"\n{title}")
    print("=" * max_width)
    
    # Format the data for display
    display_data = data.copy()
    
    # Safely truncate titles and genres
    def truncate_text(text, max_length):
        if pd.isna(text) or not isinstance(text, str):
            return str(text)[:max_length]
        return text[:max_length] + ('...' if len(text) > max_length else '')
    
    display_data['Series_Title'] = display_data['Series_Title'].apply(lambda x: truncate_text(x, 40))
    display_data['Genre_y'] = display_data['Genre_y'].apply(lambda x: truncate_text(x, 30))
    
    # Print header
    print(f"{'Title':<45} {'Genre':<35} {'Rating':<8}")
    print("-" * max_width)
    
    # Print rows
    for _, row in display_data.iterrows():
        title = row['Series_Title'][:44]
        genre = row['Genre_y'][:34]
        rating = f"{row['IMDB_Rating']:.1f}"
        print(f"{title:<45} {genre:<35} {rating:<8}")
    
    print("=" * max_width)

# =========================
# 1. Content-Based Filtering (Enhanced)
# =========================
def content_based_recommend(movie_title=None, genre=None, top_n=5):
    """Enhanced content-based filtering with genre fallback and series prioritization"""
    
    if movie_title:
        # Find similar titles
        similar_titles = find_similar_titles(movie_title, merged_df['Series_Title'].tolist())
        
        if not similar_titles:
            print(f"❌ No movies found matching '{movie_title}'")
            if genre:
                print("🔄 Falling back to genre-based recommendations...")
                return content_based_recommend_by_genre(genre, top_n)
            return None
        
        # Use the first match
        target_title = similar_titles[0]
        if len(similar_titles) > 1:
            print(f"🔍 Found multiple matches: {', '.join(similar_titles[:3])}")
            print(f"📽️ Using: {target_title}")
        
        # Get series movies first (like Avengers series)
        series_movies = get_series_movies(target_title, merged_df['Series_Title'].tolist())
        print(f"🎬 Found series movies: {series_movies}")
        
        # Create feature matrix from genres
        cv = CountVectorizer(stop_words="english")
        count_matrix = cv.fit_transform(merged_df['Genre_y'].fillna(''))
        
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        
        indices = pd.Series(merged_df.index, index=merged_df['Series_Title'])
        
        if target_title not in indices:
            print(f"❌ '{target_title}' not found in dataset.")
            return None
        
        idx = indices[target_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Prioritize series movies
        recommendations = []
        
        # Add series movies first (if any) - give them highest priority
        for series_movie in series_movies:
            if series_movie in indices:
                series_idx = indices[series_movie]
                recommendations.append((series_idx, 1.0))  # High similarity for series
                print(f"🎯 Added series movie: {series_movie}")
        
        # Add other similar movies (excluding the target movie itself)
        for movie_idx, sim_score in sim_scores[1:]:  # Skip the movie itself
            movie_title_at_idx = merged_df.iloc[movie_idx]['Series_Title']
            if movie_title_at_idx not in series_movies and movie_title_at_idx != target_title:
                recommendations.append((movie_idx, sim_score))
        
        # Take top N recommendations
        recommendations = recommendations[:top_n]
        movie_indices = [rec[0] for rec in recommendations]
        
        result = merged_df[['Series_Title', 'Genre_y', 'IMDB_Rating']].iloc[movie_indices]
        
        return result
    
    elif genre:
        return content_based_recommend_by_genre(genre, top_n)
    
    return None

def content_based_recommend_by_genre(genre, top_n=5):
    """Recommend movies by genre"""
    genre_lower = genre.lower()
    
    # Find movies with matching genre
    matching_movies = []
    for idx, row in merged_df.iterrows():
        if pd.notna(row['Genre_y']):
            movie_genres = [g.strip().lower() for g in row['Genre_y'].split(',')]
            if genre_lower in movie_genres:
                matching_movies.append((row['Series_Title'], row['Genre_y'], row['IMDB_Rating']))
    
    if not matching_movies:
        print(f"❌ No movies found with genre '{genre}'")
        return None
    
    # Sort by IMDB rating and return top N
    matching_movies.sort(key=lambda x: x[2], reverse=True)
    recommendations = pd.DataFrame(matching_movies[:top_n], columns=['Series_Title', 'Genre_y', 'IMDB_Rating'])
    
    return recommendations

# =========================
# 2. Collaborative Filtering (Enhanced)
# =========================
def collaborative_recommend(movie_title, top_n=5):
    """Enhanced collaborative filtering based on user behavior patterns"""
    
    if not movie_title:
        return None
    
    # Find similar titles
    similar_titles = find_similar_titles(movie_title, merged_df['Series_Title'].tolist())
    
    if not similar_titles:
        print(f"❌ No movies found matching '{movie_title}' for collaborative filtering")
        return None
    
    target_title = similar_titles[0]
    
    # Create user-movie matrix based on certificates (as proxy for user preferences)
    pivot_table = merged_df.pivot_table(index='Series_Title', 
                                        columns='Certificate', 
                                        values='IMDB_Rating').fillna(0)
    
    if target_title not in pivot_table.index:
        print(f"❌ '{target_title}' not found in collaborative filtering dataset.")
        return None
    
    # Use KNN to find similar movies
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(pivot_table.values)
    
    idx = pivot_table.index.get_loc(target_title)
    distances, indices = knn.kneighbors([pivot_table.iloc[idx]], n_neighbors=top_n+1)
    
    recs = [pivot_table.index[i] for i in indices.flatten()][1:]  # Exclude the movie itself
    
    recommendations = merged_df[merged_df['Series_Title'].isin(recs)][['Series_Title', 'Genre_y', 'IMDB_Rating']]
    
    return recommendations

# =========================
# 3. Hybrid (Enhanced)
# =========================
def hybrid_recommend(movie_title=None, genre=None, top_n=5):
    """Enhanced hybrid recommendation combining content and collaborative"""
    
    content_recs = content_based_recommend(movie_title, genre, top_n*2)
    collab_recs = collaborative_recommend(movie_title, top_n*2) if movie_title else None
    
    if content_recs is None and collab_recs is None:
        return None
    
    # Combine recommendations
    all_recs = []
    
    if content_recs is not None:
        all_recs.extend(content_recs.to_dict('records'))
    
    if collab_recs is not None:
        all_recs.extend(collab_recs.to_dict('records'))
    
    # Remove duplicates and sort by rating
    seen_titles = set()
    unique_recs = []
    for rec in all_recs:
        if rec['Series_Title'] not in seen_titles:
            unique_recs.append(rec)
            seen_titles.add(rec['Series_Title'])
    
    # Sort by IMDB rating and take top N
    unique_recs.sort(key=lambda x: x['IMDB_Rating'], reverse=True)
    final_recs = unique_recs[:top_n]
    
    recommendations = pd.DataFrame(final_recs)
    
    return recommendations

# =========================
# Main Application
# =========================
def main():
    while True:
        # Get user input
        title, genre = get_user_input()
        
        if not title and not genre:
            print("❌ Please provide either a movie title or genre!")
            continue
        
        # Show available genres if user wants to see them
        if not genre:
            show_genres = input("Would you like to see available genres? (y/n): ").strip().lower()
            if show_genres == 'y':
                display_available_genres()
                genre = input("Enter genre (or press Enter to skip): ").strip()
        
        print("\n" + "="*60)
        print("🎬 GENERATING RECOMMENDATIONS...")
        print("="*60)
        
        # Generate recommendations
        if title:
            print(f"\n📽️ Input: {title}")
            
            # Content-based
            content_results = content_based_recommend(title, genre, 5)
            if content_results is not None:
                print_pretty_table(content_results, "🎬 Top 5 Content-Based Recommendations")
            
            # Collaborative
            collab_results = collaborative_recommend(title, 5)
            if collab_results is not None:
                print_pretty_table(collab_results, "👥 Top 5 Collaborative Recommendations")
            
            # Hybrid
            hybrid_results = hybrid_recommend(title, genre, 5)
            if hybrid_results is not None:
                print_pretty_table(hybrid_results, "🔀 Top 5 Hybrid Recommendations")
        
        elif genre:
            print(f"\n🎭 Input: {genre}")
            
            # Content-based by genre
            content_results = content_based_recommend_by_genre(genre, 10)
            if content_results is not None:
                print_pretty_table(content_results, f"🎬 Top 10 {genre.title()} Movies")
        
        # Ask if user wants to continue
        print("\n" + "="*60)
        continue_choice = input("Would you like to get more recommendations? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("👋 Thanks for using the Movie Recommendation System!")
            break

if __name__ == "__main__":
    main()