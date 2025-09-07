import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from content_based import content_based_filtering_enhanced, predict_content_ratings
from collaborative import collaborative_filtering_enhanced, train_svd_model, svd_model, predict_collaborative_ratings
import warnings

warnings.filterwarnings('ignore')

# =====================================================================================
# == Functions for Streamlit App (main.py)
# =====================================================================================

def smart_hybrid_recommendation(user_id=1, movie_title=None, genre_input=None, df=None, 
                               ratings_df=None, top_n=10):
    """
    TRUE hybrid recommendation system that actually combines collaborative and content-based filtering.
    FinalScore = 0.3×Content + 0.5×Collaborative + 0.1×Popularity + 0.1×Recency
    """
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Load ratings if not provided
    if ratings_df is None:
        ratings_df = load_user_ratings()
    
    # Ensure SVD model is trained
    global svd_model
    if svd_model is None and ratings_df is not None:
        svd_model = train_svd_model(ratings_df)
    
    # ========================================
    # CRITICAL FIX: Start with ALL movies
    # ========================================
    all_movie_ids = df['Movie_ID'].dropna().unique().tolist()
    print(f"Total movies in dataset: {len(all_movie_ids)}")  # Debug info
    
    # Remove movies already rated by user (optional)
    user_rated = []
    if ratings_df is not None and not ratings_df.empty:
        user_rated = ratings_df[ratings_df['User_ID'] == user_id]['Movie_ID'].tolist()
        print(f"User has rated {len(user_rated)} movies")  # Debug info
    
    # ========================================
    # 1. CONTENT-BASED SCORES FOR ALL MOVIES
    # ========================================
    content_scores = {}
    
    # Initialize ALL movies with base content score
    for movie_id in all_movie_ids:
        content_scores[movie_id] = 0.2  # Base score
    
    # Calculate content similarity if movie is provided
    if movie_title:
        df_copy = df.copy().fillna('')
        df_copy['soup'] = (df_copy['Genre'] + ' ' + 
                          df_copy.get('Overview', '') + ' ' + 
                          df_copy.get('Director', '') + ' ' + 
                          df_copy.get('Stars', ''))
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df_copy['soup'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        indices = pd.Series(df_copy.index, index=df_copy['Series_Title']).drop_duplicates()
        
        if movie_title in indices.index:
            idx = indices[movie_title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # Update content scores based on similarity
            for i, score in sim_scores:
                if i < len(df_copy) and 'Movie_ID' in df_copy.iloc[i]:
                    movie_id = df_copy.iloc[i]['Movie_ID']
                    if pd.notna(movie_id):
                        # Boost similar movies
                        content_scores[movie_id] = max(content_scores.get(movie_id, 0.2), score)
            
            print(f"Content-based calculated similarities for movie: {movie_title}")
    
    # Apply genre boost if provided
    if genre_input:
        genre_col = 'Genre_y' if 'Genre_y' in df.columns else 'Genre'
        genre_boost_count = 0
        
        for _, row in df.iterrows():
            if 'Movie_ID' in row and pd.notna(row['Movie_ID']):
                movie_id = row['Movie_ID']
                if genre_input.lower() in str(row.get(genre_col, '')).lower():
                    content_scores[movie_id] = min(content_scores.get(movie_id, 0.2) + 0.3, 1.0)
                    genre_boost_count += 1
        
        print(f"Genre boost applied to {genre_boost_count} movies for genre: {genre_input}")
    
    # ========================================
    # 2. COLLABORATIVE SCORES FOR ALL MOVIES
    # ========================================
    collab_scores = {}
    
    # Initialize ALL movies with base collaborative score
    for movie_id in all_movie_ids:
        collab_scores[movie_id] = 5.0  # Neutral rating
    
    if SURPRISE_AVAILABLE and svd_model and ratings_df is not None:
        collab_predictions = 0
        high_predictions = 0
        
        # Get collaborative predictions for ALL movies
        for movie_id in all_movie_ids:
            try:
                pred = svd_model.predict(user_id, movie_id)
                collab_scores[movie_id] = pred.est
                collab_predictions += 1
                if pred.est >= 7.0:
                    high_predictions += 1
            except:
                collab_scores[movie_id] = 5.0
        
        print(f"Collaborative filtering: {collab_predictions} predictions made, {high_predictions} high ratings (≥7.0)")
        
        # DIVERSITY MECHANISM: Boost different-genre movies with high collaborative scores
        if movie_title:
            input_movie_info = df[df['Series_Title'] == movie_title]
            diversity_boost_count = 0
            
            if not input_movie_info.empty:
                input_genre = str(input_movie_info.iloc[0].get('Genre_y' if 'Genre_y' in df.columns else 'Genre', ''))
                input_genres = set(g.strip().lower() for g in input_genre.split(','))
                
                for movie_id in all_movie_ids:
                    if collab_scores[movie_id] >= 6.5:  # Good collaborative prediction
                        movie_info = df[df['Movie_ID'] == movie_id]
                        if not movie_info.empty:
                            movie_genre = str(movie_info.iloc[0].get('Genre_y' if 'Genre_y' in df.columns else 'Genre', ''))
                            movie_genres = set(g.strip().lower() for g in movie_genre.split(','))
                            
                            # Boost movies with minimal genre overlap
                            overlap = len(input_genres.intersection(movie_genres))
                            if overlap == 0 and collab_scores[movie_id] >= 7.0:  # Different genre + high score
                                collab_scores[movie_id] = min(collab_scores[movie_id] + 2.0, 10.0)
                                diversity_boost_count += 1
                            elif overlap <= 1 and collab_scores[movie_id] >= 7.5:  # Slight overlap + very high score
                                collab_scores[movie_id] = min(collab_scores[movie_id] + 1.0, 10.0)
                                diversity_boost_count += 1
                
                print(f"Diversity boost applied to {diversity_boost_count} different-genre movies")
    else:
        print("Collaborative filtering not available (SVD model or ratings data missing)")
    
    # ========================================
    # 3. POPULARITY SCORES FOR ALL MOVIES
    # ========================================
    popularity_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'
    votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else None
    
    for _, row in df.iterrows():
        if 'Movie_ID' in row and pd.notna(row['Movie_ID']):
            movie_id = row['Movie_ID']
            rating = float(row.get(rating_col, 7.0)) if pd.notna(row.get(rating_col)) else 7.0
            
            votes = 1000  # Default
            if votes_col and pd.notna(row.get(votes_col)):
                votes_str = str(row[votes_col]).replace(',', '')
                try:
                    votes = float(votes_str)
                except:
                    votes = 1000
            
            popularity_scores[movie_id] = rating * np.log(votes + 1)
    
    # ========================================
    # 4. RECENCY SCORES FOR ALL MOVIES
    # ========================================
    recency_scores = {}
    year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
    current_year = 2024
    
    for _, row in df.iterrows():
        if 'Movie_ID' in row and pd.notna(row['Movie_ID']):
            movie_id = row['Movie_ID']
            year = current_year - 10  # Default
            
            if year_col in row and pd.notna(row[year_col]):
                try:
                    year = int(float(str(row[year_col])[:4]))
                except:
                    year = current_year - 10
            
            years_old = max(current_year - year, 0)
            recency_scores[movie_id] = max(0.1, 1.0 - (years_old * 0.05))
    
    # ========================================
    # 5. COMBINE ALL SCORES
    # ========================================
    print(f"Scores calculated for {len(all_movie_ids)} movies:")
    print(f"  Content scores: {len([s for s in content_scores.values() if s > 0.2])} above baseline")
    print(f"  Collaborative scores: {len([s for s in collab_scores.values() if s > 6.0])} above average")
    
    # Build comprehensive scores DataFrame
    scores_data = []
    for movie_id in all_movie_ids:
        scores_data.append({
            'Movie_ID': movie_id,
            'content_score': content_scores.get(movie_id, 0.2),
            'collab_score': collab_scores.get(movie_id, 5.0),
            'popularity_score': popularity_scores.get(movie_id, 50.0),
            'recency_score': recency_scores.get(movie_id, 0.5)
        })
    
    scores_df = pd.DataFrame(scores_data)
    
    # Normalize all scores to 0-1 range
    scaler = MinMaxScaler()
    scores_df['content_norm'] = scaler.fit_transform(scores_df[['content_score']]).flatten()
    scores_df['collab_norm'] = scaler.fit_transform(scores_df[['collab_score']]).flatten()
    scores_df['popularity_norm'] = scaler.fit_transform(scores_df[['popularity_score']]).flatten()
    scores_df['recency_norm'] = scaler.fit_transform(scores_df[['recency_score']]).flatten()
    
    # Calculate final hybrid score with proper weighting
    scores_df['hybrid_score'] = (
        scores_df['content_norm'] * 0.3 +      # 30% content-based
        scores_df['collab_norm'] * 0.5 +       # 50% collaborative
        scores_df['popularity_norm'] * 0.1 +   # 10% popularity
        scores_df['recency_norm'] * 0.1        # 10% recency
    )
    
    # Sort by hybrid score and merge with movie data
    scores_df = scores_df.sort_values('hybrid_score', ascending=False)
    result_df = pd.merge(scores_df, df, on='Movie_ID', how='left')
    
    # Remove input movie if present
    if movie_title and not result_df.empty:
        result_df = result_df[result_df['Series_Title'] != movie_title]
    
    # Remove already rated movies
    if user_rated:
        result_df = result_df[~result_df['Movie_ID'].isin(user_rated)]
    
    # Add debug info to see the component scores
    final_results = result_df.head(top_n).copy()
    print(f"\nTop {len(final_results)} hybrid recommendations:")
    for i, (_, row) in enumerate(final_results.head(3).iterrows()):
        print(f"  {i+1}. {row['Series_Title']} - Hybrid: {row['hybrid_score']:.3f} "
              f"(Content: {row['content_norm']:.2f}, Collab: {row['collab_norm']:.2f})")
    
    return final_results
