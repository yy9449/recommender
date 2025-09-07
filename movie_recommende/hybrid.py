import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Import collaborative filtering components
try:
    from surprise import SVD, Reader, Dataset
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

# Global SVD model cache
svd_model = None

@st.cache_data
def load_user_ratings():
    """Load user ratings from session state or CSV"""
    try:
        if 'user_ratings_df' in st.session_state:
            df = st.session_state['user_ratings_df']
            if df is not None and not df.empty:
                return df
    except Exception:
        pass
    try:
        return pd.read_csv('user_movie_rating.csv')
    except Exception:
        return None

def train_svd_model(ratings_df):
    """Train SVD model for collaborative filtering"""
    global svd_model
    if not SURPRISE_AVAILABLE or ratings_df is None or ratings_df.empty:
        return None
    
    try:
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings_df[['User_ID', 'Movie_ID', 'Rating']], reader)
        trainset = data.build_full_trainset()
        
        svd = SVD(n_epochs=20, n_factors=50, random_state=42)
        svd.fit(trainset)
        svd_model = svd
        return svd
    except Exception:
        return None

def smart_hybrid_recommendation(user_id=1, movie_title=None, genre_input=None, df=None, 
                               ratings_df=None, top_n=10):
    """
    Simplified hybrid recommendation system.
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
    
    # 1. Content-Based Scores
    content_scores = {}
    df_copy = df.copy().fillna('')
    df_copy['soup'] = df_copy['Genre'] + ' ' + df_copy['Overview'] + ' ' + df_copy['Director'] + ' ' + df_copy['Stars']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_copy['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    if movie_title:
        indices = pd.Series(df_copy.index, index=df_copy['Series_Title']).drop_duplicates()
        if movie_title in indices:
            idx = indices[movie_title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # Genre boost if provided
            if genre_input:
                genre_col = 'Genre_y' if 'Genre_y' in df_copy.columns else 'Genre'
                for i, score in sim_scores:
                    if genre_input.lower() in str(df_copy.iloc[i][genre_col]).lower():
                        sim_scores[i] = (i, score + 0.2)
            
            for i, score in sim_scores:
                if 'Movie_ID' in df_copy.iloc[i]:
                    content_scores[df_copy.iloc[i]['Movie_ID']] = score
    
    elif genre_input:
        # Genre-based scoring when no movie selected
        genre_col = 'Genre_y' if 'Genre_y' in df_copy.columns else 'Genre'
        for idx, row in df_copy.iterrows():
            if 'Movie_ID' in row:
                score = 0.8 if genre_input.lower() in str(row[genre_col]).lower() else 0.1
                content_scores[row['Movie_ID']] = score
    
    # 2. Collaborative Filtering Scores
    collab_scores = {}
    if SURPRISE_AVAILABLE and svd_model and ratings_df is not None:
        user_rated = ratings_df[ratings_df['User_ID'] == user_id]['Movie_ID'].tolist()
        unrated_movies = [mid for mid in df['Movie_ID'].unique() if mid not in user_rated]
        
        for movie_id in unrated_movies:
            try:
                pred = svd_model.predict(user_id, movie_id)
                collab_scores[movie_id] = pred.est
            except:
                collab_scores[movie_id] = 5.0
    
    # 3. Popularity Scores (IMDB_Rating × log(votes))
    popularity_scores = {}
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'
    votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else None
    
    for _, row in df.iterrows():
        if 'Movie_ID' not in row or pd.isna(row['Movie_ID']):
            continue
        
        movie_id = row['Movie_ID']
        rating = row[rating_col] if rating_col and pd.notna(row[rating_col]) else 7.0
        votes = 1000  # Default
        
        if votes_col and pd.notna(row[votes_col]):
            votes_str = str(row[votes_col]).replace(',', '')
            votes = float(votes_str) if votes_str.isdigit() else 1000
        
        popularity_scores[movie_id] = float(rating) * np.log(votes + 1)
    
    # 4. Recency Scores (simple year-based)
    recency_scores = {}
    year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
    current_year = 2024
    
    for _, row in df.iterrows():
        if 'Movie_ID' not in row or pd.isna(row['Movie_ID']):
            continue
        
        movie_id = row['Movie_ID']
        year = current_year - 10  # Default to 10 years ago
        
        if year_col and pd.notna(row[year_col]):
            try:
                year = int(float(str(row[year_col])[:4]))
            except:
                pass
        
        years_old = max(current_year - year, 0)
        recency_scores[movie_id] = max(0.1, 1.0 - (years_old * 0.05))  # Decay 5% per year
    
    # 5. Combine all scores
    all_movie_ids = set()
    all_movie_ids.update(content_scores.keys())
    all_movie_ids.update(collab_scores.keys())
    all_movie_ids.update(popularity_scores.keys())
    all_movie_ids.update(recency_scores.keys())
    
    if not all_movie_ids:
        return pd.DataFrame()
    
    # Build scores DataFrame
    scores_data = []
    for movie_id in all_movie_ids:
        scores_data.append({
            'Movie_ID': movie_id,
            'content_score': content_scores.get(movie_id, 0.1),
            'collab_score': collab_scores.get(movie_id, 5.0),
            'popularity_score': popularity_scores.get(movie_id, 50.0),
            'recency_score': recency_scores.get(movie_id, 0.5)
        })
    
    scores_df = pd.DataFrame(scores_data)
    
    # Normalize scores to 0-1
    scaler = MinMaxScaler()
    scores_df['content_norm'] = scaler.fit_transform(scores_df[['content_score']])
    scores_df['collab_norm'] = scaler.fit_transform(scores_df[['collab_score']])
    scores_df['popularity_norm'] = scaler.fit_transform(scores_df[['popularity_score']])
    scores_df['recency_norm'] = scaler.fit_transform(scores_df[['recency_score']])
    
    # Final hybrid score: 0.3×Content + 0.5×Collaborative + 0.1×Popularity + 0.1×Recency
    scores_df['hybrid_score'] = (
        scores_df['content_norm'] * 0.3 +
        scores_df['collab_norm'] * 0.5 +
        scores_df['popularity_norm'] * 0.1 +
        scores_df['recency_norm'] * 0.1
    )
    
    # Sort and merge with movie data
    scores_df = scores_df.sort_values('hybrid_score', ascending=False)
    result_df = pd.merge(scores_df, df, on='Movie_ID', how='left')
    
    # Remove input movie if present
    if movie_title and not result_df.empty:
        result_df = result_df[result_df['Series_Title'] != movie_title]
    
    return result_df.head(top_n)

# Backward compatibility functions
def predict_hybrid_ratings(user_id, movie_id, train_df, movies_df, tfidf_matrix, 
                          cosine_sim, indices, svd_model, content_weight=0.3, collab_weight=0.5):
    """Single rating prediction for evaluation"""
    
    # Content prediction
    content_pred = 3.0
    if movie_id in indices:
        user_ratings = train_df[train_df['User_ID'] == user_id]
        if not user_ratings.empty:
            target_idx = indices[movie_id]
            rated_indices = [indices[mid] for mid in user_ratings['Movie_ID'] if mid in indices]
            if rated_indices:
                sim_scores = cosine_sim[target_idx, rated_indices]
                if len(sim_scores) > 0:
                    weighted_sim = np.dot(sim_scores, user_ratings['Rating']) / user_ratings['Rating'].sum()
                    content_pred = weighted_sim * 9 + 1
    
    # Collaborative prediction
    collab_pred = 5.0
    if svd_model:
        try:
            pred = svd_model.predict(user_id, movie_id)
            collab_pred = pred.est
        except:
            pass
    
    # Simple popularity and recency
    popularity_score = 0.5
    recency_score = 0.5
    
    # Combine: 0.3×Content + 0.5×Collaborative + 0.1×Popularity + 0.1×Recency
    final_score = (content_pred * 0.3 + collab_pred * 0.5 + 
                   popularity_score * 0.1 + recency_score * 0.1)
    
    return final_score