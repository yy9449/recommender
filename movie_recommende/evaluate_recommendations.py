# evaluate_recommendations.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    mean_squared_error, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from surprise import SVD, Reader, Dataset
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
RATING_THRESHOLD = 5.0
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Data Loading ---
def load_and_prepare_data():
    movies = pd.read_csv('movies.csv')
    imdb = pd.read_csv('imdb_top_1000.csv')
    user_ratings = pd.read_csv('user_movie_rating.csv')

    if 'Movie_ID' not in movies.columns:
        movies['Movie_ID'] = range(len(movies))

    merged_df = pd.merge(movies, imdb, on='Series_Title', how='left')

    for col in ['Genre', 'Overview', 'Director']:
        col_x, col_y = f'{col}_x', f'{col}_y'
        if col_y in merged_df.columns:
            merged_df[col] = merged_df[col_y].fillna(merged_df.get(col_x, ''))
        elif col_x in merged_df.columns:
             merged_df[col] = merged_df[col_x]
        else:
             merged_df[col] = ''

    for col in ['IMDB_Rating', 'No_of_Votes', 'Released_Year']:
        if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    merged_df = merged_df.dropna(subset=['Movie_ID']).drop_duplicates(subset=['Movie_ID'])
    merged_df['Movie_ID'] = merged_df['Movie_ID'].astype(int)
    
    return merged_df, user_ratings

# --- Prediction Model ---
def predict_content_based(user_id, movie_id, train_df, content_sim_matrix, movie_id_to_idx, global_mean):
    if movie_id not in movie_id_to_idx: return global_mean
    user_ratings = train_df[train_df['User_ID'] == user_id]
    if user_ratings.empty: return global_mean

    target_movie_idx = movie_id_to_idx[movie_id]
    sim_scores = content_sim_matrix[target_movie_idx]
    
    numerator, denominator = 0, 0
    for _, row in user_ratings.iterrows():
        rated_movie_id = row['Movie_ID']
        if rated_movie_id in movie_id_to_idx:
            rated_movie_idx = movie_id_to_idx[rated_movie_id]
            similarity = sim_scores[rated_movie_idx]
            if similarity > 0:
                numerator += similarity * row['Rating']
                denominator += similarity
    
    return numerator / denominator if denominator > 0 else global_mean

# --- Evaluation ---
if __name__ == "__main__":
    print("Loading and preparing data...")
    merged_df, user_ratings = load_and_prepare_data()
    
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(user_ratings[['User_ID', 'Movie_ID', 'Rating']], reader)
    trainset = data.build_full_trainset()
    
    train_df, test_df = train_test_split(user_ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    global_mean_rating = train_df['Rating'].mean()

    print("Building content-based similarity matrix with OPTIMIZED TF-IDF features...")
    
    w_overview, w_genre, w_title, w_director = 4, 3, 2, 1
    soup = (
        (merged_df['Overview'].fillna('') + ' ') * w_overview +
        (merged_df['Genre'].fillna('') + ' ') * w_genre +
        (merged_df['Series_Title'].fillna('') + ' ') * w_title +
        (merged_df['Director'].fillna('') + ' ') * w_director
    )

    tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(soup)
    content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    merged_df_reset = merged_df.reset_index(drop=True)
    movie_id_to_idx = {mid: i for i, mid in enumerate(merged_df_reset['Movie_ID'])}

    print("Training SVD model...")
    svd = SVD(n_epochs=25, n_factors=100, random_state=RANDOM_STATE)
    svd.fit(trainset)

    print("Evaluating models...")
    predictions = defaultdict(list)
    true_ratings = list(test_df['Rating'])

    for _, row in test_df.iterrows():
        user_id, movie_id = row['User_ID'], row['Movie_ID']
        pred_svd = svd.predict(user_id, movie_id).est
        predictions['collab'].append(pred_svd)
        pred_cb = predict_content_based(user_id, movie_id, train_df, content_sim_matrix, movie_id_to_idx, global_mean_rating)
        predictions['content'].append(pred_cb)
        predictions['hybrid'].append(0.8 * pred_svd + 0.2 * pred_cb)
    
    print("\n--- Evaluation Results ---\n")
    results = {}
    y_true_cls = [1 if r >= RATING_THRESHOLD else 0 for r in true_ratings]
    for model_name in ['content', 'collab', 'hybrid']:
        y_pred_reg = np.clip(predictions[model_name], 1, 10)
        y_pred_cls = [1 if r >= RATING_THRESHOLD else 0 for r in y_pred_reg]
        
        results[model_name] = {
            'report': classification_report(y_true_cls, y_pred_cls, target_names=['negative', 'positive'], zero_division=0),
            'accuracy': accuracy_score(y_true_cls, y_pred_cls),
            'rmse': np.sqrt(mean_squared_error(true_ratings, y_pred_reg)),
            'precision': precision_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0),
            'recall': recall_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)
        }
        title = model_name.replace('content', 'Content-Based (Optimized)').replace('collab', 'Collaborative (SVD)').replace('hybrid', 'Hybrid (SVD-Heavy)')
        print(f"Model: {title}\nAccuracy: {results[model_name]['accuracy']:.3f}\n{results[model_name]['report']}\n{'-'*30}")

    print("\n--- Performance Comparison ---\n")
    summary_df = pd.DataFrame({
        "Method Used": [name.replace('content', 'Content-Based (Optimized)').replace('collab', 'Collaborative (SVD)').replace('hybrid', 'Hybrid (SVD-Heavy)') for name in results.keys()],
        "Precision": [res['precision'] for res in results.values()],
        "Recall": [res['recall'] for res in results.values()],
        "RMSE": [res['rmse'] for res in results.values()]
    })
    print(summary_df.to_string(index=False, formatters={'Precision':'{:.2f}'.format, 'Recall':'{:.2f}'.format, 'RMSE':'{:.2f}'.format}))