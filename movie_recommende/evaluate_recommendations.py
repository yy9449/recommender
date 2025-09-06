# evaluate_recommendations.py

import pandas as pd
import numpy as np
import os  # <-- Added OS import
import sys # <-- Added Sys import
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    mean_squared_error, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
RATING_THRESHOLD = 7.0
TEST_SIZE_PER_USER = 0.3
RANDOM_STATE = 42

# --- Data Loading and Preprocessing ---
def load_and_prepare_data():
    """Loads all datasets, merges them, and cleans data."""
    movies = pd.read_csv('movies.csv')
    imdb = pd.read_csv('imdb_top_1000.csv')
    user_ratings = pd.read_csv('user_movie_rating.csv')

    if 'Movie_ID' not in movies.columns:
        movies['Movie_ID'] = range(len(movies))

    imdb['Released_Year'] = pd.to_numeric(imdb['Released_Year'], errors='coerce')
    imdb['No_of_Votes'] = pd.to_numeric(imdb['No_of_Votes'], errors='coerce')
    merged_df = pd.merge(movies, imdb, on='Series_Title', how='left')
    
    merged_df['IMDB_Rating'] = merged_df['IMDB_Rating'].fillna(merged_df['IMDB_Rating'].mean())
    merged_df['No_of_Votes'] = merged_df['No_of_Votes'].fillna(0)
    merged_df['Released_Year'] = merged_df['Released_Year'].fillna(merged_df['Released_Year'].mode()[0])
    merged_df = merged_df.dropna(subset=['Movie_ID']).drop_duplicates(subset=['Movie_ID'])
    merged_df['Movie_ID'] = merged_df['Movie_ID'].astype(int)

    return merged_df, user_ratings

def create_train_test_split(user_ratings):
    """Splits ratings data into train and test sets."""
    user_counts = user_ratings['User_ID'].value_counts()
    active_users = user_counts[user_counts >= 2].index
    ratings_subset = user_ratings[user_ratings['User_ID'].isin(active_users)]

    train, test = train_test_split(ratings_subset, test_size=TEST_SIZE_PER_USER, stratify=ratings_subset['User_ID'], random_state=RANDOM_STATE)
    return train, test

# --- Prediction Models ---
def predict_rating(user_id, movie_id, train_df, sim_matrix, movie_id_to_idx):
    """Generic prediction function for content-based and collaborative models."""
    if movie_id not in movie_id_to_idx:
        return train_df['Rating'].mean()

    user_ratings = train_df[train_df['User_ID'] == user_id]
    if user_ratings.empty:
        return train_df['Rating'].mean()

    target_movie_idx = movie_id_to_idx[movie_id]
    sim_scores = sim_matrix[target_movie_idx]
    
    numerator = 0
    denominator = 0
    
    for _, row in user_ratings.iterrows():
        rated_movie_id = row['Movie_ID']
        if rated_movie_id in movie_id_to_idx:
            rated_movie_idx = movie_id_to_idx[rated_movie_id]
            similarity = sim_scores[rated_movie_idx]
            rating = row['Rating']
            numerator += similarity * rating
            denominator += similarity
    
    return numerator / denominator if denominator > 0 else train_df['Rating'].mean()

def predict_rating_hybrid(user_id, movie_id, train_df, content_sim, collab_sim, movie_id_to_idx):
    """Predicts a rating using a simple hybrid blend."""
    alpha, beta = 0.5, 0.5
    pred_content = predict_rating(user_id, movie_id, train_df, content_sim, movie_id_to_idx)
    pred_collab = predict_rating(user_id, movie_id, train_df, collab_sim, movie_id_to_idx)
    return (alpha * pred_content) + (beta * pred_collab)

# --- Evaluation Metrics ---
def compute_metrics(y_true, y_pred, y_true_cls, y_pred_cls):
    """Computes and returns a dictionary of metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'accuracy': accuracy_score(y_true_cls, y_pred_cls),
        'precision': precision_score(y_true_cls, y_pred_cls, average='weighted'),
        'recall': recall_score(y_true_cls, y_pred_cls, average='weighted'),
        'f1': f1_score(y_true_cls, y_pred_cls, average='weighted'),
        'report': classification_report(y_true_cls, y_pred_cls, target_names=['negative', 'positive'])
    }

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading and preparing data...")
    merged_df, user_ratings = load_and_prepare_data()
    train_df, test_df = create_train_test_split(user_ratings)
    
    merged_df = merged_df.reset_index(drop=True)
    movie_id_to_idx = {mid: i for i, mid in enumerate(merged_df['Movie_ID'])}
    
    print("Building similarity matrices...")
    # Content-Based
    soup = (merged_df['Genre'].fillna('') + ' ' + merged_df['Series_Title'].fillna(''))
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(soup)
    content_sim_matrix = cosine_similarity(tfidf_matrix)

    # Collaborative
    aligned_user_item = train_df.pivot_table(index='Movie_ID', columns='User_ID', values='Rating').reindex(merged_df['Movie_ID']).fillna(0)
    collab_sim_matrix = cosine_similarity(aligned_user_item.values)

    print("Evaluating models on test set...")
    predictions = defaultdict(list)
    true_ratings = list(test_df['Rating'])

    for _, row in test_df.iterrows():
        user_id, movie_id = row['User_ID'], row['Movie_ID']
        predictions['content'].append(predict_rating(user_id, movie_id, train_df, content_sim_matrix, movie_id_to_idx))
        predictions['collab'].append(predict_rating(user_id, movie_id, train_df, collab_sim_matrix, movie_id_to_idx))
        predictions['hybrid'].append(predict_rating_hybrid(user_id, movie_id, train_df, content_sim_matrix, collab_sim_matrix, movie_id_to_idx))

    print("\n--- Evaluation Results ---\n")
    results = {}
    
    for model in predictions:
        predictions[model] = np.clip(predictions[model], 1, 10)

    y_true_cls = [1 if r >= RATING_THRESHOLD else 0 for r in true_ratings]

    for model_name in ['content', 'collab', 'hybrid']:
        y_pred_reg = predictions[model_name]
        y_pred_cls = [1 if r >= RATING_THRESHOLD else 0 for r in y_pred_reg]
        results[model_name] = compute_metrics(true_ratings, y_pred_reg, y_true_cls, y_pred_cls)
        
        title = model_name.replace('content', 'Content-Based').replace('collab', 'Collaborative').replace('hybrid', 'Hybrid')
        print(f"Model: {title}")
        print(f"Accuracy: {results[model_name]['accuracy']:.3f}")
        print(results[model_name]['report'])
        print("-" * 30)

    # --- Final Summary Table ---
    print("\n--- Performance Comparison ---\n")
    summary_data = [
        {
            "Method Used": name.replace('content', 'Content-Based').replace('collab', 'Collaborative').replace('hybrid', 'Hybrid'),
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "RMSE": metrics['rmse'],
        } for name, metrics in results.items()
    ]

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, formatters={'Precision':'{:.2f}'.format, 'Recall':'{:.2f}'.format, 'RMSE':'{:.2f}'.format}))