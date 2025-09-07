#!/usr/bin/env python3
"""
Simple Movie Recommendation System Evaluation
Usage: python evaluate_recommendations.py
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

def load_data():
    """Load datasets"""
    print("Loading datasets...")
    
    # Load movies and IMDB data
    movies_df = pd.read_csv("movies.csv")
    imdb_df = pd.read_csv("imdb_top_1000.csv")
    
    # Add Movie_ID if missing
    if 'Movie_ID' not in movies_df.columns:
        movies_df['Movie_ID'] = range(len(movies_df))
    
    # Merge datasets
    merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
    merged_df = merged_df.drop_duplicates(subset="Series_Title")
    
    # Ensure Movie_ID is preserved
    if 'Movie_ID' not in merged_df.columns:
        merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on="Series_Title", how="inner")
    
    print(f"Dataset loaded: {len(merged_df)} movies")
    return merged_df

def import_algorithms():
    """Import recommendation functions"""
    try:
        from content_based import content_based_filtering_enhanced
        from collaborative import collaborative_filtering_enhanced
        from hybrid import smart_hybrid_recommendation
        
        return {
            'Content-Based': content_based_filtering_enhanced,
            'Collaborative': collaborative_filtering_enhanced,
            'Hybrid': smart_hybrid_recommendation
        }
    except ImportError as e:
        print(f"Error importing algorithms: {e}")
        return None

def create_test_movies(merged_df):
    """Create test set"""
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    
    # Select popular, high-rated movies
    test_candidates = merged_df[
        (merged_df[rating_col] >= 7.5) & 
        (merged_df['No_of_Votes'] >= 100000)
    ]
    
    # Sample 15 movies for testing
    test_movies = test_candidates.sample(min(15, len(test_candidates)))['Series_Title'].tolist()
    return test_movies

def evaluate_algorithm(merged_df, algorithm_func, algorithm_name, test_movies):
    """Evaluate single algorithm"""
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    precision_scores = []
    recall_scores = []
    mse_scores = []
    successful_tests = 0
    
    for test_movie in test_movies:
        try:
            # Get baseline movie info
            movie_info = merged_df[merged_df['Series_Title'] == test_movie]
            if movie_info.empty:
                continue
                
            baseline = movie_info.iloc[0]
            baseline_rating = baseline[rating_col]
            baseline_genres = set()
            
            if pd.notna(baseline[genre_col]):
                baseline_genres = set(g.strip() for g in baseline[genre_col].split(','))
            
            # Get recommendations
            if algorithm_name == 'Hybrid':
                primary_genre = list(baseline_genres)[0] if baseline_genres else None
                recs = algorithm_func(merged_df, test_movie, primary_genre, 10)
            else:
                if algorithm_name == 'Collaborative':
                    recs = algorithm_func(merged_df, test_movie, 10)
                else:  # Content-Based
                    recs = algorithm_func(merged_df, test_movie, None, 10)
            
            if recs is None or recs.empty:
                continue
            
            successful_tests += 1
            
            # Calculate relevance
            relevant_count = 0
            predicted_ratings = []
            
            for _, rec in recs.iterrows():
                rec_rating = rec[rating_col]
                rec_genres = set()
                
                if pd.notna(rec[genre_col]):
                    rec_genres = set(g.strip() for g in rec[genre_col].split(','))
                
                # Relevance: genre overlap OR similar rating (within 1.5 points)
                genre_overlap = len(baseline_genres & rec_genres) > 0 if baseline_genres else True
                rating_similar = abs(rec_rating - baseline_rating) <= 1.5 if pd.notna(rec_rating) else False
                
                if genre_overlap or rating_similar:
                    relevant_count += 1
                
                if pd.notna(rec_rating):
                    predicted_ratings.append(rec_rating)
            
            # Calculate metrics
            precision = relevant_count / len(recs) if len(recs) > 0 else 0
            precision_scores.append(precision)
            
            # Recall (estimate 12 total relevant items)
            recall = min(relevant_count / 12, 1.0)
            recall_scores.append(recall)
            
            # MSE for rating prediction
            if predicted_ratings:
                avg_predicted = np.mean(predicted_ratings)
                mse_scores.append((avg_predicted - baseline_rating) ** 2)
            
        except Exception:
            continue
    
    # Calculate final metrics
    if successful_tests > 0:
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        rmse = np.sqrt(np.mean(mse_scores)) if mse_scores else float('inf')
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score,
            'rmse': rmse,
            'success_rate': successful_tests / len(test_movies)
        }
    
    return None

def main():
    """Main evaluation function"""
    print("Movie Recommendation System Evaluation")
    print("=" * 50)
    
    # Load data
    try:
        merged_df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Import algorithms
    algorithms = import_algorithms()
    if not algorithms:
        return
    
    # Create test set
    test_movies = create_test_movies(merged_df)
    print(f"Testing with {len(test_movies)} movies")
    print()
    
    # Evaluate each algorithm
    results = {}
    for alg_name, alg_func in algorithms.items():
        print(f"Evaluating {alg_name}...")
        result = evaluate_algorithm(merged_df, alg_func, alg_name, test_movies)
        if result:
            results[alg_name] = result
        print(f"  Completed: {result['success_rate']*100:.0f}% success rate" if result else "  Failed")
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    if results:
        # Header
        print(f"{'Algorithm':<15} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'RMSE':<8} {'Success%':<9}")
        print("-" * 70)
        
        # Results
        for alg_name, metrics in results.items():
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1_score']
            rmse = metrics['rmse']
            success = metrics['success_rate'] * 100
            
            rmse_str = f"{rmse:.3f}" if rmse != float('inf') else "N/A"
            print(f"{alg_name:<15} {precision:<10.3f} {recall:<8.3f} {f1:<9.3f} {rmse_str:<8} {success:<9.1f}")
        
        # Find best algorithm
        best_alg = max(results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\nBest Overall: {best_alg[0]} (F1-Score: {best_alg[1]['f1_score']:.3f})")
        
    else:
        print("No results to display - all algorithms failed")

if __name__ == "__main__":
    main()
