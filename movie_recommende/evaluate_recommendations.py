#!/usr/bin/env python3
"""
Movie Recommendation System Evaluation
Run this script in terminal to evaluate your system's performance
Usage: python evaluate_recommendations.py
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import json

warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """Terminal-based evaluation system for movie recommendations"""
    
    def __init__(self):
        self.merged_df = None
        self.user_ratings_df = None
        self.test_users = {}
        self.algorithms = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare datasets"""
        print("=" * 60)
        print("MOVIE RECOMMENDATION SYSTEM EVALUATION")
        print("=" * 60)
        print("\n[1/5] Loading datasets...")
        
        # Load movies and IMDB data
        datasets = {}
        
        for filename in ["movies.csv", "imdb_top_1000.csv"]:
            found = False
            for path in [filename, f"./{filename}", f"data/{filename}", f"../{filename}"]:
                if os.path.exists(path):
                    datasets[filename] = pd.read_csv(path)
                    print(f"✅ Found {filename} at: {path}")
                    found = True
                    break
            
            if not found:
                print(f"❌ {filename} not found!")
                return False
        
        # Load user ratings if available
        user_ratings_found = False
        for path in ["user_movie_rating.csv", "./user_movie_rating.csv", "data/user_movie_rating.csv"]:
            if os.path.exists(path):
                self.user_ratings_df = pd.read_csv(path)
                user_ratings_found = True
                print(f"✅ Found user_movie_rating.csv at: {path}")
                break
        
        if not user_ratings_found:
            print("⚠️ user_movie_rating.csv not found - will use synthetic evaluation")
        
        # Prepare merged dataset
        movies_df = datasets["movies.csv"]
        imdb_df = datasets["imdb_top_1000.csv"]
        
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
        
        self.merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
        self.merged_df = self.merged_df.drop_duplicates(subset="Series_Title")
        
        if 'Movie_ID' not in self.merged_df.columns:
            self.merged_df = pd.merge(
                movies_df[['Movie_ID', 'Series_Title']], 
                self.merged_df, 
                on="Series_Title", 
                how="inner"
            )
        
        print(f"📊 Dataset ready: {len(self.merged_df)} movies")
        return True
    
    def import_algorithms(self):
        """Import recommendation algorithms"""
        print("\n[2/5] Importing recommendation algorithms...")
        
        try:
            # Add current directory to path for imports
            sys.path.append(os.getcwd())
            
            from content_based import content_based_filtering_enhanced
            from collaborative import collaborative_filtering_enhanced
            from hybrid import smart_hybrid_recommendation
            
            self.algorithms = {
                'content_based': content_based_filtering_enhanced,
                'collaborative': collaborative_filtering_enhanced,
                'hybrid': smart_hybrid_recommendation
            }
            
            print("✅ Successfully imported all algorithms")
            return True
            
        except ImportError as e:
            print(f"❌ Failed to import algorithms: {e}")
            print("Make sure content_based.py, collaborative.py, and hybrid.py are in the current directory")
            return False
    
    def create_test_set(self):
        """Create test set for evaluation"""
        print("\n[3/5] Creating test set...")
        
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in self.merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in self.merged_df.columns else 'Genre'
        
        # Select diverse test movies
        test_movies = []
        
        # High-quality popular movies
        popular_movies = self.merged_df[
            (self.merged_df[rating_col] >= 7.5) & 
            (self.merged_df['No_of_Votes'] >= 100000)
        ]
        
        # Sample across different genres
        genres_to_test = ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance', 'Crime', 'Horror', 'Sci-Fi']
        
        for genre in genres_to_test:
            genre_movies = popular_movies[
                popular_movies[genre_col].str.contains(genre, na=False, case=False)
            ]
            if not genre_movies.empty:
                sample_size = min(2, len(genre_movies))
                test_movies.extend(genre_movies.sample(sample_size)['Series_Title'].tolist())
        
        # Add some general popular movies
        remaining = popular_movies[~popular_movies['Series_Title'].isin(test_movies)]
        if not remaining.empty:
            additional = min(5, len(remaining))
            test_movies.extend(remaining.sample(additional)['Series_Title'].tolist())
        
        # Limit to 20 test cases for manageable evaluation
        self.test_movies = test_movies[:20]
        
        print(f"📋 Created test set with {len(self.test_movies)} movies:")
        for i, movie in enumerate(self.test_movies[:5], 1):
            print(f"   {i}. {movie}")
        if len(self.test_movies) > 5:
            print(f"   ... and {len(self.test_movies) - 5} more")
        
        return True
    
    def evaluate_recommendations(self):
        """Evaluate recommendation quality"""
        print("\n[4/5] Running evaluation...")
        
        rating_col = 'IMDB_Rating' if 'IMDB_Rating' in self.merged_df.columns else 'Rating'
        genre_col = 'Genre_y' if 'Genre_y' in self.merged_df.columns else 'Genre'
        
        results = {}
        
        for alg_name, alg_func in self.algorithms.items():
            print(f"\n  Testing {alg_name.title().replace('_', '-')} Algorithm...")
            
            # Metrics storage
            relevance_scores = []
            rating_predictions = []
            actual_ratings = []
            precision_scores = []
            recall_scores = []
            
            successful_tests = 0
            
            for i, test_movie in enumerate(self.test_movies):
                try:
                    # Get baseline movie info
                    movie_info = self.merged_df[self.merged_df['Series_Title'] == test_movie]
                    if movie_info.empty:
                        continue
                    
                    baseline = movie_info.iloc[0]
                    baseline_rating = baseline[rating_col]
                    baseline_genres = set()
                    
                    if pd.notna(baseline[genre_col]):
                        baseline_genres = set(g.strip() for g in baseline[genre_col].split(','))
                    
                    # Get recommendations
                    if alg_name == 'hybrid':
                        primary_genre = list(baseline_genres)[0] if baseline_genres else None
                        recs = alg_func(self.merged_df, test_movie, primary_genre, 10)
                    else:
                        recs = alg_func(self.merged_df, test_movie, None, 10)
                    
                    if recs is None or recs.empty:
                        continue
                    
                    successful_tests += 1
                    
                    # Calculate relevance (genre overlap + rating similarity)
                    relevant_count = 0
                    rec_ratings = []
                    
                    for _, rec in recs.iterrows():
                        rec_rating = rec[rating_col]
                        rec_genres = set()
                        
                        if pd.notna(rec[genre_col]):
                            rec_genres = set(g.strip() for g in rec[genre_col].split(','))
                        
                        # Relevance criteria:
                        # 1. Genre overlap OR
                        # 2. Similar rating (within 1.5 points)
                        genre_overlap = len(baseline_genres & rec_genres) > 0 if baseline_genres else True
                        rating_similar = abs(rec_rating - baseline_rating) <= 1.5 if pd.notna(rec_rating) else False
                        
                        if genre_overlap or rating_similar:
                            relevant_count += 1
                        
                        if pd.notna(rec_rating):
                            rec_ratings.append(rec_rating)
                    
                    # Precision calculation
                    precision = relevant_count / len(recs) if len(recs) > 0 else 0
                    precision_scores.append(precision)
                    
                    # Recall calculation (simulate total relevant items as 15)
                    estimated_total_relevant = 15  # Realistic estimate
                    recall = min(relevant_count / estimated_total_relevant, 1.0)
                    recall_scores.append(recall)
                    
                    # Rating prediction accuracy
                    if rec_ratings:
                        avg_predicted_rating = np.mean(rec_ratings)
                        rating_predictions.append(avg_predicted_rating)
                        actual_ratings.append(baseline_rating)
                    
                    # Progress indicator
                    if (i + 1) % 5 == 0:
                        progress = ((i + 1) / len(self.test_movies)) * 100
                        print(f"    Progress: {progress:.0f}% ({successful_tests} successful)")
                
                except Exception as e:
                    continue
            
            # Calculate final metrics
            if successful_tests > 0:
                avg_precision = np.mean(precision_scores) if precision_scores else 0
                avg_recall = np.mean(recall_scores) if recall_scores else 0
                f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
                
                # Rating prediction metrics
                mse = mean_squared_error(actual_ratings, rating_predictions) if len(actual_ratings) > 0 else float('inf')
                rmse = np.sqrt(mse) if mse != float('inf') else float('inf')
                
                results[alg_name] = {
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1_score': f1_score,
                    'mse': mse,
                    'rmse': rmse,
                    'success_rate': successful_tests / len(self.test_movies),
                    'successful_tests': successful_tests
                }
                
                print(f"    Completed: {successful_tests}/{len(self.test_movies)} tests successful")
            else:
                print(f"    ❌ No successful tests for {alg_name}")
        
        self.results = results
        return True
    
    def display_results(self):
        """Display evaluation results in terminal"""
        print("\n[5/5] Evaluation Results")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        # Header
        header = f"{'Algorithm':<15} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'RMSE':<8} {'Success%':<9}"
        print(header)
        print("-" * 80)
        
        # Results for each algorithm
        best_precision = 0
        best_recall = 0
        best_f1 = 0
        best_rmse = float('inf')
        
        for alg_name, metrics in self.results.items():
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1_score']
            rmse = metrics['rmse']
            success_rate = metrics['success_rate'] * 100
            
            # Track best scores
            if precision > best_precision:
                best_precision = precision
            if recall > best_recall:
                best_recall = recall
            if f1 > best_f1:
                best_f1 = f1
            if rmse < best_rmse:
                best_rmse = rmse
            
            # Display row
            rmse_str = f"{rmse:.3f}" if rmse != float('inf') else "N/A"
            row = f"{alg_name.replace('_', '-').title():<15} {precision:<10.3f} {recall:<8.3f} {f1:<9.3f} {rmse_str:<8} {success_rate:<9.1f}"
            print(row)
        
        # Summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"📊 Test Set Size: {len(self.test_movies)} movies")
        print(f"🎯 Best Precision: {best_precision:.3f} (higher is better)")
        print(f"🔍 Best Recall: {best_recall:.3f} (higher is better)")
        print(f"⚖️  Best F1-Score: {best_f1:.3f} (higher is better)")
        if best_rmse != float('inf'):
            print(f"📈 Best RMSE: {best_rmse:.3f} (lower is better)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        best_overall = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_alg_name = best_overall[0].replace('_', '-').title()
        
        print(f"   • Best Overall Algorithm: {best_alg_name} (F1-Score: {best_overall[1]['f1_score']:.3f})")
        
        if best_precision >= 0.7:
            print("   • Precision is good - recommendations are mostly relevant")
        elif best_precision >= 0.5:
            print("   • Precision is moderate - consider improving relevance criteria")
        else:
            print("   • Precision needs improvement - many irrelevant recommendations")
        
        if best_recall >= 0.6:
            print("   • Recall is good - finding many relevant items")
        elif best_recall >= 0.4:
            print("   • Recall is moderate - missing some relevant items")
        else:
            print("   • Recall needs improvement - missing too many relevant items")
        
        if best_rmse != float('inf') and best_rmse <= 1.0:
            print("   • Rating predictions are accurate")
        elif best_rmse != float('inf') and best_rmse <= 1.5:
            print("   • Rating predictions are reasonably accurate")
        else:
            print("   • Rating predictions need improvement")
    
    def save_results(self):
        """Save results to file"""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        filename = f"evaluation_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_data = []
        for alg_name, metrics in self.results.items():
            csv_data.append({
                'Algorithm': alg_name.replace('_', '-').title(),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'Success_Rate': metrics['success_rate']
            })
        
        csv_filename = f"evaluation_summary_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_filename, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   • Detailed: {filename}")
        print(f"   • Summary: {csv_filename}")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        try:
            if not self.load_data():
                return False
            
            if not self.import_algorithms():
                return False
            
            if not self.create_test_set():
                return False
            
            if not self.evaluate_recommendations():
                return False
            
            self.display_results()
            self.save_results()
            
            print(f"\n✅ Evaluation completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {str(e)}")
            return False

def main():
    """Main function"""
    evaluator = RecommendationEvaluator()
    success = evaluator.run_evaluation()
    
    if not success:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure all CSV files are in the current directory")
        print("2. Check that recommendation modules are properly implemented")
        print("3. Verify Python environment has required packages:")
        print("   pip install pandas numpy scikit-learn")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
