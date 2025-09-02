import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import sys
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    def __init__(self, merged_df, user_ratings_df=None):
        """
        Initialize evaluator with movie data and user ratings
        
        Args:
            merged_df: Combined movie dataset with features
            user_ratings_df: User-movie ratings (User_ID, Movie_ID, Rating)
        """
        self.merged_df = merged_df
        self.user_ratings_df = user_ratings_df
        self.rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        self.genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
        
        # Import your recommendation modules
        try:
            from content_based import content_based_filtering_enhanced
            from collaborative import collaborative_filtering_enhanced, create_synthetic_user_profiles_enhanced
            from hybrid import smart_hybrid_recommendation
            
            self.content_based_func = content_based_filtering_enhanced
            self.collaborative_func = collaborative_filtering_enhanced
            self.hybrid_func = smart_hybrid_recommendation
            self.synthetic_func = create_synthetic_user_profiles_enhanced
        except ImportError as e:
            print(f"Warning: Could not import recommendation modules: {e}")
            self.content_based_func = None
            self.collaborative_func = None
            self.hybrid_func = None
    
    def create_evaluation_dataset(self, test_size=0.2, min_ratings_per_user=5):
        """
        Create train/test split for evaluation
        
        Returns:
            train_data, test_data, test_users
        """
        if self.user_ratings_df is None:
            print("No user ratings data available. Creating synthetic evaluation data...")
            return self.create_synthetic_evaluation_data()
        
        print(f"Creating evaluation dataset from {len(self.user_ratings_df)} real user ratings...")
        
        # Filter users with minimum ratings
        user_counts = self.user_ratings_df['User_ID'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        
        filtered_ratings = self.user_ratings_df[
            self.user_ratings_df['User_ID'].isin(valid_users)
        ].copy()
        
        print(f"Using {len(valid_users)} users with at least {min_ratings_per_user} ratings each")
        
        # Split by users to avoid data leakage
        train_users, test_users = train_test_split(
            valid_users, test_size=test_size, random_state=42
        )
        
        train_data = filtered_ratings[filtered_ratings['User_ID'].isin(train_users)]
        test_data = filtered_ratings[filtered_ratings['User_ID'].isin(test_users)]
        
        print(f"Train set: {len(train_data)} ratings from {len(train_users)} users")
        print(f"Test set: {len(test_data)} ratings from {len(test_users)} users")
        
        return train_data, test_data, test_users
    
    def create_synthetic_evaluation_data(self, n_users=100, test_size=0.2):
        """Create synthetic data for evaluation when real data isn't available"""
        print("Creating synthetic user profiles for evaluation...")
        
        # Generate synthetic user-movie matrix
        user_movie_matrix, user_names = self.synthetic_func(self.merged_df, n_users=n_users)
        
        # Convert to ratings dataframe
        ratings_data = []
        for user_idx, user_name in enumerate(user_names):
            for movie_idx, rating in enumerate(user_movie_matrix[user_idx]):
                if rating > 0:  # Only include actual ratings
                    movie_id = self.merged_df.iloc[movie_idx].get('Movie_ID', movie_idx)
                    ratings_data.append({
                        'User_ID': user_idx,
                        'Movie_ID': movie_id,
                        'Rating': rating
                    })
        
        synthetic_ratings_df = pd.DataFrame(ratings_data)
        
        # Split into train/test
        train_users, test_users = train_test_split(
            list(range(n_users)), test_size=test_size, random_state=42
        )
        
        train_data = synthetic_ratings_df[synthetic_ratings_df['User_ID'].isin(train_users)]
        test_data = synthetic_ratings_df[synthetic_ratings_df['User_ID'].isin(test_users)]
        
        print(f"Synthetic data created - Train: {len(train_data)} ratings, Test: {len(test_data)} ratings")
        
        return train_data, test_data, test_users
    
    def calculate_precision_recall_at_k(self, true_ratings, predicted_movies, k=5, rating_threshold=4.0):
        """
        Calculate Precision@K and Recall@K for recommendation quality
        
        Args:
            true_ratings: Dict of {movie_title: rating} for user's actual ratings
            predicted_movies: List of recommended movie titles
            k: Number of top recommendations to consider
            rating_threshold: Minimum rating to consider as "relevant"
        """
        if not predicted_movies or not true_ratings:
            return 0.0, 0.0
        
        # Get top-k predictions
        top_k_predictions = predicted_movies[:k]
        
        # Find relevant items in true ratings (above threshold)
        relevant_items = {title for title, rating in true_ratings.items() 
                         if rating >= rating_threshold}
        
        # Find relevant items in predictions
        relevant_predictions = {title for title in top_k_predictions 
                              if title in relevant_items}
        
        # Calculate metrics
        precision = len(relevant_predictions) / len(top_k_predictions) if top_k_predictions else 0.0
        recall = len(relevant_predictions) / len(relevant_items) if relevant_items else 0.0
        
        return precision, recall
    
    def evaluate_rating_prediction(self, true_ratings, predicted_ratings):
        """
        Evaluate rating prediction accuracy using RMSE and MAE
        
        Args:
            true_ratings: Dict of {movie_title: actual_rating}
            predicted_ratings: Dict of {movie_title: predicted_rating}
        """
        common_movies = set(true_ratings.keys()) & set(predicted_ratings.keys())
        
        if not common_movies:
            return float('inf'), float('inf')
        
        true_values = [true_ratings[movie] for movie in common_movies]
        pred_values = [predicted_ratings[movie] for movie in common_movies]
        
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        mae = mean_absolute_error(true_values, pred_values)
        
        return rmse, mae
    
    def get_user_movie_ratings(self, user_id, ratings_data):
        """Get all ratings for a specific user"""
        user_ratings = ratings_data[ratings_data['User_ID'] == user_id]
        ratings_dict = {}
        
        for _, row in user_ratings.iterrows():
            # Map Movie_ID to title
            movie_info = self.merged_df[self.merged_df['Movie_ID'] == row['Movie_ID']]
            if not movie_info.empty:
                movie_title = movie_info.iloc[0]['Series_Title']
                ratings_dict[movie_title] = row['Rating']
        
        return ratings_dict
    
    def predict_with_algorithm(self, algorithm_name, target_movie, genre=None, top_n=10):
        """
        Get predictions from specified algorithm
        
        Returns:
            List of recommended movie titles
        """
        try:
            if algorithm_name == 'content_based':
                if self.content_based_func:
                    results = self.content_based_func(self.merged_df, target_movie, genre, top_n)
                else:
                    return []
            elif algorithm_name == 'collaborative':
                if self.collaborative_func:
                    results = self.collaborative_func(self.merged_df, target_movie, top_n)
                else:
                    return []
            elif algorithm_name == 'hybrid':
                if self.hybrid_func:
                    results = self.hybrid_func(self.merged_df, target_movie, genre, top_n)
                else:
                    return []
            else:
                return []
            
            if results is not None and not results.empty:
                return results['Series_Title'].tolist()
            return []
            
        except Exception as e:
            print(f"Error in {algorithm_name}: {e}")
            return []
    
    def evaluate_single_algorithm(self, algorithm_name, test_data, test_users, k=5):
        """
        Evaluate a single recommendation algorithm
        
        Returns:
            Dict with evaluation metrics
        """
        print(f"\nEvaluating {algorithm_name.upper()} algorithm...")
        
        precisions = []
        recalls = []
        f1_scores = []
        rmses = []
        maes = []
        successful_predictions = 0
        
        # Sample subset of test users for faster evaluation
        sample_users = np.random.choice(test_users, min(20, len(test_users)), replace=False)
        
        for user_id in sample_users:
            user_ratings = self.get_user_movie_ratings(user_id, test_data)
            
            if len(user_ratings) < 2:  # Need at least 2 ratings for meaningful evaluation
                continue
            
            # Pick a random movie from user's high-rated movies as query
            high_rated_movies = {title: rating for title, rating in user_ratings.items() 
                               if rating >= 4.0}
            
            if not high_rated_movies:
                continue
            
            target_movie = np.random.choice(list(high_rated_movies.keys()))
            
            # Get movie's primary genre for hybrid evaluation
            movie_info = self.merged_df[self.merged_df['Series_Title'] == target_movie]
            primary_genre = None
            if not movie_info.empty and pd.notna(movie_info.iloc[0][self.genre_col]):
                genres = movie_info.iloc[0][self.genre_col].split(',')
                primary_genre = genres[0].strip() if genres else None
            
            # Get recommendations
            if algorithm_name == 'hybrid':
                predicted_movies = self.predict_with_algorithm(algorithm_name, target_movie, primary_genre, k*2)
            else:
                predicted_movies = self.predict_with_algorithm(algorithm_name, target_movie, None, k*2)
            
            if not predicted_movies:
                continue
            
            successful_predictions += 1
            
            # Calculate precision and recall
            precision, recall = self.calculate_precision_recall_at_k(
                user_ratings, predicted_movies, k, rating_threshold=4.0
            )
            
            precisions.append(precision)
            recalls.append(recall)
            
            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)
            
            # For rating prediction, use IMDB ratings as proxy
            predicted_ratings = {}
            for movie_title in predicted_movies[:k]:
                movie_info = self.merged_df[self.merged_df['Series_Title'] == movie_title]
                if not movie_info.empty:
                    # Convert IMDB rating to 1-5 scale
                    imdb_rating = movie_info.iloc[0][self.rating_col]
                    if pd.notna(imdb_rating):
                        predicted_ratings[movie_title] = (imdb_rating / 10.0) * 5.0
            
            if predicted_ratings:
                rmse, mae = self.evaluate_rating_prediction(user_ratings, predicted_ratings)
                if rmse != float('inf'):
                    rmses.append(rmse)
                    maes.append(mae)
        
        # Calculate average metrics
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_rmse = np.mean(rmses) if rmses else float('inf')
        avg_mae = np.mean(maes) if maes else float('inf')
        
        return {
            'algorithm': algorithm_name,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'rmse': avg_rmse,
            'mae': avg_mae,
            'successful_predictions': successful_predictions,
            'total_users_tested': len(sample_users)
        }
    
    def comprehensive_evaluation(self, k=5):
        """
        Run comprehensive evaluation of all algorithms
        
        Returns:
            DataFrame with results for all algorithms
        """
        print("=" * 60)
        print("🎬 MOVIE RECOMMENDATION SYSTEM EVALUATION")
        print("=" * 60)
        
        # Create evaluation dataset
        train_data, test_data, test_users = self.create_evaluation_dataset()
        
        if test_data is None or test_data.empty:
            print("❌ Cannot create evaluation dataset. Check your data.")
            return None
        
        # Evaluate each algorithm
        algorithms = ['content_based', 'collaborative', 'hybrid']
        results = []
        
        for algorithm in algorithms:
            try:
                metrics = self.evaluate_single_algorithm(algorithm, test_data, test_users, k)
                results.append(metrics)
                
                print(f"\n✅ {algorithm.upper()} Results:")
                print(f"   Precision@{k}: {metrics['precision']:.3f}")
                print(f"   Recall@{k}: {metrics['recall']:.3f}")
                print(f"   F1-Score@{k}: {metrics['f1_score']:.3f}")
                print(f"   RMSE: {metrics['rmse']:.3f}")
                print(f"   MAE: {metrics['mae']:.3f}")
                print(f"   Success Rate: {metrics['successful_predictions']}/{metrics['total_users_tested']}")
                
            except Exception as e:
                print(f"❌ Error evaluating {algorithm}: {e}")
                # Add placeholder result
                results.append({
                    'algorithm': algorithm,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'successful_predictions': 0,
                    'total_users_tested': 0
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add notes column
        results_df['notes'] = [
            'Good with rich metadata' if alg == 'content_based' 
            else 'Works well with dense ratings' if alg == 'collaborative'
            else 'Best balance between approaches' 
            for alg in results_df['algorithm']
        ]
        
        return results_df
    
    def benchmark_single_movie(self, movie_title, k=5):
        """
        Benchmark all algorithms for a single movie
        
        Args:
            movie_title: Movie to use as query
            k: Number of recommendations to generate
        """
        print(f"\n🎯 BENCHMARKING ALGORITHMS FOR: '{movie_title}'")
        print("-" * 50)
        
        # Get movie info
        movie_info = self.merged_df[self.merged_df['Series_Title'] == movie_title]
        if movie_info.empty:
            print(f"❌ Movie '{movie_title}' not found in dataset")
            return
        
        movie_row = movie_info.iloc[0]
        primary_genre = None
        if pd.notna(movie_row[self.genre_col]):
            genres = movie_row[self.genre_col].split(',')
            primary_genre = genres[0].strip() if genres else None
        
        print(f"📽️ Movie: {movie_title}")
        print(f"🎭 Genre: {movie_row[self.genre_col]}")
        print(f"⭐ Rating: {movie_row[self.rating_col]}/10")
        
        algorithms = {
            'Content-Based': lambda: self.predict_with_algorithm('content_based', movie_title, None, k),
            'Collaborative': lambda: self.predict_with_algorithm('collaborative', movie_title, None, k),
            'Hybrid': lambda: self.predict_with_algorithm('hybrid', movie_title, primary_genre, k)
        }
        
        # Test each algorithm
        for alg_name, alg_func in algorithms.items():
            print(f"\n🔬 {alg_name}:")
            try:
                start_time = datetime.now()
                recommendations = alg_func()
                end_time = datetime.now()
                
                execution_time = (end_time - start_time).total_seconds()
                
                if recommendations:
                    print(f"   ✅ Generated {len(recommendations)} recommendations in {execution_time:.2f}s")
                    for i, title in enumerate(recommendations[:k], 1):
                        movie_info = self.merged_df[self.merged_df['Series_Title'] == title]
                        if not movie_info.empty:
                            rating = movie_info.iloc[0][self.rating_col]
                            print(f"   {i}. {title} (⭐{rating}/10)")
                else:
                    print(f"   ❌ No recommendations generated")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def predict_with_algorithm(self, algorithm_name, target_movie, genre=None, top_n=10):
        """Helper method to get predictions from algorithm"""
        try:
            if algorithm_name == 'content_based' and self.content_based_func:
                results = self.content_based_func(self.merged_df, target_movie, genre, top_n)
            elif algorithm_name == 'collaborative' and self.collaborative_func:
                results = self.collaborative_func(self.merged_df, target_movie, top_n)
            elif algorithm_name == 'hybrid' and self.hybrid_func:
                results = self.hybrid_func(self.merged_df, target_movie, genre, top_n)
            else:
                return []
            
            if results is not None and not results.empty:
                return results['Series_Title'].tolist()
            return []
            
        except Exception:
            return []

def load_datasets():
    """Load required datasets"""
    datasets = {}
    
    # Load movies.csv
    for path in ["movies.csv", "./movies.csv", "data/movies.csv"]:
        if os.path.exists(path):
            datasets['movies'] = pd.read_csv(path)
            print(f"✅ Loaded movies.csv from {path}")
            break
    
    # Load imdb_top_1000.csv
    for path in ["imdb_top_1000.csv", "./imdb_top_1000.csv", "data/imdb_top_1000.csv"]:
        if os.path.exists(path):
            datasets['imdb'] = pd.read_csv(path)
            print(f"✅ Loaded imdb_top_1000.csv from {path}")
            break
    
    # Load user ratings (optional)
    for path in ["user_movie_rating.csv", "./user_movie_rating.csv", "data/user_movie_rating.csv"]:
        if os.path.exists(path):
            datasets['user_ratings'] = pd.read_csv(path)
            print(f"✅ Loaded user_movie_rating.csv from {path}")
            break
    
    return datasets

def create_results_table(results_df):
    """Create formatted results table"""
    if results_df is None or results_df.empty:
        print("❌ No results to display")
        return
    
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    # Format the table
    print(f"{'Method Used':<15} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'RMSE':<8} {'Notes':<25}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        method = row['algorithm'].replace('_', '-').title()
        precision = f"{row['precision']:.3f}"
        recall = f"{row['recall']:.3f}"
        f1_score = f"{row['f1_score']:.3f}"
        rmse = f"{row['rmse']:.3f}" if row['rmse'] != float('inf') else "N/A"
        notes = row['notes']
        
        print(f"{method:<15} {precision:<10} {recall:<8} {f1_score:<9} {rmse:<8} {notes:<25}")
    
    # Find best performing algorithm
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    print(f"\n🏆 BEST OVERALL: {best_f1['algorithm'].replace('_', '-').title()} (F1-Score: {best_f1['f1_score']:.3f})")

def main():
    """Main evaluation function"""
    print("🎬 Movie Recommendation System Evaluator")
    print("Loading datasets...")
    
    # Load datasets
    datasets = load_datasets()
    
    if 'movies' not in datasets or 'imdb' not in datasets:
        print("❌ Required datasets (movies.csv, imdb_top_1000.csv) not found!")
        print("Please ensure these files are in the current directory or data/ folder")
        return
    
    # Merge datasets
    movies_df = datasets['movies']
    imdb_df = datasets['imdb']
    
    # Add Movie_ID if missing
    if 'Movie_ID' not in movies_df.columns:
        print("⚠️ Adding sequential Movie_IDs to movies.csv")
        movies_df['Movie_ID'] = range(len(movies_df))
    
    # Merge on Series_Title
    merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
    merged_df = merged_df.drop_duplicates(subset="Series_Title")
    
    # Preserve Movie_ID in merged dataset
    if 'Movie_ID' not in merged_df.columns and 'Movie_ID' in movies_df.columns:
        merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on="Series_Title", how="inner")
    
    print(f"📊 Merged dataset: {len(merged_df)} movies")
    
    # Get user ratings if available
    user_ratings_df = datasets.get('user_ratings', None)
    if user_ratings_df is not None:
        print(f"👥 User ratings: {len(user_ratings_df)} ratings from {user_ratings_df['User_ID'].nunique()} users")
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(merged_df, user_ratings_df)
    
    # Interactive menu
    while True:
        print("\n" + "=" * 50)
        print("🔧 EVALUATION OPTIONS:")
        print("1. Full Evaluation (All Algorithms)")
        print("2. Single Movie Benchmark")
        print("3. Quick Algorithm Test")
        print("4. Exit")
        print("=" * 50)
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                # Full evaluation
                print("\n🔄 Running comprehensive evaluation...")
                results_df = evaluator.comprehensive_evaluation(k=5)
                if results_df is not None:
                    create_results_table(results_df)
                    
                    # Save results to CSV
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_filename = f"evaluation_results_{timestamp}.csv"
                    results_df.to_csv(results_filename, index=False)
                    print(f"\n💾 Results saved to {results_filename}")
            
            elif choice == '2':
                # Single movie benchmark
                print("\nAvailable movies (sample):")
                sample_movies = merged_df['Series_Title'].head(10).tolist()
                for i, title in enumerate(sample_movies, 1):
                    print(f"{i}. {title}")
                
                movie_input = input("\nEnter movie title (or number from list): ").strip()
                
                # Handle numeric input
                if movie_input.isdigit():
                    idx = int(movie_input) - 1
                    if 0 <= idx < len(sample_movies):
                        movie_title = sample_movies[idx]
                    else:
                        print("❌ Invalid number")
                        continue
                else:
                    movie_title = movie_input
                
                evaluator.benchmark_single_movie(movie_title, k=5)
            
            elif choice == '3':
                # Quick test
                print("\n🚀 Quick Algorithm Test")
                sample_movie = merged_df.sample(1)['Series_Title'].iloc[0]
                print(f"Testing with random movie: {sample_movie}")
                evaluator.benchmark_single_movie(sample_movie, k=5)
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Evaluation stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
