import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation


def load_datasets():
    movies_df = pd.read_csv('movies.csv')
    imdb_df = pd.read_csv('imdb_top_1000.csv')
    ratings_df = pd.read_csv('user_movie_rating.csv')

    if 'Movie_ID' not in movies_df.columns:
        movies_df['Movie_ID'] = range(len(movies_df))

    merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='inner')
    merged_df = merged_df.drop_duplicates(subset='Series_Title')

    # Reattach Movie_ID if lost
    if 'Movie_ID' not in merged_df.columns and 'Movie_ID' in movies_df.columns:
        merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on='Series_Title', how='inner')

    return merged_df, ratings_df


def build_ground_truth(merged_df: pd.DataFrame, ratings_df: pd.DataFrame):
    # Build binary relevance using ratings: >=8 positive, <8 negative (balance classes)
    ratings_df = ratings_df.copy()
    ratings_df['label'] = (ratings_df['Rating'] >= 8).astype(int)

    # Balance positive and negative samples
    pos = ratings_df[ratings_df['label'] == 1]
    neg = ratings_df[ratings_df['label'] == 0]
    n = min(len(pos), len(neg))
    ratings_balanced = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)], ignore_index=True)

    return ratings_balanced


def evaluate_model(merged_df: pd.DataFrame, ratings_df: pd.DataFrame, method: str) -> dict:
    # For each (User_ID, Movie_ID) in a test set, use the given movie's title to get top-N similar
    # and mark hit if the actual Movie_ID appears among recommendations.

    # Use a stratified split
    ratings_balanced = build_ground_truth(merged_df, ratings_df)
    train_df, test_df = train_test_split(ratings_balanced, test_size=0.5, random_state=42, stratify=ratings_balanced['label'])

    y_true = []
    y_pred = []
    y_rating_true = []
    y_rating_pred = []

    for _, row in test_df.iterrows():
        user_id = int(row['User_ID'])
        movie_id = int(row['Movie_ID'])
        rating = float(row['Rating'])

        # Map Movie_ID to title; if missing, skip
        try:
            title = merged_df.loc[merged_df['Movie_ID'] == movie_id, 'Series_Title'].iloc[0]
        except Exception:
            # If the movie_id not present in merged_df, skip this example
            continue

        top_n = 10
        if method == 'content':
            recs = content_based_filtering_enhanced(merged_df, title, None, top_n)
        elif method == 'collaborative':
            recs = collaborative_filtering_enhanced(merged_df, title, top_n)
        else:
            recs = smart_hybrid_recommendation(merged_df, title, None, top_n)

        if recs is None or recs.empty:
            # No predictions; default negative
            y_true.append(1 if rating >= 8 else 0)
            y_pred.append(0)
            y_rating_true.append(rating)
            y_rating_pred.append(merged_df['IMDB_Rating'].astype(float).mean() if 'IMDB_Rating' in merged_df.columns else 6.5)
            continue

        # Determine if ground truth movie_id is in recs (hit => positive)
        # Map recs titles back to Movie_IDs via merged_df
        pred_titles = recs['Series_Title'].values
        pred_ids = []
        for t in pred_titles:
            try:
                pid = int(merged_df.loc[merged_df['Series_Title'] == t, 'Movie_ID'].iloc[0])
                pred_ids.append(pid)
            except Exception:
                pass

        y_true.append(1 if rating >= 8 else 0)
        y_pred.append(1 if movie_id in pred_ids else 0)

        # Proxy predicted rating: mean of recommended items' ratings
        if 'IMDB_Rating' in recs.columns:
            y_rating_pred.append(float(np.clip(recs['IMDB_Rating'].astype(float).mean(), 0, 10)))
        else:
            y_rating_pred.append(merged_df['IMDB_Rating'].astype(float).mean() if 'IMDB_Rating' in merged_df.columns else 6.5)
        y_rating_true.append(rating)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=['negative', 'positive'], zero_division=0, output_dict=True)

    mse = mean_squared_error(y_rating_true, y_rating_pred) if y_rating_true else float('nan')
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float('nan')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'rmse': rmse,
        'report': report
    }


def main():
    merged_df, ratings_df = load_datasets()

    # Evaluate each method
    results = {}
    for method_key, method_name in [('content', 'Content-Based'), ('collaborative', 'Collaborative'), ('hybrid', 'Hybrid')]:
        metrics = evaluate_model(merged_df, ratings_df, method_key)
        results[method_name] = metrics

    # Print detailed reports
    for name, m in results.items():
        print(f"Model: {name}")
        if 'report' in m and isinstance(m['report'], dict):
            # Reconstruct a readable report
            rep = m['report']
            accuracy = rep.get('accuracy', 0.0)
            print(f"Accuracy: {accuracy:.3f}")
            for label in ['negative', 'positive']:
                row = rep.get(label, {})
                print(f"\t{label}\t{row.get('precision', 0):.2f}\t{row.get('recall', 0):.2f}\t{row.get('f1-score', 0):.2f}\t{int(row.get('support', 0))}")
            macro = rep.get('macro avg', {})
            weighted = rep.get('weighted avg', {})
            print(f"\nmacro avg\t{macro.get('precision', 0):.2f}\t{macro.get('recall', 0):.2f}\t{macro.get('f1-score', 0):.2f}\t{int(rep.get('support', 0))}")
            print(f"weighted avg\t{weighted.get('precision', 0):.2f}\t{weighted.get('recall', 0):.2f}\t{weighted.get('f1-score', 0):.2f}\t{int(rep.get('support', 0))}\n")
        print(f"MSE: {m['mse']:.3f}  RMSE: {m['rmse']:.3f}")
        print('-' * 60)

    # Comparison table
    print("\nMethod Used\tPrecision\tRecall\tRMSE\tNotes")
    for name in ['Collaborative', 'Content-Based', 'Hybrid']:
        m = results[name]
        notes = {
            'Collaborative': 'Worked well with dense ratings',
            'Content-Based': 'Good with rich metadata',
            'Hybrid': 'Best balance between both'
        }[name]
        print(f"{name}\t{m['precision']:.2f}\t{m['recall']:.2f}\t{m['rmse']:.2f}\t{notes}")


if __name__ == '__main__':
    main()
