import os
import io
import sys
import math
import random
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    mean_squared_error,
)

# Import recommenders (they use Streamlit caches internally; fine for CLI use)
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation


warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)


def _data_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_local_data():
    """Load local CSVs located alongside this script."""
    base = _data_dir()
    movies = pd.read_csv(os.path.join(base, "movies.csv"))
    imdb = pd.read_csv(os.path.join(base, "imdb_top_1000.csv"))
    # Ratings are optional but required for evaluation
    ratings_path = os.path.join(base, "user_movie_rating.csv")
    ratings = pd.read_csv(ratings_path) if os.path.exists(ratings_path) else None
    return movies, imdb, ratings


def prepare_merged(movies: pd.DataFrame, imdb: pd.DataFrame) -> pd.DataFrame:
    """Merge movie metadata with IMDB data and ensure Movie_ID exists."""
    if "Movie_ID" not in movies.columns:
        movies = movies.copy()
        movies["Movie_ID"] = range(len(movies))

    merged = pd.merge(movies, imdb, on="Series_Title", how="inner")
    merged = merged.drop_duplicates(subset="Series_Title")
    if "Movie_ID" not in merged.columns and "Movie_ID" in movies.columns:
        merged = pd.merge(movies[["Movie_ID", "Series_Title"]], merged, on="Series_Title", how="inner")
    return merged


def detect_cols(df: pd.DataFrame):
    rating_col = "IMDB_Rating" if "IMDB_Rating" in df.columns else ("Rating" if "Rating" in df.columns else None)
    genre_col = "Genre_y" if "Genre_y" in df.columns else ("Genre" if "Genre" in df.columns else None)
    year_col = "Released_Year" if "Released_Year" in df.columns else ("Year" if "Year" in df.columns else None)
    return rating_col, genre_col, year_col


def rating_threshold(user_ratings: pd.DataFrame) -> float:
    """Pick a sensible positive-rating threshold based on observed scale."""
    if user_ratings is None or user_ratings.empty:
        return 4.0
    max_r = float(user_ratings["Rating"].max())
    if max_r <= 5:
        return 4.0
    if max_r <= 10:
        return 8.0
    # Fallback: 80% of max
    return 0.8 * max_r


def scale_dataset_rating_to_user_scale(ds_rating: float, user_max: float) -> float:
    if ds_rating is None or np.isnan(ds_rating):
        return np.nan
    # Assume dataset rating likely on 1..10; map to user scale if 1..5
    if user_max <= 5:
        return float(ds_rating) / 2.0
    return float(ds_rating)


def build_user_index_maps(merged: pd.DataFrame):
    id_to_title = {}
    title_to_id = {}
    for _, row in merged.iterrows():
        mid = row.get("Movie_ID")
        title = row.get("Series_Title")
        if pd.notna(mid) and pd.notna(title):
            id_to_title[int(mid)] = str(title)
            title_to_id[str(title).lower()] = int(mid)
    return id_to_title, title_to_id


def evaluate_algorithm(
    algo_name: str,
    merged: pd.DataFrame,
    user_ratings: pd.DataFrame,
    top_n: int = 8,
    max_users: int = 50,
):
    """
    Evaluate a recommender by sampling users, seeding by one liked movie,
    generating recommendations, and computing classification metrics.

    Returns dict with accuracy, macro precision/recall/f1, report string, MSE, RMSE.
    """
    rating_col, _, _ = detect_cols(merged)
    if user_ratings is None or user_ratings.empty:
        raise RuntimeError("user_movie_rating.csv not found or empty; evaluation requires user ratings")

    thr = rating_threshold(user_ratings)
    user_max = float(user_ratings["Rating"].max())
    id_to_title, title_to_id = build_user_index_maps(merged)

    # Eligible users: must have at least 1 positive and 1 negative
    eligible_users = []
    for uid, grp in user_ratings.groupby("User_ID"):
        pos = grp[grp["Rating"] >= thr]
        neg = grp[grp["Rating"] < thr]
        if not pos.empty and not neg.empty:
            eligible_users.append(int(uid))

    if not eligible_users:
        raise RuntimeError("No eligible users with both positive and negative ratings found.")

    sampled_users = eligible_users[: max_users]

    y_true = []
    y_pred = []
    true_ratings = []
    pred_ratings = []

    for uid in sampled_users:
        ugrp = user_ratings[user_ratings["User_ID"] == uid]
        u_pos = ugrp[ugrp["Rating"] >= thr]
        u_neg = ugrp[ugrp["Rating"] < thr]

        # Pick a seed positive movie that exists in merged
        seed_mid = None
        if not u_pos.empty:
            # Highest rated first
            u_pos_sorted = u_pos.sort_values("Rating", ascending=False)
            for _, r in u_pos_sorted.iterrows():
                cand_mid = int(r["Movie_ID"]) if "Movie_ID" in r else None
                if cand_mid in id_to_title:
                    seed_mid = cand_mid
                    break
        if seed_mid is None:
            continue

        seed_title = id_to_title[seed_mid]

        # Generate recommendations
        if algo_name == "Content-Based":
            rec_df = content_based_filtering_enhanced(merged, target_movie=seed_title, genre=None, top_n=top_n)
        elif algo_name == "Collaborative":
            rec_df = collaborative_filtering_enhanced(merged, target_movie=seed_title, top_n=top_n)
        else:  # Hybrid
            rec_df = smart_hybrid_recommendation(merged, target_movie=seed_title, genre=None, top_n=top_n)

        if rec_df is None or rec_df.empty:
            continue

        rec_titles = [str(t) for t in rec_df["Series_Title"].tolist() if pd.notna(t)]
        rec_ids = [title_to_id[t.lower()] for t in rec_titles if t.lower() in title_to_id]

        # Positive predictions: recommended items rated by the same user
        rated_rec = ugrp[ugrp["Movie_ID"].isin(rec_ids)]
        for _, rr in rated_rec.iterrows():
            actual = 1 if rr["Rating"] >= thr else 0
            y_true.append(actual)
            y_pred.append(1)  # recommended => predicted positive

            # Predicted rating proxy from dataset ratings (scaled to user scale)
            mid = int(rr["Movie_ID"]) if "Movie_ID" in rr else None
            if mid in id_to_title and rating_col in merged.columns:
                title = id_to_title[mid]
                row = merged[merged["Series_Title"] == title]
                if not row.empty:
                    ds_rating = float(row.iloc[0][rating_col]) if pd.notna(row.iloc[0][rating_col]) else np.nan
                    pr = scale_dataset_rating_to_user_scale(ds_rating, user_max)
                    if not np.isnan(pr):
                        true_ratings.append(float(rr["Rating"]))
                        pred_ratings.append(pr)

        # Negative predictions: sample same number of user's negatives not in recs
        num_pos_samples = len(rated_rec)
        if num_pos_samples > 0 and not u_neg.empty:
            cand_negs = u_neg[~u_neg["Movie_ID"].isin(rec_ids)]
            if not cand_negs.empty:
                sampled = cand_negs.sample(n=min(num_pos_samples, len(cand_negs)), random_state=42)
                for _, rn in sampled.iterrows():
                    y_true.append(0)
                    y_pred.append(0)  # not recommended => predicted negative

    # Metrics
    if not y_true:
        raise RuntimeError("Insufficient overlap between recommendations and user ratings for evaluation.")

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=["negative", "positive"], zero_division=0)

    mse = float("nan")
    rmse = float("nan")
    if pred_ratings and true_ratings:
        mse = mean_squared_error(true_ratings, pred_ratings)
        rmse = math.sqrt(mse)

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "report": report,
        "mse": mse,
        "rmse": rmse,
    }


def main():
    print("Loading data...")
    movies, imdb, ratings = load_local_data()
    merged = prepare_merged(movies, imdb)

    methods = [
        ("Content-Based", lambda: None),
        ("Collaborative", lambda: None),
        ("Hybrid", lambda: None),
    ]

    # 1) Compute all results first (no printing during loop)
    detailed_results = []
    summary_rows = []
    for name, _ in methods:
        try:
            res = evaluate_algorithm(name, merged, ratings, top_n=8, max_users=50)
            detailed_results.append((name, res))
            summary_rows.append({
                "Method": name,
                "Precision": res["precision_macro"],
                "Recall": res["recall_macro"],
                "RMSE": res["rmse"],
                "Notes": "RMSE uses dataset rating as proxy",
            })
        except Exception as e:
            detailed_results.append((name, {"error": str(e)}))

    # 2) Print classification reports and accuracies for each model
    for name, res in detailed_results:
        print(f"\nModel: {name}")
        if "error" in res:
            print(f"Evaluation failed: {res['error']}")
            continue
        print(f"Accuracy: {res['accuracy']:.3f}")
        print(res["report"])  # precision/recall/f1 per class and averages

    # 3) Print comparison table
    if summary_rows:
        print("\nCompare and Discuss Results")
        header = f"{'Method Used':<16} {'Precision':>9} {'Recall':>9} {'RMSE':>9}  Notes"
        print(header)
        print("-" * len(header))
        for r in summary_rows:
            prec = f"{r['Precision']:.2f}" if r['Precision'] is not None and not np.isnan(r['Precision']) else "N/A"
            rec = f"{r['Recall']:.2f}" if r['Recall'] is not None and not np.isnan(r['Recall']) else "N/A"
            rmse_val = r['RMSE']
            rmse_str = f"{rmse_val:.2f}" if rmse_val is not None and not np.isnan(rmse_val) else "N/A"
            print(f"{r['Method']:<16} {prec:>9} {rec:>9} {rmse_str:>9}  {r['Notes']}")


if __name__ == "__main__":
    main()


