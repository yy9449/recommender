import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from content_based import create_content_features
from collaborative import collaborative_filtering_enhanced, load_user_ratings


class LinearHybridRecommender:
    def __init__(self, merged_df: pd.DataFrame):
        self.merged_df = merged_df
        self.rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
        self.genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'

        # Required weights (tuned)
        self.alpha = 0.45  # Content-based weight
        self.beta = 0.35   # Collaborative weight
        self.gamma = 0.15  # Popularity weight
        self.delta = 0.05  # Recency weight

        # Optional: user ratings availability
        self.user_ratings_df = load_user_ratings()

    def _get_content_scores(self, target_movie: str | None, genre: str | None, top_n: int) -> dict:
        if not target_movie and not genre:
            return {}

        # Build TF-IDF features once
        try:
            tfidf_matrix = create_content_features(self.merged_df)
        except Exception:
            return {}

        scores = {}

        if target_movie:
            # Find target index
            try:
                target_idx = self.merged_df[self.merged_df['Series_Title'].str.lower() == target_movie.lower()].index[0]
            except Exception:
                return {}

            target_vec = tfidf_matrix[self.merged_df.index.get_loc(target_idx)].reshape(1, -1)
            sims = cosine_similarity(target_vec, tfidf_matrix).flatten()
            # Convert to ranking score (0..1)
            ranks = np.argsort(-sims)
            for r in ranks[:top_n * 3]:
                if r == self.merged_df.index.get_loc(target_idx):
                    continue
                title = self.merged_df.iloc[r]['Series_Title']
                scores[title] = float(sims[r])
        elif genre:
            # Genre query by cosine similarity in TF-IDF space using simple approach
            # Build a query vector by averaging vectors of movies with that genre
            mask = self.merged_df[self.genre_col].fillna('').str.contains(genre, case=False)
            if not mask.any():
                return {}
            genre_indices = [self.merged_df.index.get_loc(i) for i in self.merged_df[mask].index]
            genre_matrix = tfidf_matrix[genre_indices]
            centroid = genre_matrix.mean(axis=0)
            sims = cosine_similarity(centroid, tfidf_matrix).flatten()
            ranks = np.argsort(-sims)
            for r in ranks[:top_n * 3]:
                title = self.merged_df.iloc[r]['Series_Title']
                scores[title] = float(sims[r])

        # Normalize to 0..1
        if scores:
            max_score = max(scores.values()) or 1.0
            scores = {k: v / max_score for k, v in scores.items()}
        return scores

    def _get_cf_scores(self, target_movie: str | None, top_n: int) -> dict:
        if not target_movie:
            return {}
        if self.user_ratings_df is None or self.user_ratings_df.empty:
            return {}

        cf_df = collaborative_filtering_enhanced(self.merged_df, target_movie, top_n=top_n * 3)
        scores = {}
        if cf_df is not None and not cf_df.empty and self.rating_col in cf_df.columns:
            max_rating = max(cf_df[self.rating_col].max(), 1.0)
            for _, row in cf_df.iterrows():
                scores[row['Series_Title']] = float(row[self.rating_col]) / max_rating
        return scores

    def _get_popularity_scores(self) -> dict:
        scores = {}
        votes_col = 'No_of_Votes' if 'No_of_Votes' in self.merged_df.columns else ('Votes' if 'Votes' in self.merged_df.columns else None)
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            rating = movie.get(self.rating_col, 7.0)
            votes = movie.get(votes_col, 1000) if votes_col else 1000
            rating = 7.0 if pd.isna(rating) else rating
            votes = 1000 if pd.isna(votes) else votes
            pop = float(rating) * np.log10(float(votes) + 1.0) / 10.0
            scores[title] = min(max(pop, 0.0), 1.0)
        return scores

    def _get_recency_scores(self) -> dict:
        current_year = pd.Timestamp.now().year
        scores = {}
        year_col = 'Released_Year' if 'Released_Year' in self.merged_df.columns else 'Year'
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            year = movie.get(year_col, 2000)
            try:
                year = int(year)
            except Exception:
                year = 2000
            diff = max(current_year - year, 0)
            rec = float(np.exp(-diff / 20.0))
            scores[title] = max(min(rec, 1.0), 0.1)
        return scores

    def recommend(self, target_movie: str | None = None, genre: str | None = None, top_n: int = 8) -> pd.DataFrame | None:
        # Collect component scores
        content_scores = self._get_content_scores(target_movie, genre, top_n)
        cf_scores = self._get_cf_scores(target_movie, top_n)
        popularity_scores = self._get_popularity_scores()
        recency_scores = self._get_recency_scores()

        # Candidate pool
        candidates = set(content_scores) | set(cf_scores)
        if len(candidates) < top_n * 2:
            candidates |= set(sorted(popularity_scores, key=popularity_scores.get, reverse=True)[:top_n * 2])

        final_scores = {}
        for title in candidates:
            c = content_scores.get(title, 0.0)
            f = cf_scores.get(title, 0.0)
            p = popularity_scores.get(title, 0.5)
            r = recency_scores.get(title, 0.5)
            final_scores[title] = self.alpha * c + self.beta * f + self.gamma * p + self.delta * r

        if not final_scores:
            return None

        top_titles = [t for t, _ in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        result = self.merged_df[self.merged_df['Series_Title'].isin(top_titles)].copy()
        if result.empty:
            return None
        rank_map = {t: i for i, t in enumerate(top_titles)}
        result['rank_order'] = result['Series_Title'].map(rank_map)
        result = result.sort_values('rank_order').drop(columns=['rank_order'])
        return result[[c for c in ['Series_Title', self.genre_col, self.rating_col] if c in result.columns]]


@st.cache_data
def smart_hybrid_recommendation(merged_df: pd.DataFrame, target_movie: str | None = None, genre: str | None = None, top_n: int = 8) -> pd.DataFrame | None:
    """Linear weighted blend hybrid: 0.4*Content + 0.3*CF + 0.2*Popularity + 0.1*Recency."""
    return LinearHybridRecommender(merged_df).recommend(target_movie, genre, top_n)
