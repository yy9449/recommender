import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from content_based import content_based_filtering_enhanced, create_content_features, find_rating_column, find_genre_column
from collaborative import collaborative_knn, load_user_ratings
import warnings
warnings.filterwarnings('ignore')


class LinearHybridRecommender:
    def __init__(self, merged_df):
        self.merged_df = merged_df
        self.rating_col = find_rating_column(merged_df)
        self.genre_col = find_genre_column(merged_df)
        self.user_ratings_df = load_user_ratings()
        # Weights
        self.alpha = 0.4  # Content
        self.beta = 0.3   # Collaborative
        self.gamma = 0.2  # Popularity
        self.delta = 0.1  # Recency

    def _content_scores(self, target_movie, genre, top_n):
        scores = {}
        titles = self.merged_df['Series_Title'].tolist()

        # Case 1: Movie-based similarity using TF-IDF content features
        if target_movie:
            # Locate target index (case-insensitive exact match)
            target_idx = None
            target_lower = target_movie.strip().lower()
            for i, t in enumerate(titles):
                if isinstance(t, str) and t.lower() == target_lower:
                    target_idx = i
                    break
            if target_idx is None:
                return scores

            # Compute cosine similarity over content features
            content_features = create_content_features(self.merged_df)
            sim = cosine_similarity(content_features[target_idx], content_features).flatten()
            # Build score dict, excluding the target itself
            for i, title in enumerate(titles):
                if i == target_idx:
                    continue
                # Cosine similarity in [0,1] for TF-IDF
                scores[title] = float(np.clip(sim[i], 0.0, 1.0))

            # Keep top candidates
            if len(scores) > top_n * 3:
                scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[: top_n * 3])
            return scores

        # Case 2: Genre query similarity (genre-only TF-IDF)
        if genre:
            genre_col = self.genre_col
            genre_corpus = self.merged_df[genre_col].fillna('').astype(str).tolist()
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(genre_corpus)
            query_vec = tfidf.transform([genre])
            sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
            for i, title in enumerate(titles):
                scores[title] = float(np.clip(sim[i], 0.0, 1.0))
            if len(scores) > top_n * 3:
                scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[: top_n * 3])
            return scores

        return scores

    def _collab_scores(self, target_movie, top_n):
        scores = {}
        if not target_movie:
            return scores
        if self.user_ratings_df is None or self.user_ratings_df.empty:
            return scores
        if 'Movie_ID' not in self.merged_df.columns or 'Series_Title' not in self.merged_df.columns:
            return scores

        # Map title -> Movie_ID
        title_to_id = dict(self.merged_df[['Series_Title', 'Movie_ID']].values)
        target_movie_id = None
        if target_movie in title_to_id:
            target_movie_id = int(title_to_id[target_movie])
        else:
            # case-insensitive
            match = self.merged_df[self.merged_df['Series_Title'].str.lower() == target_movie.lower()]
            if not match.empty:
                target_movie_id = int(match.iloc[0]['Movie_ID'])
        if target_movie_id is None:
            return scores

        # Build user-item matrix and KNN model
        ratings = self.user_ratings_df
        ratings = ratings[ratings['Movie_ID'].isin(self.merged_df['Movie_ID'])].copy()
        if ratings.empty:
            return scores
        user_item = ratings.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
        item_vectors = user_item.fillna(0.0).T
        if target_movie_id not in item_vectors.index:
            return scores

        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(item_vectors)

        idx = item_vectors.index.get_loc(target_movie_id)
        distances, indices = knn.kneighbors(item_vectors.iloc[[idx]], n_neighbors=min(1 + (top_n * 3), len(item_vectors)))
        # Convert to similarity and map to titles
        id_to_title = dict(self.merged_df[['Movie_ID', 'Series_Title']].values)
        for d, i in zip(distances[0], indices[0]):
            nb_movie = int(item_vectors.index[i])
            if nb_movie == target_movie_id:
                continue
            sim = 1.0 - float(d)
            title = id_to_title.get(nb_movie)
            if title is not None:
                scores[title] = float(np.clip(sim, 0.0, 1.0))

        # Keep top candidates
        if len(scores) > top_n * 3:
            scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[: top_n * 3])
        return scores

    def _popularity_scores(self):
        pop = {}
        votes_col = 'No_of_Votes' if 'No_of_Votes' in self.merged_df.columns else 'Votes'
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            rating = movie.get(self.rating_col, 7.0)
            votes = movie.get(votes_col, 1000)
            try:
                votes_val = float(str(votes).replace(',', ''))
            except Exception:
                votes_val = 1000.0
            if pd.isna(rating):
                rating = 7.0
            popularity = (float(rating) * np.log10(votes_val + 1.0)) / 10.0
            pop[title] = float(np.clip(popularity, 0.0, 1.0))
        # Optional: light boost based on user interactions if available
        if self.user_ratings_df is not None and 'Movie_ID' in self.merged_df.columns:
            interaction_counts = self.user_ratings_df['Movie_ID'].value_counts()
            for mid, cnt in interaction_counts.items():
                match = self.merged_df[self.merged_df['Movie_ID'] == mid]
                if not match.empty:
                    t = match.iloc[0]['Series_Title']
                    boost = min(cnt / 100.0, 1.0)
                    pop[t] = 0.6 * pop.get(t, 0.5) + 0.4 * boost
        return pop

    def _recency_scores(self):
        rec = {}
        year_col = 'Released_Year' if 'Released_Year' in self.merged_df.columns else 'Year'
        current_year = pd.Timestamp.now().year
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            year = movie.get(year_col, 2000)
            try:
                year_val = int(str(year).split()[0]) if not pd.isna(year) else 2000
            except Exception:
                year_val = 2000
            diff = max(0, current_year - year_val)
            recency = np.exp(-diff / 20.0)
            rec[title] = float(np.clip(recency, 0.0, 1.0))
        return rec

    def recommend(self, target_movie=None, genre=None, top_n=8):
        # Gather component scores
        content_scores = self._content_scores(target_movie, genre, top_n)
        collab_scores = self._collab_scores(target_movie, top_n)
        popularity_scores = self._popularity_scores()
        recency_scores = self._recency_scores()

        # Candidate set
        candidates = set(content_scores.keys()) | set(collab_scores.keys())
        if len(candidates) < top_n * 2:
            # Add top popular titles
            for t, _ in sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]:
                candidates.add(t)

        final_scores = {}
        for title in candidates:
            c = content_scores.get(title, 0.0)
            cf = collab_scores.get(title, 0.0)
            pop = popularity_scores.get(title, 0.5)
            rec = recency_scores.get(title, 0.5)
            score = self.alpha * c + self.beta * cf + self.gamma * pop + self.delta * rec
            final_scores[title] = float(score)

        top_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_titles = [t for t, _ in top_items]
        result_df = self.merged_df[self.merged_df['Series_Title'].isin(top_titles)]
        if result_df.empty:
            return None
        # Preserve order
        order = {t: i for i, t in enumerate(top_titles)}
        result_df = result_df.copy()
        result_df['rank_order'] = result_df['Series_Title'].map(order)
        result_df = result_df.sort_values('rank_order').drop(columns=['rank_order'])
        return result_df[['Series_Title', self.genre_col, self.rating_col]]


@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    recommender = LinearHybridRecommender(merged_df)
    return recommender.recommend(target_movie, genre, top_n)