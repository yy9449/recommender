def smart_hybrid_recommendation(user_id=1, movie_title=None, genre_input=None, df=None, 
                               ratings_df=None, top_n=10):
    """
    Fixed hybrid recommendation system that properly integrates all components.
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
    
    # Get all available movies for recommendation
    all_movie_ids = df['Movie_ID'].dropna().unique().tolist()
    
    # Remove movies already rated by user (optional - you might want to keep them)
    user_rated = []
    if ratings_df is not None and not ratings_df.empty:
        user_rated = ratings_df[ratings_df['User_ID'] == user_id]['Movie_ID'].tolist()
    
    # 1. CONTENT-BASED SCORES
    content_scores = {}
    df_copy = df.copy().fillna('')
    df_copy['soup'] = df_copy['Genre'] + ' ' + df_copy['Overview'] + ' ' + df_copy['Director'] + ' ' + df_copy['Stars']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_copy['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    if movie_title:
        # Movie-based content filtering
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
            
            # Store content scores for ALL movies, not just similar ones
            for i, score in sim_scores:
                if 'Movie_ID' in df_copy.iloc[i] and pd.notna(df_copy.iloc[i]['Movie_ID']):
                    content_scores[df_copy.iloc[i]['Movie_ID']] = score
        else:
            # Movie not found, give all movies a base content score
            for movie_id in all_movie_ids:
                content_scores[movie_id] = 0.1
    
    elif genre_input:
        # Genre-based content filtering
        genre_col = 'Genre_y' if 'Genre_y' in df_copy.columns else 'Genre'
        for idx, row in df_copy.iterrows():
            if 'Movie_ID' in row and pd.notna(row['Movie_ID']):
                score = 0.8 if genre_input.lower() in str(row[genre_col]).lower() else 0.1
                content_scores[row['Movie_ID']] = score
    else:
        # No specific content preferences, give all movies equal content score
        for movie_id in all_movie_ids:
            content_scores[movie_id] = 0.5
    
    # 2. COLLABORATIVE FILTERING SCORES - ENHANCED
    collab_scores = {}
    
    # Initialize all movies with base collaborative score
    for movie_id in all_movie_ids:
        collab_scores[movie_id] = 5.0  # Neutral score
    
    if SURPRISE_AVAILABLE and svd_model and ratings_df is not None:
        # Get collaborative scores for ALL movies
        for movie_id in all_movie_ids:
            try:
                pred = svd_model.predict(user_id, movie_id)
                collab_scores[movie_id] = pred.est
            except:
                collab_scores[movie_id] = 5.0  # Default score
        
        # IMPORTANT: Add collaborative diversity boost
        # This ensures collaborative filtering can recommend different types of movies
        if movie_title:
            # Find the input movie's genre
            input_movie_info = df[df['Series_Title'] == movie_title]
            if not input_movie_info.empty:
                input_genre = input_movie_info.iloc[0].get('Genre_y' if 'Genre_y' in df.columns else 'Genre', '')
                
                # Boost movies from DIFFERENT genres that have high collaborative scores
                for movie_id in all_movie_ids:
                    if collab_scores[movie_id] >= 7.0:  # High collaborative score
                        movie_info = df[df['Movie_ID'] == movie_id]
                        if not movie_info.empty:
                            movie_genre = movie_info.iloc[0].get('Genre_y' if 'Genre_y' in df.columns else 'Genre', '')
                            
                            # Check if genres are different
                            if input_genre and movie_genre:
                                input_genres = set(g.strip().lower() for g in str(input_genre).split(','))
                                movie_genres = set(g.strip().lower() for g in str(movie_genre).split(','))
                                
                                # If no overlap in genres, boost the collaborative score
                                if not input_genres.intersection(movie_genres):
                                    collab_scores[movie_id] = min(collab_scores[movie_id] + 1.0, 10.0)
        
        # BONUS: User similarity boost (same as before)
        if user_rated:
            similar_users = []
            for rated_movie in user_rated[-10]:
                user_rating_rows = ratings_df[
                    (ratings_df['User_ID'] == user_id) & 
                    (ratings_df['Movie_ID'] == rated_movie)
                ]
                if not user_rating_rows.empty:
                    user_rating = user_rating_rows['Rating'].iloc[0]
                    
                    if user_rating >= 7:
                        similar_users.extend(
                            ratings_df[
                                (ratings_df['Movie_ID'] == rated_movie) & 
                                (ratings_df['Rating'] >= 7) & 
                                (ratings_df['User_ID'] != user_id)
                            ]['User_ID'].tolist()
                        )
            
            if similar_users:
                similar_users = list(set(similar_users))[:20]
                for movie_id in all_movie_ids:
                    if movie_id not in user_rated:
                        similar_ratings = ratings_df[
                            (ratings_df['User_ID'].isin(similar_users)) & 
                            (ratings_df['Movie_ID'] == movie_id) & 
                            (ratings_df['Rating'] >= 7)
                        ]
                        if len(similar_ratings) > 0:
                            boost = min(len(similar_ratings) * 0.1, 1.0)
                            collab_scores[movie_id] = min(collab_scores.get(movie_id, 5.0) + boost, 10.0)
    
    # 3. POPULARITY SCORES (same as before)
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
    
    # 4. RECENCY SCORES (same as before)
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
    
    # 5. COMBINE ALL SCORES - Now all components contribute equally
    all_movie_ids = set(all_movie_ids)  # Convert to set to remove duplicates
    
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
    
    # Remove movies user has already rated (optional)
    if user_rated:
        result_df = result_df[~result_df['Movie_ID'].isin(user_rated)]
    
    return result_df.head(top_n)
