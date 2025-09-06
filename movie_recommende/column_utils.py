# column_utils.py

import pandas as pd
from typing import Optional, List

def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find the first existing column from a list of possible names.
    
    Args:
        df: DataFrame to search in
        possible_names: List of possible column names in order of preference
        
    Returns:
        The first matching column name, or None if none found
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def get_genre_column(df: pd.DataFrame) -> Optional[str]:
    """Get the appropriate genre column name from the DataFrame."""
    possible_names = ['Genre', 'Genre_y', 'Genre_x', 'Genres']
    return find_column(df, possible_names)

def get_overview_column(df: pd.DataFrame) -> Optional[str]:
    """Get the appropriate overview column name from the DataFrame."""
    possible_names = ['Overview', 'Overview_y', 'Overview_x', 'Plot', 'Description', 'Summary']
    return find_column(df, possible_names)

def get_rating_column(df: pd.DataFrame) -> Optional[str]:
    """Get the appropriate rating column name from the DataFrame."""
    possible_names = ['IMDB_Rating', 'Rating', 'IMDB_Rating_y', 'IMDB_Rating_x', 'Rating_y', 'Rating_x']
    return find_column(df, possible_names)

def get_year_column(df: pd.DataFrame) -> Optional[str]:
    """Get the appropriate year column name from the DataFrame."""
    possible_names = ['Released_Year', 'Year', 'Release_Year', 'Released_Year_y', 'Released_Year_x', 'Year_y', 'Year_x']
    return find_column(df, possible_names)

def get_director_column(df: pd.DataFrame) -> Optional[str]:
    """Get the appropriate director column name from the DataFrame."""
    possible_names = ['Director', 'Director_y', 'Director_x']
    return find_column(df, possible_names)

def get_votes_column(df: pd.DataFrame) -> Optional[str]:
    """Get the appropriate votes column name from the DataFrame."""
    possible_names = ['No_of_Votes', 'Votes', 'No_of_Votes_y', 'No_of_Votes_x', 'Votes_y', 'Votes_x']
    return find_column(df, possible_names)

def safe_get_column_data(df: pd.DataFrame, column_name: Optional[str], default_value: str = '') -> pd.Series:
    """
    Safely get column data with fallback to default values.
    
    Args:
        df: DataFrame to get data from
        column_name: Name of the column (can be None)
        default_value: Default value to use if column doesn't exist
        
    Returns:
        Series with the column data or default values
    """
    if column_name and column_name in df.columns:
        return df[column_name].fillna(default_value).astype(str)
    else:
        return pd.Series([default_value] * len(df), index=df.index)

def apply_genre_filter(df: pd.DataFrame, genre_filter: str) -> pd.DataFrame:
    """
    Apply genre filter to DataFrame using the correct genre column.
    
    Args:
        df: DataFrame to filter
        genre_filter: Genre string to filter by
        
    Returns:
        Filtered DataFrame
    """
    genre_col = get_genre_column(df)
    if genre_col:
        return df[df[genre_col].str.contains(genre_filter, case=False, na=False)]
    else:
        # If no genre column found, return empty DataFrame
        return pd.DataFrame()

def get_movie_display_info(df: pd.DataFrame, movie_row: pd.Series) -> dict:
    """
    Extract display information for a movie using correct column names.
    
    Args:
        df: Full DataFrame for reference
        movie_row: Single movie row
        
    Returns:
        Dictionary with standardized movie information
    """
    rating_col = get_rating_column(df)
    genre_col = get_genre_column(df)
    year_col = get_year_column(df)
    
    return {
        'title': movie_row.get('Series_Title', 'Unknown'),
        'rating': movie_row.get(rating_col, 'N/A') if rating_col else 'N/A',
        'genre': movie_row.get(genre_col, 'N/A') if genre_col else 'N/A',
        'year': movie_row.get(year_col, 'N/A') if year_col else 'N/A',
        'poster': movie_row.get('Poster_Link', '')
    }
