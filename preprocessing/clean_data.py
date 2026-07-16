"""
Data cleaning module for LightGCN Recommender System.
Handles data quality issues: duplicates, missing values, invalid IDs/ratings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

from utils.config import get_config

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and validates user-item interaction data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaner with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads from global config.
        """
        self.config = config or get_config().preprocessing
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        self.handle_missing = self.config.get('handle_missing', 'drop')
        self.rating_scale = get_config().data.get('rating_scale', [1, 5])
        
        logger.info(f"DataCleaner initialized: remove_duplicates={self.remove_duplicates}, "
                   f"handle_missing={self.handle_missing}")
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting cleaning pipeline on {len(df)} rows")
        
        initial_rows = len(df)
        
        # 1. Remove duplicates
        if self.remove_duplicates:
            df = self.remove_duplicate_interactions(df)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df)
        
        # 3. Validate and clean IDs
        df = self.validate_ids(df)
        
        # 4. Validate and clean ratings
        df = self.validate_ratings(df)
        
        # 5. Remove corrupted rows
        df = self.remove_corrupted_rows(df)
        
        final_rows = len(df)
        logger.info(f"Cleaning complete: {initial_rows} -> {final_rows} rows "
                   f"({initial_rows - final_rows} removed)")
        
        return df
    
    def remove_duplicate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate user-item interactions.
        
        Keeps the latest interaction (by timestamp) if available,
        otherwise keeps the first occurrence.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicate interactions")
        
        initial_count = len(df)
        
        # Check for duplicates
        dup_mask = df.duplicated(subset=['user_id', 'item_id'], keep=False)
        n_duplicates = dup_mask.sum()
        
        if n_duplicates == 0:
            logger.info("No duplicates found")
            return df
        
        logger.info(f"Found {n_duplicates} duplicate interactions")
        
        if 'timestamp' in df.columns:
            # Keep latest by timestamp
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
            logger.info("Kept latest interaction per user-item pair (by timestamp)")
        else:
            # Keep first occurrence
            df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
            logger.info("Kept first occurrence per user-item pair")
        
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        initial_count = len(df)
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
        
        # Handle missing values based on strategy
        if self.handle_missing == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
            logger.info(f"Dropped rows with missing values")
            
        elif self.handle_missing == 'fill_mean':
            # Fill numeric columns with mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.info(f"Filled missing values in '{col}' with mean: {mean_val:.2f}")
                    
        elif self.handle_missing == 'fill_median':
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled missing values in '{col}' with median: {median_val:.2f}")
                    
        else:
            logger.warning(f"Unknown missing value strategy: {self.handle_missing}, using 'drop'")
            df = df.dropna()
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows due to missing values")
        
        return df
    
    def validate_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate user and item IDs are positive integers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid IDs
        """
        logger.info("Validating user and item IDs")
        
        initial_count = len(df)
        
        # Check for non-positive IDs
        invalid_user_mask = (df['user_id'] <= 0) | (df['user_id'].isna())
        invalid_item_mask = (df['item_id'] <= 0) | (df['item_id'].isna())
        
        n_invalid_users = invalid_user_mask.sum()
        n_invalid_items = invalid_item_mask.sum()
        
        if n_invalid_users > 0:
            logger.warning(f"Found {n_invalid_users} invalid user IDs (<= 0 or NaN)")
        if n_invalid_items > 0:
            logger.warning(f"Found {n_invalid_items} invalid item IDs (<= 0 or NaN)")
        
        # Remove invalid IDs
        valid_mask = ~(invalid_user_mask | invalid_item_mask)
        df = df[valid_mask].copy()
        
        # Ensure integer type
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid IDs")
        
        return df
    
    def validate_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate ratings are within expected range.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid ratings
        """
        logger.info("Validating ratings")
        
        if 'rating' not in df.columns:
            logger.info("No rating column, skipping rating validation")
            return df
        
        initial_count = len(df)
        min_rating, max_rating = self.rating_scale
        
        # Check for out-of-range ratings
        invalid_rating_mask = (
            (df['rating'] < min_rating) | 
            (df['rating'] > max_rating) | 
            (df['rating'].isna())
        )
        
        n_invalid = invalid_rating_mask.sum()
        
        if n_invalid > 0:
            logger.warning(f"Found {n_invalid} invalid ratings (outside [{min_rating}, {max_rating}] or NaN)")
            
            # Show distribution of invalid ratings
            invalid_ratings = df.loc[invalid_rating_mask, 'rating']
            logger.debug(f"Invalid rating values: {invalid_ratings.value_counts().head()}")
        
        # Remove invalid ratings
        df = df[~invalid_rating_mask].copy()
        
        # Ensure float type
        df['rating'] = df['rating'].astype(float)
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid ratings")
        
        return df
    
    def remove_corrupted_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with any corrupted/invalid data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with only valid rows
        """
        logger.info("Removing corrupted rows")
        
        initial_count = len(df)
        
        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        
        for col in numeric_cols:
            inf_mask[col] = np.isinf(df[col])
        
        any_inf = inf_mask.any(axis=1)
        n_inf = any_inf.sum()
        
        if n_inf > 0:
            logger.warning(f"Found {n_inf} rows with infinite values")
            df = df[~any_inf].copy()
        
        # Check for extreme outliers in ratings (if applicable)
        if 'rating' in df.columns:
            # Using IQR method
            Q1 = df['rating'].quantile(0.25)
            Q3 = df['rating'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (df['rating'] < lower_bound) | (df['rating'] > upper_bound)
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                logger.warning(f"Found {n_outliers} rating outliers (IQR method)")
                # Note: We don't remove outliers by default as they might be valid
                # Uncomment next line to remove:
                # df = df[~outlier_mask].copy()
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} corrupted rows")
        
        return df
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save cleaned dataset to file.
        
        Args:
            df: Cleaned DataFrame
            output_path: Output file path
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.parquet':
            df.to_parquet(path, index=False)
        elif path.suffix == '.csv':
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path.with_suffix('.parquet'), index=False)
        
        logger.info(f"Saved cleaned data to {path}")


def clean_data(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Convenience function to clean data.
    
    Args:
        df: Raw DataFrame
        config: Optional configuration
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(config)
    return cleaner.clean(df)