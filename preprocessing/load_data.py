"""
Data loading module for LightGCN Recommender System.
Responsible for loading and validating raw datasets.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from utils.config import get_config
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial validation of raw datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads from global config.
        """
        self.config = config or get_config().data
        self.raw_data_path = self.config.get('raw_data_path', 'data/raw/ml-100k/u.data')
        self.expected_columns = self.config.get('expected_columns', 
            ['user_id', 'item_id', 'rating', 'timestamp'])
        self.rating_scale = self.config.get('rating_scale', [1, 5])
        self.separator = self.config.get('separator', '\t')
        self.header = self.config.get('header', None)
        
        logger.info(f"DataLoader initialized with path: {self.raw_data_path}")
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            file_path: Path to data file. If None, uses config path.
            
        Returns:
            DataFrame with raw data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = file_path or self.raw_data_path
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info(f"Loading data from {path}")
        
        try:
            # Try different formats based on extension
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix == '.csv':
                df = pd.read_csv(path, sep=self.separator, header=self.header)
            elif path.suffix in ['.data', '.txt', '.dat']:
                df = pd.read_csv(
                    path, 
                    sep=self.separator, 
                    header=self.header,
                    names=self.expected_columns
                )
            else:
                # Try CSV as default
                df = pd.read_csv(path, sep=self.separator, header=self.header)
                
        except Exception as e:
            logger.error(f"Failed to load data from {path}: {e}")
            raise ValueError(f"Could not parse data file: {e}")
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame schema matches expectations.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If schema validation fails
        """
        logger.info("Validating data schema")
        
        # Check required columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for extra columns (warning only)
        extra_cols = set(df.columns) - set(self.expected_columns)
        if extra_cols:
            logger.warning(f"Extra columns found (will be ignored): {extra_cols}")
        
        # Check data types
        expected_types = {
            'user_id': ['int64', 'int32', 'object'],
            'item_id': ['int64', 'int32', 'object'],
            'rating': ['float64', 'float32', 'int64', 'int32'],
            'timestamp': ['int64', 'int32', 'float64', 'object']
        }
        
        for col, valid_types in expected_types.items():
            if col in df.columns:
                if df[col].dtype.name not in valid_types:
                    logger.warning(
                        f"Column '{col}' has unexpected dtype: {df[col].dtype}. "
                        f"Expected one of: {valid_types}"
                    )
        
        # Check rating scale
        if 'rating' in df.columns:
            min_rating, max_rating = df['rating'].min(), df['rating'].max()
            expected_min, expected_max = self.rating_scale
            if min_rating < expected_min or max_rating > expected_max:
                logger.warning(
                    f"Rating values outside expected range [{expected_min}, {expected_max}]: "
                    f"found [{min_rating}, {max_rating}]"
                )
        
        logger.info("Schema validation passed")
        return True
    
    def get_dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive dataset summary statistics.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating dataset summary")
        
        summary = {
            'n_interactions': len(df),
            'n_users': df['user_id'].nunique(),
            'n_items': df['item_id'].nunique(),
            'sparsity': 1 - (len(df) / (df['user_id'].nunique() * df['item_id'].nunique())),
            'rating_stats': {
                'min': float(df['rating'].min()),
                'max': float(df['rating'].max()),
                'mean': float(df['rating'].mean()),
                'median': float(df['rating'].median()),
                'std': float(df['rating'].std()),
            },
            'user_interaction_stats': {
                'min': int(df.groupby('user_id').size().min()),
                'max': int(df.groupby('user_id').size().max()),
                'mean': float(df.groupby('user_id').size().mean()),
                'median': float(df.groupby('user_id').size().median()),
            },
            'item_interaction_stats': {
                'min': int(df.groupby('item_id').size().min()),
                'max': int(df.groupby('item_id').size().max()),
                'mean': float(df.groupby('item_id').size().mean()),
                'median': float(df.groupby('item_id').size().median()),
            }
        }
        
        # Add timestamp range if available
        if 'timestamp' in df.columns:
            try:
                summary['timestamp_range'] = {
                    'min': int(df['timestamp'].min()),
                    'max': int(df['timestamp'].max()),
                }
            except:
                pass
        
        logger.info(
            f"Dataset summary: {summary['n_interactions']} interactions, "
            f"{summary['n_users']} users, {summary['n_items']} items, "
            f"sparsity={summary['sparsity']:.4f}"
        )
        
        return summary
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """Print formatted dataset summary to console."""
        summary = self.get_dataset_summary(df)
        
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Interactions: {summary['n_interactions']:,}")
        print(f"Unique Users: {summary['n_users']:,}")
        print(f"Unique Items: {summary['n_items']:,}")
        print(f"Sparsity: {summary['sparsity']:.4%}")
        print(f"\nRating Statistics:")
        for k, v in summary['rating_stats'].items():
            print(f"  {k.capitalize()}: {v:.2f}")
        print(f"\nUser Interactions:")
        for k, v in summary['user_interaction_stats'].items():
            print(f"  {k.capitalize()}: {v:.1f}")
        print(f"\nItem Interactions:")
        for k, v in summary['item_interaction_stats'].items():
            print(f"  {k.capitalize()}: {v:.1f}")
        if 'timestamp_range' in summary:
            print(f"\nTimestamp Range: {summary['timestamp_range']['min']} - {summary['timestamp_range']['max']}")
        print("="*50 + "\n")


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load raw data.
    
    Args:
        file_path: Optional path to data file
        
    Returns:
        Raw DataFrame
    """
    loader = DataLoader()
    return loader.load_data(file_path)


def validate_and_summarize(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to validate and summarize data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Summary dictionary
    """
    loader = DataLoader()
    loader.validate_schema(df)
    return loader.get_dataset_summary(df)