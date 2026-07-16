"""
Data splitting module for LightGCN Recommender System.
Performs train/validation/test splits with temporal or random strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

from utils.config import get_config

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Splits interaction data into train/validation/test sets.
    
    Supports multiple splitting strategies:
    - Random split
    - Temporal split (by timestamp)
    - Leave-one-out (last interaction per user)
    - User-based split (disjoint user sets)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataSplitter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().preprocessing
        
        self.validation_split = self.config.get('validation_split', 0.1)
        self.test_split = self.config.get('test_split', 0.1)
        self.random_seed = self.config.get('random_seed', 42)
        self.strategy = self.config.get('split_strategy', 'random')  # random, temporal, leave_one_out
        self.timestamp_col = self.config.get('timestamp_col', 'timestamp')
        self.user_col = self.config.get('user_col', 'user_id')
        self.item_col = self.config.get('item_col', 'item_id')
        
        # Validate splits
        total_split = self.validation_split + self.test_split
        if total_split >= 1.0:
            raise ValueError(f"Validation + test split ({total_split}) must be < 1.0")
        
        logger.info(f"DataSplitter initialized: strategy={self.strategy}, "
                   f"val={self.validation_split}, test={self.test_split}")
    
    def split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data according to configured strategy.
        
        Args:
            df: Input DataFrame with interactions
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data with strategy: {self.strategy}")
        
        if self.strategy == 'random':
            return self._random_split(df)
        elif self.strategy == 'temporal':
            return self._temporal_split(df)
        elif self.strategy == 'leave_one_out':
            return self._leave_one_out_split(df)
        elif self.strategy == 'user_based':
            return self._user_based_split(df)
        else:
            raise ValueError(f"Unknown split strategy: {self.strategy}")
    
    def _random_split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random shuffle split.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        logger.info("Performing random split")
        
        # Shuffle
        df = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        
        n = len(df)
        n_test = int(n * self.test_split)
        n_val = int(n * self.validation_split)
        n_train = n - n_val - n_test
        
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        
        logger.info(f"Random split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _temporal_split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split by timestamp (oldest -> train, middle -> val, newest -> test).
        
        Args:
            df: Input DataFrame with timestamp column
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        logger.info("Performing temporal split")
        
        if self.timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found for temporal split")
        
        # Sort by timestamp
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        
        n = len(df)
        n_test = int(n * self.test_split)
        n_val = int(n * self.validation_split)
        n_train = n - n_val - n_test
        
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        
        logger.info(f"Temporal split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        logger.info(f"Train time range: {train_df[self.timestamp_col].min()} - {train_df[self.timestamp_col].max()}")
        logger.info(f"Val time range: {val_df[self.timestamp_col].min()} - {val_df[self.timestamp_col].max()}")
        logger.info(f"Test time range: {test_df[self.timestamp_col].min()} - {test_df[self.timestamp_col].max()}")
        
        return train_df, val_df, test_df
    
    def _leave_one_out_split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Leave-one-out split: last interaction per user -> test, 
        second-to-last -> val, rest -> train.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        logger.info("Performing leave-one-out split")
        
        if self.timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' required for leave-one-out split")
        
        # Sort by user and timestamp
        df = df.sort_values([self.user_col, self.timestamp_col]).reset_index(drop=True)
        
        # Get interaction rank per user (1 = oldest, max = newest)
        df['interaction_rank'] = df.groupby(self.user_col).cumcount() + 1
        df['n_interactions'] = df.groupby(self.user_col)[self.user_col].transform('count')
        
        # Assign splits
        # Test: last interaction (rank == n_interactions)
        # Val: second-to-last (rank == n_interactions - 1)
        # Train: all others
        test_mask = df['interaction_rank'] == df['n_interactions']
        val_mask = df['interaction_rank'] == df['n_interactions'] - 1
        train_mask = ~(test_mask | val_mask)
        
        test_df = df[test_mask].drop(columns=['interaction_rank', 'n_interactions']).copy()
        val_df = df[val_mask].drop(columns=['interaction_rank', 'n_interactions']).copy()
        train_df = df[train_mask].drop(columns=['interaction_rank', 'n_interactions']).copy()
        
        # Handle users with only 1 interaction (go to test)
        # Users with 2 interactions: 1 train, 1 test (no val)
        
        logger.info(f"Leave-one-out split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _user_based_split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        User-based split: disjoint user sets for train/val/test.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        logger.info("Performing user-based split")
        
        unique_users = df[self.user_col].unique()
        np.random.seed(self.random_seed)
        np.random.shuffle(unique_users)
        
        n_users = len(unique_users)
        n_test_users = int(n_users * self.test_split)
        n_val_users = int(n_users * self.validation_split)
        n_train_users = n_users - n_val_users - n_test_users
        
        train_users = set(unique_users[:n_train_users])
        val_users = set(unique_users[n_train_users:n_train_users + n_val_users])
        test_users = set(unique_users[n_train_users + n_val_users:])
        
        train_df = df[df[self.user_col].isin(train_users)].copy()
        val_df = df[df[self.user_col].isin(val_users)].copy()
        test_df = df[df[self.user_col].isin(test_users)].copy()
        
        logger.info(f"User-based split: train_users={len(train_users)}, "
                   f"val_users={len(val_users)}, test_users={len(test_users)}")
        logger.info(f"Interactions: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str
    ) -> None:
        """
        Save split DataFrames to files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_parquet(output_path / 'train.parquet', index=False)
        val_df.to_parquet(output_path / 'validation.parquet', index=False)
        test_df.to_parquet(output_path / 'test.parquet', index=False)
        
        # Also save as CSV for readability
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'validation.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        logger.info(f"Saved splits to {output_path}")
    
    def load_splits(self, input_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load split DataFrames from files.
        
        Args:
            input_dir: Input directory
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        input_path = Path(input_dir)
        
        train_df = pd.read_parquet(input_path / 'train.parquet')
        val_df = pd.read_parquet(input_path / 'validation.parquet')
        test_df = pd.read_parquet(input_path / 'test.parquet')
        
        logger.info(f"Loaded splits from {input_path}: "
                   f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_split_stats(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute statistics for each split.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Dictionary with split statistics
        """
        stats = {}
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            stats[name] = {
                'n_interactions': len(df),
                'n_users': df[self.user_col].nunique(),
                'n_items': df[self.item_col].nunique(),
                'sparsity': 1 - len(df) / (df[self.user_col].nunique() * df[self.item_col].nunique()),
            }
            
            # User interaction stats
            user_counts = df.groupby(self.user_col).size()
            stats[name]['avg_interactions_per_user'] = user_counts.mean()
            stats[name]['median_interactions_per_user'] = user_counts.median()
            stats[name]['min_interactions_per_user'] = user_counts.min()
            stats[name]['max_interactions_per_user'] = user_counts.max()
            
            # Item interaction stats
            item_counts = df.groupby(self.item_col).size()
            stats[name]['avg_interactions_per_item'] = item_counts.mean()
            stats[name]['median_interactions_per_item'] = item_counts.median()
        
        # Overlap statistics
        train_users = set(train_df[self.user_col].unique())
        val_users = set(val_df[self.user_col].unique())
        test_users = set(test_df[self.user_col].unique())
        
        stats['user_overlap'] = {
            'train_val': len(train_users & val_users),
            'train_test': len(train_users & test_users),
            'val_test': len(val_users & test_users),
        }
        
        train_items = set(train_df[self.item_col].unique())
        val_items = set(val_df[self.item_col].unique())
        test_items = set(test_df[self.item_col].unique())
        
        stats['item_overlap'] = {
            'train_val': len(train_items & val_items),
            'train_test': len(train_items & test_items),
            'val_test': len(val_items & test_items),
        }
        
        logger.info(f"Split statistics computed")
        
        return stats


def split_data(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to split data.
    
    Args:
        df: Input DataFrame
        config: Optional configuration
        output_dir: Optional directory to save splits
        
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    splitter = DataSplitter(config)
    train_df, val_df, test_df = splitter.split(df)
    
    if output_dir:
        splitter.save_splits(train_df, val_df, test_df, output_dir)
    
    return train_df, val_df, test_df