"""Data preprocessing module for LightGCN Recommender System."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
from scipy.sparse import csr_matrix, coo_matrix

from utils import get_logger
from utils.config import get_config


logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocesses user-item interaction data for LightGCN training.
    
    Handles:
    - Loading raw interaction data
    - Filtering users/items with insufficient interactions
    - Creating user/item ID mappings
    - Splitting data into train/validation/test sets
    - Building adjacency matrices
    """
    
    def __init__(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize the preprocessor.
        
        Args:
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            test_ratio: Ratio of test data
            val_ratio: Ratio of validation data
            random_seed: Random seed for reproducibility
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        self.config = get_config()
        
        # Mappings
        self.user2id: Dict[int, int] = {}
        self.id2user: Dict[int, int] = {}
        self.item2id: Dict[int, int] = {}
        self.id2item: Dict[int, int] = {}
        
        # Data containers
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.full_data: Optional[pd.DataFrame] = None
        
        # Statistics
        self.n_users: int = 0
        self.n_items: int = 0
        self.n_interactions: int = 0
        
        logger.info(
            f"Initialized DataPreprocessor with "
            f"min_user_interactions={min_user_interactions}, "
            f"min_item_interactions={min_item_interactions}, "
            f"test_ratio={test_ratio}, val_ratio={val_ratio}"
        )
    
    def load_data(
        self,
        file_path: str,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        rating_col: Optional[str] = 'rating',
        timestamp_col: Optional[str] = 'timestamp',
        sep: str = ','
    ) -> pd.DataFrame:
        """
        Load interaction data from file.
        
        Args:
            file_path: Path to data file
            user_col: User ID column name
            item_col: Item ID column name
            rating_col: Rating column name (optional)
            timestamp_col: Timestamp column name (optional)
            sep: Separator for CSV files
            
        Returns:
            DataFrame with standardized column names
        """
        logger.info(f"Loading data from {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load based on file extension
        if path.suffix == '.csv':
            df = pd.read_csv(file_path, sep=sep)
        elif path.suffix in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        elif path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Standardize column names
        rename_map = {
            user_col: 'user_id',
            item_col: 'item_id'
        }
        if rating_col and rating_col in df.columns:
            rename_map[rating_col] = 'rating'
        if timestamp_col and timestamp_col in df.columns:
            rename_map[timestamp_col] = 'timestamp'
        
        df = df.rename(columns=rename_map)
        
        # Ensure required columns exist
        required_cols = ['user_id', 'item_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert to appropriate types
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        
        if 'rating' in df.columns:
            df['rating'] = df['rating'].astype(float)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        logger.info(f"Loaded {len(df)} interactions from {file_path}")
        logger.info(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}")
        
        self.full_data = df.copy()
        return df
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter users and items with insufficient interactions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering users and items with insufficient interactions")
        
        initial_users = df['user_id'].nunique()
        initial_items = df['item_id'].nunique()
        initial_interactions = len(df)
        
        # Iterative filtering until convergence
        prev_users, prev_items = 0, 0
        iteration = 0
        
        while (prev_users != df['user_id'].nunique() or 
               prev_items != df['item_id'].nunique()):
            prev_users = df['user_id'].nunique()
            prev_items = df['item_id'].nunique()
            
            # Filter users
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            df = df[df['user_id'].isin(valid_users)]
            
            # Filter items
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            df = df[df['item_id'].isin(valid_items)]
            
            iteration += 1
            if iteration > 10:
                logger.warning("Filtering did not converge after 10 iterations")
                break
        
        logger.info(
            f"Filtering complete: "
            f"Users {initial_users} -> {df['user_id'].nunique()}, "
            f"Items {initial_items} -> {df['item_id'].nunique()}, "
            f"Interactions {initial_interactions} -> {len(df)}"
        )
        
        return df
    
    def create_id_mappings(self, df: pd.DataFrame) -> None:
        """
        Create continuous ID mappings for users and items.
        
        Args:
            df: Filtered DataFrame
        """
        logger.info("Creating ID mappings")
        
        # Create user mapping
        unique_users = sorted(df['user_id'].unique())
        self.user2id = {uid: idx for idx, uid in enumerate(unique_users)}
        self.id2user = {idx: uid for uid, idx in self.user2id.items()}
        
        # Create item mapping
        unique_items = sorted(df['item_id'].unique())
        self.item2id = {iid: idx for idx, iid in enumerate(unique_items)}
        self.id2item = {idx: iid for iid, idx in self.item2id.items()}
        
        self.n_users = len(self.user2id)
        self.n_items = len(self.item2id)
        
        logger.info(f"Created mappings: {self.n_users} users, {self.n_items} items")
    
    def map_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map original IDs to continuous indices.
        
        Args:
            df: DataFrame with original IDs
            
        Returns:
            DataFrame with mapped IDs
        """
        df = df.copy()
        df['user_idx'] = df['user_id'].map(self.user2id)
        df['item_idx'] = df['item_id'].map(self.item2id)
        
        # Check for unmapped IDs
        if df['user_idx'].isna().any() or df['item_idx'].isna().any():
            raise ValueError("Some IDs could not be mapped. Check mappings.")
        
        df['user_idx'] = df['user_idx'].astype(int)
        df['item_idx'] = df['item_idx'].astype(int)
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        strategy: str = 'random'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: DataFrame with mapped IDs
            strategy: Split strategy ('random', 'temporal', 'leave_one_out')
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data using strategy: {strategy}")
        
        np.random.seed(self.random_seed)
        
        if strategy == 'random':
            return self._random_split(df)
        elif strategy == 'temporal':
            return self._temporal_split(df)
        elif strategy == 'leave_one_out':
            return self._leave_one_out_split(df)
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
    
    def _random_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Random split by interactions."""
        n = len(df)
        indices = np.random.permutation(n)
        
        test_size = int(n * self.test_ratio)
        val_size = int(n * self.val_ratio)
        
        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size + val_size]
        train_idx = indices[test_size + val_size:]
        
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    
    def _temporal_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal split based on timestamp."""
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column, falling back to random split")
            return self._random_split(df)
        
        df = df.sort_values('timestamp')
        n = len(df)
        
        test_size = int(n * self.test_ratio)
        val_size = int(n * self.val_ratio)
        
        train_df = df.iloc[:-test_size - val_size]
        val_df = df.iloc[-test_size - val_size:-test_size]
        test_df = df.iloc[-test_size:]
        
        return train_df, val_df, test_df
    
    def _leave_one_out_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Leave-one-out split per user (last interaction as test)."""
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column, using random leave-one-out")
            df = df.sample(frac=1, random_state=self.random_seed)
        
        # Sort by user and timestamp
        df = df.sort_values(['user_idx', 'timestamp'])
        
        # Get last interaction per user as test
        test_df = df.groupby('user_idx').tail(1)
        
        # Get second-to-last as validation
        val_df = df.groupby('user_idx').apply(
            lambda x: x.iloc[-2:-1] if len(x) > 1 else x.iloc[0:0]
        ).reset_index(drop=True)
        
        # Rest is training
        train_df = df.drop(test_df.index).drop(val_df.index)
        
        return train_df, val_df, test_df
    
    def build_adjacency_matrix(
        self,
        df: pd.DataFrame,
        normalize: bool = True
    ) -> csr_matrix:
        """
        Build normalized adjacency matrix for LightGCN.
        
        Args:
            df: Training DataFrame with mapped indices
            normalize: Whether to apply symmetric normalization
            
        Returns:
            Normalized adjacency matrix (n_users + n_items) x (n_users + n_items)
        """
        logger.info("Building adjacency matrix")
        
        n_nodes = self.n_users + self.n_items
        
        # User-item interactions
        user_indices = df['user_idx'].values
        item_indices = df['item_idx'].values + self.n_users  # Offset items
        
        # Create bipartite adjacency matrix
        # Rows: users (0 to n_users-1), Cols: items (n_users to n_users+n_items-1)
        data = np.ones(len(user_indices), dtype=np.float32)
        
        # User -> Item edges
        row = np.concatenate([user_indices, item_indices])
        col = np.concatenate([item_indices, user_indices])
        data = np.concatenate([data, data])
        
        adj = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        adj = adj.tocsr()
        
        if normalize:
            adj = self._normalize_adjacency(adj)
        
        logger.info(f"Built adjacency matrix: {adj.shape}, nnz: {adj.nnz}")
        return adj
    
    def _normalize_adjacency(self, adj: csr_matrix) -> csr_matrix:
        """
        Apply symmetric normalization: D^(-1/2) * A * D^(-1/2).
        
        Args:
            adj: Adjacency matrix
            
        Returns:
            Normalized adjacency matrix
        """
        # Compute degree matrix
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        
        # Create diagonal matrix
        d_mat_inv_sqrt = csr_matrix(
            (d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))),
            shape=adj.shape
        )
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        return normalized.tocsr()
    
    def get_user_item_interactions(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        Get user-item interaction dictionary for evaluation.
        
        Args:
            df: DataFrame with mapped indices
            
        Returns:
            Dictionary mapping user_idx to list of item_idx
        """
        interactions = df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        return interactions
    
    def process(
        self,
        file_path: str,
        **load_kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, csr_matrix]:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to raw data file
            **load_kwargs: Arguments for load_data
            
        Returns:
            Tuple of (train_df, val_df, test_df, adj_matrix)
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Load data
        df = self.load_data(file_path, **load_kwargs)
        
        # Filter
        df = self.filter_data(df)
        
        # Create mappings
        self.create_id_mappings(df)
        
        # Map IDs
        df = self.map_ids(df)
        
        # Split
        self.train_data, self.val_data, self.test_data = self.split_data(df)
        
        # Build adjacency matrix from training data
        adj_matrix = self.build_adjacency_matrix(self.train_data)
        
        self.n_interactions = len(df)
        
        logger.info(
            f"Preprocessing complete: "
            f"Train={len(self.train_data)}, Val={len(self.val_data)}, "
            f"Test={len(self.test_data)}, Users={self.n_users}, "
            f"Items={self.n_items}, Interactions={self.n_interactions}"
        )
        
        return self.train_data, self.val_data, self.test_data, adj_matrix
    
    def save_processed_data(self, output_dir: str) -> None:
        """
        Save processed data and mappings.
        
        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        if self.train_data is not None:
            self.train_data.to_parquet(output_path / 'train.parquet', index=False)
        if self.val_data is not None:
            self.val_data.to_parquet(output_path / 'val.parquet', index=False)
        if self.test_data is not None:
            self.test_data.to_parquet(output_path / 'test.parquet', index=False)
        
        # Save mappings
        import json
        with open(output_path / 'user2id.json', 'w') as f:
            json.dump(self.user2id, f)
        with open(output_path / 'item2id.json', 'w') as f:
            json.dump(self.item2id, f)
        
        # Save statistics
        stats = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_interactions': self.n_interactions,
            'n_train': len(self.train_data) if self.train_data is not None else 0,
            'n_val': len(self.val_data) if self.val_data is not None else 0,
            'n_test': len(self.test_data) if self.test_data is not None else 0,
        }
        with open(output_path / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")
    
    @classmethod
    def load_processed_data(cls, input_dir: str) -> 'DataPreprocessor':
        """
        Load previously processed data.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            DataPreprocessor instance with loaded data
        """
        input_path = Path(input_dir)
        
        preprocessor = cls()
        
        # Load mappings
        import json
        with open(input_path / 'user2id.json', 'r') as f:
            preprocessor.user2id = {int(k): v for k, v in json.load(f).items()}
        with open(input_path / 'item2id.json', 'r') as f:
            preprocessor.item2id = {int(k): v for k, v in json.load(f).items()}
        
        preprocessor.id2user = {v: k for k, v in preprocessor.user2id.items()}
        preprocessor.id2item = {v: k for k, v in preprocessor.item2id.items()}
        preprocessor.n_users = len(preprocessor.user2id)
        preprocessor.n_items = len(preprocessor.item2id)
        
        # Load splits
        preprocessor.train_data = pd.read_parquet(input_path / 'train.parquet')
        preprocessor.val_data = pd.read_parquet(input_path / 'val.parquet')
        preprocessor.test_data = pd.read_parquet(input_path / 'test.parquet')
        
        # Load stats
        with open(input_path / 'stats.json', 'r') as f:
            stats = json.load(f)
        preprocessor.n_interactions = stats['n_interactions']
        
        logger.info(f"Loaded processed data from {input_dir}")
        return preprocessor