"""
ID encoding module for LightGCN Recommender System.
Maps original user/item IDs to continuous indices for model training.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

from utils.config import get_config

logger = logging.getLogger(__name__)


class IDEncoder:
    """
    Encodes categorical user/item IDs to continuous integer indices.
    
    Provides bidirectional mapping between original IDs and encoded indices.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IDEncoder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().preprocessing.get('encoding', {})
        self.save_enabled = self.config.get('save_mappings', True)
        self.mapping_path = self.config.get('mapping_path', 'data/processed/mappings')
        
        # Mappings
        self.user2id: Dict[int, int] = {}
        self.id2user: Dict[int, int] = {}
        self.item2id: Dict[int, int] = {}
        self.id2item: Dict[int, int] = {}
        
        logger.info("IDEncoder initialized")
    
    def fit(self, df: pd.DataFrame, user_col: str = 'user_id', item_col: str = 'item_id') -> 'IDEncoder':
        """
        Learn mappings from data.
        
        Args:
            df: DataFrame with user/item columns
            user_col: User ID column name
            item_col: Item ID column name
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting ID encoder")
        
        # Get unique IDs sorted for reproducibility
        unique_users = sorted(df[user_col].unique())
        unique_items = sorted(df[item_col].unique())
        
        # Create mappings
        self.user2id = {uid: idx for idx, uid in enumerate(unique_users)}
        self.id2user = {idx: uid for uid, idx in self.user2id.items()}
        self.item2id = {iid: idx for idx, iid in enumerate(unique_items)}
        self.id2item = {idx: iid for iid, idx in self.item2id.items()}
        
        logger.info(f"Fitted encoder: {len(self.user2id)} users, {len(self.item2id)} items")
        
        return self
    
    def transform(
        self, 
        df: pd.DataFrame, 
        user_col: str = 'user_id', 
        item_col: str = 'item_id',
        user_idx_col: str = 'user_idx',
        item_idx_col: str = 'item_idx'
    ) -> pd.DataFrame:
        """
        Transform original IDs to indices.
        
        Args:
            df: DataFrame with original IDs
            user_col: Original user ID column
            item_col: Original item ID column
            user_idx_col: Output user index column
            item_idx_col: Output item index column
            
        Returns:
            DataFrame with added index columns
        """
        logger.info("Transforming IDs to indices")
        
        df = df.copy()
        
        # Map user IDs
        df[user_idx_col] = df[user_col].map(self.user2id)
        
        # Map item IDs
        df[item_idx_col] = df[item_col].map(self.item2id)
        
        # Check for unmapped IDs
        n_unmapped_users = df[user_idx_col].isna().sum()
        n_unmapped_items = df[item_idx_col].isna().sum()
        
        if n_unmapped_users > 0:
            logger.warning(f"{n_unmapped_users} user IDs could not be mapped")
        if n_unmapped_items > 0:
            logger.warning(f"{n_unmapped_items} item IDs could not be mapped")
        
        # Convert to int (NaN will become NaN, handled above)
        df[user_idx_col] = df[user_idx_col].astype('Int64')  # nullable int
        df[item_idx_col] = df[item_idx_col].astype('Int64')
        
        logger.info(f"Transformed {len(df)} rows")
        
        return df
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        user_idx_col: str = 'user_idx',
        item_idx_col: str = 'item_idx',
        user_col: str = 'user_id',
        item_col: str = 'item_id'
    ) -> pd.DataFrame:
        """
        Transform indices back to original IDs.
        
        Args:
            df: DataFrame with index columns
            user_idx_col: User index column
            item_idx_col: Item index column
            user_col: Output user ID column
            item_col: Output item ID column
            
        Returns:
            DataFrame with original ID columns
        """
        logger.info("Inverse transforming indices to IDs")
        
        df = df.copy()
        
        df[user_col] = df[user_idx_col].map(self.id2user)
        df[item_col] = df[item_idx_col].map(self.id2item)
        
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        user_idx_col: str = 'user_idx',
        item_idx_col: str = 'item_idx'
    ) -> pd.DataFrame:
        """
        Fit encoder and transform data in one step.
        
        Args:
            df: DataFrame with original IDs
            user_col: User ID column
            item_col: Item ID column
            user_idx_col: Output user index column
            item_idx_col: Output item index column
            
        Returns:
            DataFrame with added index columns
        """
        return self.fit(df, user_col, item_col).transform(
            df, user_col, item_col, user_idx_col, item_idx_col
        )
    
    def save_mappings(self, path: Optional[str] = None) -> None:
        """
        Save mappings to JSON files.
        
        Args:
            path: Directory path to save mappings
        """
        if not self.save_enabled:
            logger.info("Mapping saving disabled in config")
            return
        
        save_path = Path(path or self.mapping_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (keys must be strings in JSON)
        with open(save_path / 'user2id.json', 'w') as f:
            json.dump({str(k): v for k, v in self.user2id.items()}, f, indent=2)
        
        with open(save_path / 'id2user.json', 'w') as f:
            json.dump({str(k): v for k, v in self.id2user.items()}, f, indent=2)
        
        with open(save_path / 'item2id.json', 'w') as f:
            json.dump({str(k): v for k, v in self.item2id.items()}, f, indent=2)
        
        with open(save_path / 'id2item.json', 'w') as f:
            json.dump({str(k): v for k, v in self.id2item.items()}, f, indent=2)
        
        logger.info(f"Saved mappings to {save_path}")
    
    def load_mappings(self, path: Optional[str] = None) -> 'IDEncoder':
        """
        Load mappings from JSON files.
        
        Args:
            path: Directory path to load mappings from
            
        Returns:
            Self for chaining
        """
        load_path = Path(path or self.mapping_path)
        
        with open(load_path / 'user2id.json', 'r') as f:
            self.user2id = {int(k): v for k, v in json.load(f).items()}
        
        with open(load_path / 'id2user.json', 'r') as f:
            self.id2user = {int(k): v for k, v in json.load(f).items()}
        
        with open(load_path / 'item2id.json', 'r') as f:
            self.item2id = {int(k): v for k, v in json.load(f).items()}
        
        with open(load_path / 'id2item.json', 'r') as f:
            self.id2item = {int(k): v for k, v in json.load(f).items()}
        
        logger.info(f"Loaded mappings from {load_path}: "
                   f"{len(self.user2id)} users, {len(self.item2id)} items")
        
        return self
    
    @property
    def n_users(self) -> int:
        """Number of unique users."""
        return len(self.user2id)
    
    @property
    def n_items(self) -> int:
        """Number of unique items."""
        return len(self.item2id)
    
    def get_user_mapping(self) -> Dict[int, int]:
        """Get user ID to index mapping."""
        return self.user2id.copy()
    
    def get_item_mapping(self) -> Dict[int, int]:
        """Get item ID to index mapping."""
        return self.item2id.copy()
    
    def get_reverse_user_mapping(self) -> Dict[int, int]:
        """Get index to user ID mapping."""
        return self.id2user.copy()
    
    def get_reverse_item_mapping(self) -> Dict[int, int]:
        """Get index to item ID mapping."""
        return self.id2item.copy()


def encode_ids(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, IDEncoder]:
    """
    Convenience function to encode IDs.
    
    Args:
        df: DataFrame with original IDs
        user_col: User ID column
        item_col: Item ID column
        save_path: Optional path to save mappings
        
    Returns:
        Tuple of (encoded DataFrame, fitted encoder)
    """
    encoder = IDEncoder()
    encoded_df = encoder.fit_transform(df, user_col, item_col)
    
    if save_path:
        encoder.save_mappings(save_path)
    
    return encoded_df, encoder