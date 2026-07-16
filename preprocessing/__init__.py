"""
Data Preprocessing Module for LightGCN Recommender System.

This package provides utilities for loading, cleaning, encoding, and splitting
user-item interaction data for training LightGCN recommendation models.

Modules:
    - load_data: Load and validate raw datasets
    - clean_data: Clean and validate data quality
    - encode: Encode user/item IDs to continuous indices
    - split_data: Split data into train/validation/test sets
"""

from preprocessing.load_data import DataLoader, load_raw_data, validate_and_summarize
from preprocessing.clean_data import DataCleaner, clean_data
from preprocessing.encode import IDEncoder, encode_ids
from preprocessing.split_data import DataSplitter, split_data

__all__ = [
    'DataLoader',
    'DataCleaner',
    'IDEncoder',
    'DataSplitter',
    'load_raw_data',
    'validate_and_summarize',
    'clean_data',
    'encode_ids',
    'split_data',
]

__version__ = '1.0.0'
__author__ = 'LightGCN Recommender System'