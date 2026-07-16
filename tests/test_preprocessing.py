"""
Test script for the Data Preprocessing module.
Verifies that all preprocessing components work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_users: int = 100, n_items: int = 50, n_interactions: int = 1000) -> pd.DataFrame:
    """Create sample interaction data for testing."""
    np.random.seed(42)
    
    data = {
        'user_id': np.random.randint(1, n_users + 1, size=n_interactions),
        'item_id': np.random.randint(1, n_items + 1, size=n_interactions),
        'rating': np.random.randint(1, 6, size=n_interactions).astype(float),
        'timestamp': np.random.randint(800000000, 1700000000, size=n_interactions),
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample data: {len(df)} interactions, "
                f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    return df


def test_load_data():
    """Test data loading functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Data Loading")
    logger.info("=" * 60)
    
    from preprocessing.load_data import DataLoader
    
    # Create sample data and save to CSV
    df = create_sample_data()
    test_file = project_root / 'data' / 'raw' / 'test_data.csv'
    test_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_file, index=False)
    
    # Load data
    loader = DataLoader(config={
        'raw_data_path': str(test_file),
        'expected_columns': ['user_id', 'item_id', 'rating', 'timestamp'],
        'rating_scale': [1, 5],
        'separator': ',',
        'header': 0,
    })
    
    loaded_df = loader.load_data()
    assert len(loaded_df) == len(df), "Loaded data should match original"
    
    # Validate schema
    assert loader.validate_schema(loaded_df), "Schema validation should pass"
    
    # Get summary
    summary = loader.get_dataset_summary(loaded_df)
    logger.info(f"Summary: {summary['n_interactions']} interactions, "
                f"{summary['n_users']} users, {summary['n_items']} items")
    
    loader.print_summary(loaded_df)
    
    # Cleanup
    test_file.unlink()
    logger.info("✓ Data loading test passed\n")


def test_clean_data():
    """Test data cleaning functionality."""
    logger.info("=" * 60)
    logger.info("TEST 2: Data Cleaning")
    logger.info("=" * 60)
    
    from preprocessing.clean_data import DataCleaner
    
    # Create a clean base dataset
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'user_id': rng.randint(1, 51, size=200),
        'item_id': rng.randint(100, 201, size=200),
        'rating': rng.uniform(1, 5, size=200).round(1),
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='h'),
    })
    
    # Add some duplicates and missing values for testing
    df_dirty = df.copy()
    df_dirty = pd.concat([df_dirty, df_dirty.iloc[:10]])  # Add duplicates
    df_dirty = df_dirty.reset_index(drop=True)  # Fix duplicate indices
    df_dirty.loc[0:4, 'rating'] = np.nan  # Add missing values
    df_dirty.loc[5:9, 'user_id'] = -1  # Add invalid IDs
    df_dirty.loc[10:14, 'rating'] = 10  # Add out-of-range ratings
    
    logger.info(f"Dirty data: {len(df_dirty)} rows")
    
    cleaner = DataCleaner(config={
        'remove_duplicates': True,
        'handle_missing': 'drop',
    })
    
    cleaned_df = cleaner.clean(df_dirty)
    
    assert len(cleaned_df) < len(df_dirty), "Cleaning should remove rows"
    assert cleaned_df['rating'].isna().sum() == 0, "No missing ratings should remain"
    assert (cleaned_df['user_id'] > 0).all(), "All user IDs should be positive"
    assert (cleaned_df['rating'] >= 1).all() and (cleaned_df['rating'] <= 5).all(), \
        "All ratings should be in range [1, 5]"
    
    logger.info(f"Cleaned data: {len(cleaned_df)} rows")
    logger.info("✓ Data cleaning test passed\n")


def test_encode_ids():
    """Test ID encoding functionality."""
    logger.info("=" * 60)
    logger.info("TEST 3: ID Encoding")
    logger.info("=" * 60)
    
    from preprocessing.encode import IDEncoder
    
    # Create test data
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'user_id': rng.randint(1, 51, size=200).astype(str),
        'item_id': rng.randint(100, 201, size=200).astype(str),
        'rating': rng.uniform(1, 5, size=200).round(1),
    })
    
    encoder = IDEncoder()
    encoded_df = encoder.fit_transform(df)
    
    assert 'user_idx' in encoded_df.columns, "user_idx column should be added"
    assert 'item_idx' in encoded_df.columns, "item_idx column should be added"
    assert encoded_df['user_idx'].min() == 0, "User indices should start at 0"
    assert encoded_df['item_idx'].min() == 0, "Item indices should start at 0"
    assert encoded_df['user_idx'].max() == encoder.n_users - 1, "Max user index should be n_users - 1"
    assert encoded_df['item_idx'].max() == encoder.n_items - 1, "Max item index should be n_items - 1"
    
    logger.info(f"Encoded {encoder.n_users} users and {encoder.n_items} items")
    logger.info(f"User index range: [{encoded_df['user_idx'].min()}, {encoded_df['user_idx'].max()}]")
    logger.info(f"Item index range: [{encoded_df['item_idx'].min()}, {encoded_df['item_idx'].max()}]")
    
    # Test save/load
    test_mapping_path = project_root / 'data' / 'processed' / 'test_mappings'
    encoder.save_mappings(str(test_mapping_path))
    
    # Load and verify
    encoder2 = IDEncoder()
    encoder2.load_mappings(str(test_mapping_path))
    assert encoder2.n_users == encoder.n_users, "Loaded encoder should match"
    assert encoder2.n_items == encoder.n_items, "Loaded encoder should match"
    
    # Cleanup
    import shutil
    if test_mapping_path.exists():
        shutil.rmtree(test_mapping_path)
    
    logger.info("✓ ID encoding test passed\n")


def test_split_data():
    """Test data splitting functionality."""
    logger.info("=" * 60)
    logger.info("TEST 4: Data Splitting")
    logger.info("=" * 60)
    
    from preprocessing.split_data import DataSplitter
    
    # Create test data
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'user_id': rng.randint(1, 51, size=500).astype(str),
        'item_id': rng.randint(100, 201, size=500).astype(str),
        'rating': rng.uniform(1, 5, size=500).round(1),
        'timestamp': pd.date_range('2024-01-01', periods=500, freq='h'),
    })
    
    # Test random split
    splitter = DataSplitter(config={
        'validation_split': 0.1,
        'test_split': 0.1,
        'random_seed': 42,
        'split_strategy': 'random',
    })
    
    train_df, val_df, test_df = splitter.split(df)
    
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df), f"Split should preserve all rows: {total} != {len(df)}"
    
    logger.info(f"Random split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    logger.info(f"Split proportions: "
                f"train={len(train_df)/len(df):.2%}, "
                f"val={len(val_df)/len(df):.2%}, "
                f"test={len(test_df)/len(df):.2%}")
    
    # Test save/load
    test_output_dir = project_root / 'data' / 'processed' / 'test_splits'
    splitter.save_splits(train_df, val_df, test_df, str(test_output_dir))
    
    # Load and verify
    train_loaded, val_loaded, test_loaded = DataSplitter().load_splits(str(test_output_dir))
    assert len(train_loaded) == len(train_df), "Loaded train should match"
    
    # Cleanup
    import shutil
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
    
    logger.info("✓ Data splitting test passed\n")


def test_full_pipeline():
    """Test the complete preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("TEST 5: Full Pipeline Integration")
    logger.info("=" * 60)
    
    # Create sample data
    df = create_sample_data(n_users=200, n_items=100, n_interactions=5000)
    
    # Save to file
    test_file = project_root / 'data' / 'raw' / 'pipeline_test.csv'
    test_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_file, index=False)
    
    # Run full pipeline
    from preprocessing.load_data import DataLoader
    from preprocessing.clean_data import DataCleaner
    from preprocessing.encode import IDEncoder
    from preprocessing.split_data import DataSplitter
    
    # 1. Load
    loader = DataLoader(config={
        'raw_data_path': str(test_file),
        'expected_columns': ['user_id', 'item_id', 'rating', 'timestamp'],
        'rating_scale': [1, 5],
        'separator': ',',
        'header': 0,
    })
    df = loader.load_data()
    loader.validate_schema(df)
    
    # 2. Clean
    cleaner = DataCleaner(config={'remove_duplicates': True, 'handle_missing': 'drop'})
    df = cleaner.clean(df)
    
    # 3. Encode
    encoder = IDEncoder()
    df = encoder.fit_transform(df)
    
    # 4. Split
    splitter = DataSplitter(config={
        'validation_split': 0.1,
        'test_split': 0.1,
        'random_seed': 42,
        'split_strategy': 'random',
    })
    train_df, val_df, test_df = splitter.split(df)
    
    logger.info(f"Pipeline complete!")
    logger.info(f"  Original: {len(df)} interactions")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    logger.info(f"  Validation: {len(val_df)} ({len(val_df)/len(df):.1%})")
    logger.info(f"  Test: {len(test_df)} ({len(test_df)/len(df):.1%})")
    logger.info(f"  Users: {encoder.n_users}")
    logger.info(f"  Items: {encoder.n_items}")
    
    # Cleanup
    test_file.unlink()
    logger.info("✓ Full pipeline test passed\n")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("LightGCN Recommender - Preprocessing Module Tests")
    logger.info("=" * 60 + "\n")
    
    try:
        # Run individual tests
        df = test_load_data()
        cleaned_df = test_clean_data(df)
        encoded_df = test_encode_ids(cleaned_df)
        test_split_data(encoded_df)
        
        # Run full pipeline test
        test_full_pipeline()
        
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED SUCCESSFULLY! ✓")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()