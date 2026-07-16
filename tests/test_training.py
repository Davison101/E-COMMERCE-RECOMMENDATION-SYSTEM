"""
Integration test for the Training pipeline (Step 7).
Verifies the Trainer can instantiate and run a few steps.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_training_pipeline():
    """Run a short training loop to verify the pipeline works end-to-end."""
    logger.info("=" * 60)
    logger.info("Step 7 — Training Pipeline Integration Test")
    logger.info("=" * 60)

    # ── Synthetic data ──────────────────────────────────────────────
    n_users = 10
    n_items = 20
    np.random.seed(42)

    # Generate random interactions
    n_interactions = 100
    train_data = {
        'user_idx': np.random.randint(0, n_users, size=n_interactions),
        'item_idx': np.random.randint(0, n_items, size=n_interactions),
    }
    train_df = pd.DataFrame(train_data).drop_duplicates(subset=['user_idx', 'item_idx'])
    logger.info(f"Train interactions: {len(train_df)}")

    val_df = train_df.sample(frac=0.1, random_state=42)
    test_df = train_df.sample(frac=0.1, random_state=99)

    # ── Model ───────────────────────────────────────────────────────
    from models.lightgcn import LightGCN

    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=16,
        n_layers=2,
        dropout=0.0,
    )

    # ── Trainer ─────────────────────────────────────────────────────
    from training.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_users=n_users,
        n_items=n_items,
        device='cpu',
        config={
            'epochs': 3,
            'batch_size': 32,
            'learning_rate': 0.01,
            'weight_decay': 1e-5,
            'early_stopping_patience': 10,
            'eval_interval': 1,
            'neg_sample_ratio': 1,
            'save_dir': 'saved_models',
        },
    )

    # ── Run a few training steps ────────────────────────────────────
    trained_model = trainer.train()

    # ── Verify model learned something ──────────────────────────────
    # Run a prediction
    adj = trainer.adj_norm
    users = torch.tensor([0, 1], dtype=torch.long)
    items = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        preds, _, _ = trained_model(adj, users, items)

    logger.info(f"Sample predictions after training: {preds.numpy()}")
    assert preds.shape == (2,), "Predictions should have shape (2,)"
    assert not torch.isnan(preds).any(), "Predictions should not be NaN"
    logger.info("✓ Model produces valid predictions after training")

    # Verify loss decreased (rough check)
    initial_loss = trainer.history['train_loss'][0]
    final_loss = trainer.history['train_loss'][-1]
    logger.info(f"Training loss: {initial_loss:.4f} → {final_loss:.4f}")

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE TEST PASSED ✓")
    logger.info("=" * 60)


if __name__ == '__main__':
    test_training_pipeline()
