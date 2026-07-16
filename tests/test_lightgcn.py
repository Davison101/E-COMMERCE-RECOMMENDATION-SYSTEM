"""
Test script for the LightGCN model (Step 6).
Verifies forward pass, propagation, and gradient flow.
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


def csr_to_sparse_tensor(adj_matrix) -> torch.sparse.Tensor:
    """Convert a scipy CSR matrix to a PyTorch sparse COO tensor."""
    import scipy.sparse as sp
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)
    adj_coo = adj_matrix.tocoo()
    indices = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
    values = torch.tensor(adj_coo.data, dtype=torch.float32)
    size = adj_coo.shape
    return torch.sparse_coo_tensor(indices, values, size).coalesce()


def test_lightgcn_forward():
    """Test LightGCN forward pass with synthetic data."""
    logger.info("=" * 60)
    logger.info("Step 6 — LightGCN Model Test")
    logger.info("=" * 60)

    # ── Synthetic data ──────────────────────────────────────────────
    n_users = 3
    n_items = 4
    n_nodes = n_users + n_items
    dim = 8
    n_layers = 2

    # Build a tiny graph first
    train_df = pd.DataFrame({
        'user_idx': [0, 0, 1, 1, 2, 2],
        'item_idx': [0, 1, 1, 2, 2, 3],
    })

    from graph.build_graph import GraphBuilder
    builder = GraphBuilder()
    builder.build_from_dataframe(train_df, n_users, n_items)
    adj_norm_sparse = csr_to_sparse_tensor(builder.get_normalized_adj())

    logger.info(f"Adjacency matrix shape: {adj_norm_sparse.shape}")
    logger.info(f"Adjacency matrix nnz: {adj_norm_sparse._nnz()}")

    # ── Instantiate model ───────────────────────────────────────────
    from models.lightgcn import LightGCN

    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=dim,
        n_layers=n_layers,
        alpha='mean',
        dropout=0.0,
    )

    logger.info(f"Model: {model}")
    logger.info(f"User embeddings shape: {model.user_embedding.weight.shape}")
    logger.info(f"Item embeddings shape: {model.item_embedding.weight.shape}")

    # ── Forward pass with batch ─────────────────────────────────────
    batch_users = torch.tensor([0, 1], dtype=torch.long)
    batch_items = torch.tensor([0, 2], dtype=torch.long)

    preds, user_out, item_out = model(adj_norm_sparse, batch_users, batch_items)

    logger.info(f"Predictions: {preds.detach().numpy()}")
    logger.info(f"User output shape: {user_out.shape}")
    logger.info(f"Item output shape: {item_out.shape}")

    assert preds.shape == (2,), f"Expected preds shape (2,), got {preds.shape}"
    assert user_out.shape == (2, dim), f"Expected user_out shape (2, {dim}), got {user_out.shape}"
    assert item_out.shape == (2, dim), f"Expected item_out shape (2, {dim}), got {item_out.shape}"
    logger.info("✓ Forward pass produces correct shapes")

    # ── Gradients flow ──────────────────────────────────────────────
    loss = preds.sum()
    loss.backward()

    assert model.user_embedding.weight.grad is not None, \
        "User embedding gradients should not be None"
    assert model.item_embedding.weight.grad is not None, \
        "Item embedding gradients should not be None"
    logger.info("✓ Gradients flow to all parameters")
    logger.info(f"  User emb grad norm: {model.user_embedding.weight.grad.norm().item():.4f}")
    logger.info(f"  Item emb grad norm: {model.item_embedding.weight.grad.norm().item():.4f}")

    # ── Propagation produces embeddings with correct shapes ─────────
    user_embs, item_embs = model.propagate(adj_norm_sparse)
    assert len(user_embs) == n_layers + 1, \
        f"Expected {n_layers + 1} user embedding layers, got {len(user_embs)}"
    assert len(item_embs) == n_layers + 1, \
        f"Expected {n_layers + 1} item embedding layers, got {len(item_embs)}"
    for k in range(n_layers + 1):
        assert user_embs[k].shape == (n_users, dim), \
            f"Layer {k} user emb shape: {user_embs[k].shape}"
        assert item_embs[k].shape == (n_items, dim), \
            f"Layer {k} item emb shape: {item_embs[k].shape}"
    logger.info("✓ Propagation produces correct embedding shapes per layer")

    # ── Layer embeddings differ (propagation mixes information) ─────
    for k in range(1, n_layers + 1):
        diff = (user_embs[k] - user_embs[0]).norm().item()
        logger.info(f"  Layer {k} user emb diff from layer 0: {diff:.4f}")
        assert diff > 0, \
            f"Layer {k} embeddings should differ from layer 0"
    logger.info("✓ Propagation changes embeddings across layers")

    # ── predict_all_users ───────────────────────────────────────────
    score_matrix = model.predict_all_users(adj_norm_sparse)
    assert score_matrix.shape == (n_users, n_items), \
        f"Expected score matrix ({n_users}, {n_items}), got {score_matrix.shape}"
    logger.info(f"✓ Score matrix shape: {score_matrix.shape}")

    # ── Save / load round-trip ──────────────────────────────────────
    save_path = project_root / 'saved_models' / 'test_lightgcn.pt'
    model.save(str(save_path))

    model2 = LightGCN(n_users, n_items, dim, n_layers)
    model2.load(str(save_path))

    with torch.no_grad():
        preds2, _, _ = model2(adj_norm_sparse, batch_users, batch_items)
    assert torch.allclose(preds, preds2), \
        "Predictions should match after save/load"
    logger.info("✓ Save / load round-trip preserves predictions")

    # Cleanup
    save_path.unlink(missing_ok=True)

    logger.info("=" * 60)
    logger.info("ALL LIGHTGCN MODEL TESTS PASSED ✓")
    logger.info("=" * 60)


if __name__ == '__main__':
    test_lightgcn_forward()
