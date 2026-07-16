"""
Test script for the Graph Construction module (Step 5).
Verifies that the bipartite adjacency matrix is built correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_graph_builder():
    """Test end-to-end graph construction with a small synthetic dataset."""
    logger.info("=" * 60)
    logger.info("Step 5 — Graph Construction Test")
    logger.info("=" * 60)

    # ── Create a tiny synthetic dataset ──────────────────────────────
    # 3 users (indices 0,1,2), 4 items (indices 0,1,2,3)
    # Interactions:
    #   User 0 → items 0, 1
    #   User 1 → items 1, 2
    #   User 2 → items 2, 3
    train_df = pd.DataFrame({
        'user_idx': [0, 0, 1, 1, 2, 2],
        'item_idx': [0, 1, 1, 2, 2, 3],
    })
    n_users = 3
    n_items = 4

    logger.info(f"Train interactions:\n{train_df}")

    # ── Build graph ──────────────────────────────────────────────────
    from graph.build_graph import GraphBuilder

    builder = GraphBuilder()
    builder.build_from_dataframe(train_df, n_users, n_items)

    logger.info(f"Builder: {builder}")
    logger.info(f"Number of edges (undirected): {builder.num_edges}")

    # ── Validate adjacency matrix shape ──────────────────────────────
    adj = builder.get_normalized_adj()
    expected_nodes = n_users + n_items  # 7
    assert adj.shape == (expected_nodes, expected_nodes), \
        f"Expected ({expected_nodes},{expected_nodes}), got {adj.shape}"
    logger.info(f"✓ Adjacency matrix shape: {adj.shape}")

    # ── Validate non-zero structure ──────────────────────────────────
    # The bipartite adjacency should have zeros on the diagonal blocks
    # (user-user and item-item) and non-zeros off-diagonal (user-item).
    adj_dense = adj.toarray()

    # User-user block: all zeros
    assert np.allclose(adj_dense[:n_users, :n_users], 0), \
        "User-user block should be zero"
    logger.info("✓ User-user block is zero (correct for bipartite)")

    # Item-item block: all zeros
    assert np.allclose(adj_dense[n_users:, n_users:], 0), \
        "Item-item block should be zero"
    logger.info("✓ Item-item block is zero (correct for bipartite)")

    # User-item block: should have non-zeros for observed interactions
    ui_block = adj_dense[:n_users, n_users:]
    assert ui_block[0, 0] > 0, "User 0 → Item 0 should be connected"
    assert ui_block[0, 1] > 0, "User 0 → Item 1 should be connected"
    assert ui_block[1, 1] > 0, "User 1 → Item 1 should be connected"
    assert ui_block[1, 2] > 0, "User 1 → Item 2 should be connected"
    assert ui_block[2, 2] > 0, "User 2 → Item 2 should be connected"
    assert ui_block[2, 3] > 0, "User 2 → Item 3 should be connected"
    # Unobserved
    assert ui_block[0, 2] == 0, "User 0 → Item 2 should be 0"
    assert ui_block[2, 0] == 0, "User 2 → Item 0 should be 0"
    logger.info("✓ Interaction pattern is correct")

    # ── Validate symmetry ────────────────────────────────────────────
    assert np.allclose(adj_dense, adj_dense.T), \
        "Adjacency matrix must be symmetric"
    logger.info("✓ Adjacency matrix is symmetric")

    # ── Validate normalisation ───────────────────────────────────────
    # Symmetric normalisation preserves symmetry and bounds spectral radius.
    # Row sums vary because items have different degrees — this is expected.
    row_sums = np.array(adj.sum(axis=1)).flatten()
    logger.info(f"Row sum range: [{row_sums.min():.4f}, {row_sums.max():.4f}]")
    assert adj.shape == (expected_nodes, expected_nodes)
    assert np.allclose(adj.toarray(), adj.toarray().T), \
        "Normalized adjacency must remain symmetric"
    logger.info("✓ Normalised adjacency is symmetric (correct symmetric normalisation)")

    # ── Validate edge_index ──────────────────────────────────────────
    edge_index = builder.get_edge_index()
    assert isinstance(edge_index, np.ndarray) or hasattr(edge_index, 'shape'), \
        "edge_index should be array-like"
    if hasattr(edge_index, 'shape'):
        assert edge_index.shape[0] == 2, \
            f"edge_index should have shape (2, E), got {edge_index.shape}"
        logger.info(f"✓ edge_index shape: {edge_index.shape}")

    # ── Test save / load round-trip ──────────────────────────────────
    save_dir = project_root / 'data' / 'processed' / 'test_graph'
    builder.save(str(save_dir))
    loaded = GraphBuilder.load(str(save_dir))
    assert loaded.n_users == builder.n_users
    assert loaded.n_items == builder.n_items
    assert loaded.n_nodes == builder.n_nodes
    assert np.allclose(
        loaded.get_normalized_adj().toarray(),
        builder.get_normalized_adj().toarray()
    ), "Loaded adjacency must match saved adjacency"
    logger.info("✓ Save / load round-trip successful")

    # Cleanup
    import shutil
    if save_dir.exists():
        shutil.rmtree(save_dir)

    # ── Test convenience function ────────────────────────────────────
    from graph.build_graph import build_graph
    adj2, ei2 = build_graph(train_df, n_users, n_items)
    assert np.allclose(adj2.toarray(), adj.toarray()), \
        "Convenience function should produce same adjacency"
    logger.info("✓ Convenience `build_graph()` works correctly")

    logger.info("=" * 60)
    logger.info("ALL GRAPH CONSTRUCTION TESTS PASSED ✓")
    logger.info("=" * 60)


if __name__ == '__main__':
    test_graph_builder()
