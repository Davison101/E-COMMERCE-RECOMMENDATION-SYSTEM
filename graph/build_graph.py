"""
Graph construction module for LightGCN Recommender System.

Converts user-item interaction data into a normalized bipartite adjacency
matrix that drives LightGCN's message-passing mechanism.

The graph structure follows the LightGCN paper (He et al., 2020):
    - Nodes: users + items (bipartite)
    - Edges: observed interactions (training only)
    - Normalization: symmetric (D⁻¹/² @ A @ D⁻¹/²)
"""

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import pickle

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, save_npz, load_npz
import torch
import logging

from utils.config import get_config

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a normalized bipartite adjacency matrix for LightGCN.

    The graph connects users (indices 0 .. n_users-1) and items
    (indices n_users .. n_users+n_items-1) via observed interactions.

    Example layout (n_users=3, n_items=2)::

        [0,0]  [0,1]  [0,2]  |  [0,3]  [0,4]      ← user rows
        [1,0]  [1,1]  [1,2]  |  [1,3]  [1,4]
        [2,0]  [2,1]  [2,2]  |  [2,3]  [2,4]
        ──────────────────────┼──────────────────
        [3,0]  [3,1]  [3,2]  |  [3,3]  [3,4]      ← item rows
        [4,0]  [4,1]  [4,2]  |  [4,3]  [4,4]

        \\____  user block ____/  \\_ item block _/
                 (zero)               (zero)

    Non-zero off-diagonal blocks encode user ↔ item edges.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Configuration dictionary. Falls back to global config.
        """
        cfg = config or get_config().graph
        self.normalization: str = cfg.get('normalization', 'symmetric')
        self.add_self_loops: bool = cfg.get('add_self_loops', False)

        # State set during build
        self.n_users: int = 0
        self.n_items: int = 0
        self.n_nodes: int = 0
        self.adj_matrix: Optional[csr_matrix] = None
        self.normalized_adj: Optional[csr_matrix] = None
        self.edge_index: Optional[torch.Tensor] = None

        logger.info(
            "GraphBuilder initialized "
            f"(normalization={self.normalization}, "
            f"self_loops={self.add_self_loops})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_dataframe(
        self,
        df,
        n_users: int,
        n_items: int,
        user_col: str = 'user_idx',
        item_col: str = 'item_idx',
        value_col: Optional[str] = None,
    ) -> 'GraphBuilder':
        """Build the adjacency matrix from a DataFrame of training interactions.

        Args:
            df: DataFrame with at least ``user_col`` and ``item_col`` columns.
            n_users: Number of unique users.
            n_items: Number of unique items.
            user_col: Column holding zero-indexed user indices.
            item_col: Column holding zero-indexed item indices.
            value_col: Optional column for edge weights (uses 1 if ``None``).

        Returns:
            Self for chaining.
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items

        users = df[user_col].values
        items = df[item_col].values + n_users  # offset items after users

        values = (
            np.ones(len(users), dtype=np.float32)
            if value_col is None
            else df[value_col].values.astype(np.float32)
        )

        # Symmetric — edges go both ways
        row = np.concatenate([users, items])
        col = np.concatenate([items, users])
        data = np.concatenate([values, values])

        self.adj_matrix = coo_matrix(
            (data, (row, col)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()

        if self.add_self_loops:
            self.adj_matrix = self.adj_matrix + eye(self.n_nodes, dtype=np.float32)

        self.normalized_adj = self._normalize(self.adj_matrix)
        self.edge_index = self._to_edge_index(self.normalized_adj)

        n_edges = self.adj_matrix.nnz // 2  # undirected
        logger.info(
            f"Graph built: {n_users} users + {n_items} items = {self.n_nodes} nodes, "
            f"{n_edges} interaction edges"
        )
        logger.debug(f"Adjacency matrix shape={self.adj_matrix.shape}, nnz={self.adj_matrix.nnz}")

        return self

    def build_from_arrays(
        self,
        users: np.ndarray,
        items: np.ndarray,
        n_users: int,
        n_items: int,
        values: Optional[np.ndarray] = None,
    ) -> 'GraphBuilder':
        """Build the adjacency matrix from numpy arrays (alternative to DataFrame).

        Args:
            users: 1-D array of zero-indexed user indices.
            items: 1-D array of zero-indexed item indices.
            n_users: Number of unique users.
            n_items: Number of unique items.
            values: Optional edge weights (uses 1 if ``None``).

        Returns:
            Self for chaining.
        """
        import pandas as pd
        df = pd.DataFrame({
            'user_idx': users,
            'item_idx': items,
            '_value': values if values is not None else np.ones_like(users),
        })
        return self.build_from_dataframe(
            df, n_users, n_items,
            value_col='_value' if values is not None else None,
        )

    def get_adj_matrix(self) -> csr_matrix:
        """Return the raw adjacency matrix (before normalization)."""
        if self.adj_matrix is None:
            raise RuntimeError("Graph not built yet. Call build_from_dataframe() first.")
        return self.adj_matrix

    def get_normalized_adj(self) -> csr_matrix:
        """Return the normalized adjacency matrix used by LightGCN."""
        if self.normalized_adj is None:
            raise RuntimeError("Graph not built yet. Call build_from_dataframe() first.")
        return self.normalized_adj

    def get_edge_index(self) -> torch.Tensor:
        """Return the edge index tensor (2 × n_edges) for PyTorch Geometric."""
        if self.edge_index is None:
            raise RuntimeError("Graph not built yet. Call build_from_dataframe() first.")
        return self.edge_index

    def get_torch_adj(self) -> torch.Tensor:
        """Return the normalized adjacency matrix as a dense PyTorch tensor.

        Note: For large graphs use sparse operations with ``edge_index`` instead.
        """
        if self.normalized_adj is None:
            raise RuntimeError("Graph not built yet. Call build_from_dataframe() first.")
        return torch.from_numpy(self.normalized_adj.toarray()).float()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str, name: str = 'graph') -> None:
        """Save graph artefacts to disk.

        Writes:
            ``{directory}/{name}_adj.npz``       — normalized adjacency
            ``{directory}/{name}_meta.pkl``       — metadata dict
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        adj_path = save_dir / f'{name}_adj.npz'
        meta_path = save_dir / f'{name}_meta.pkl'

        if self.normalized_adj is not None:
            save_npz(str(adj_path), self.normalized_adj)
        else:
            raise RuntimeError("Cannot save graph — not built yet.")

        meta = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_nodes': self.n_nodes,
            'normalization': self.normalization,
            'add_self_loops': self.add_self_loops,
        }
        with open(meta_path, 'wb') as fh:
            pickle.dump(meta, fh)

        logger.info(f"Graph saved to {save_dir}")

    @classmethod
    def load(cls, directory: str, name: str = 'graph') -> 'GraphBuilder':
        """Load a previously saved graph.

        Args:
            directory: Directory containing saved graph files.
            name: Base name used during ``save()``.

        Returns:
            Populated ``GraphBuilder`` instance.
        """
        load_dir = Path(directory)
        adj_path = load_dir / f'{name}_adj.npz'
        meta_path = load_dir / f'{name}_meta.pkl'

        if not adj_path.exists():
            raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path, 'rb') as fh:
            meta = pickle.load(fh)

        builder = cls()
        builder.n_users = meta['n_users']
        builder.n_items = meta['n_items']
        builder.n_nodes = meta['n_nodes']
        builder.normalization = meta['normalization']
        builder.add_self_loops = meta['add_self_loops']
        builder.normalized_adj = load_npz(str(adj_path))
        builder.adj_matrix = None         # not serialised (recoverable)
        builder.edge_index = builder._to_edge_index(builder.normalized_adj)

        logger.info(f"Graph loaded from {load_dir}")
        return builder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, adj: csr_matrix) -> csr_matrix:
        """Apply symmetric (LightGCN) or random-walk normalisation.

        Symmetric normalisation (default):  D^{-1/2} @ A @ D^{-1/2}
        """
        if self.normalization == 'symmetric':
            return self._symmetric_normalize(adj)
        elif self.normalization == 'random_walk':
            return self._random_walk_normalize(adj)
        else:
            raise ValueError(f"Unknown normalisation: '{self.normalization}'")

    @staticmethod
    def _symmetric_normalize(adj: csr_matrix) -> csr_matrix:
        """D^{-1/2} @ A @ D^{-1/2}."""
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(rowsum)
        np.power(rowsum, -0.5, where=rowsum > 0, out=d_inv_sqrt)
        d_mat = diags(d_inv_sqrt)
        return d_mat @ adj @ d_mat

    @staticmethod
    def _random_walk_normalize(adj: csr_matrix) -> csr_matrix:
        """D^{-1} @ A."""
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -1.0, where=rowsum > 0)
        d_inv[rowsum == 0] = 0.0
        d_mat = diags(d_inv)
        return d_mat @ adj

    @staticmethod
    def _to_edge_index(adj: csr_matrix) -> torch.Tensor:
        """Convert a sparse CSR matrix to a PyTorch edge_index tensor (2 × E)."""
        adj_coo = adj.tocoo()
        edge_index = np.stack([adj_coo.row, adj_coo.col], axis=0)
        return torch.from_numpy(edge_index).long()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_edges(self) -> int:
        """Number of undirected interaction edges."""
        if self.adj_matrix is None:
            return 0
        return self.adj_matrix.nnz // 2

    def __repr__(self) -> str:
        return (
            f"GraphBuilder(n_users={self.n_users}, n_items={self.n_items}, "
            f"n_nodes={self.n_nodes}, edges={self.num_edges})"
        )


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------

def build_graph(
    df,
    n_users: int,
    n_items: int,
    user_col: str = 'user_idx',
    item_col: str = 'item_idx',
) -> Tuple[csr_matrix, torch.Tensor]:
    """One-shot convenience: build normalised adjacency + edge index.

    Args:
        df: DataFrame of training interactions.
        n_users: Number of unique users.
        n_items: Number of unique items.
        user_col: User index column.
        item_col: Item index column.

    Returns:
        Tuple ``(normalized_adj_matrix, edge_index_tensor)``.
    """
    builder = GraphBuilder()
    builder.build_from_dataframe(df, n_users, n_items, user_col, item_col)
    return builder.get_normalized_adj(), builder.get_edge_index()
