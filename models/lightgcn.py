"""
LightGCN model implementation.

Reference:
    He et al., "LightGCN: Simplifying and Powering Graph Convolution Network
    for Recommendation", SIGIR 2020.

Key simplifications vs. standard GCN:
    - No feature transformation matrices (W₁, W₂)
    - No non-linear activation functions (ReLU)
    - No self-loops
    - Only linear embedding propagation across the bipartite graph
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class LightGCN(nn.Module):
    """Light Graph Convolution Network for collaborative filtering.

    The model learns user and item embeddings by propagating them across
    the user-item interaction graph using normalised adjacency convolution.

    Args:
        n_users: Number of unique users.
        n_items: Number of unique items.
        embedding_dim: Dimensionality of embeddings (default 64).
        n_layers: Number of propagation layers (default 3).
        alpha: Layer-weighting scheme. One of:
            - ``'mean'``:  αₖ = 1 / (n_layers + 1)  (all layers equal)
            - ``'concat'``: Use concatenation of all layers
            (default ``'mean'``)
        dropout: Dropout probability applied to input embeddings (default 0.0).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        alpha: str = 'mean',
        dropout: float = 0.0,
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if alpha not in ('mean', 'concat'):
            raise ValueError("alpha must be 'mean' or 'concat'")

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.dropout = dropout

        # ── Embeddings (initial layer-0 representations) ────────────
        init_scale = (1.0 / embedding_dim) ** 0.5
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=init_scale)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.item_embedding.weight, std=init_scale)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Layer combination weights (fixed equal weights for mean)
        if alpha == 'mean':
            self.layer_weight = 1.0 / (n_layers + 1)
        self.layer_weights = None  # not used; kept for future learnable-alpha

        logger.info(
            f"LightGCN(n_users={n_users}, n_items={n_items}, "
            f"dim={embedding_dim}, layers={n_layers}, alpha={alpha})"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, adj_norm: torch.sparse.Tensor, users: Tensor, items: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass: propagate embeddings and return predictions.

        Args:
            adj_norm: Normalised adjacency matrix (sparse, n_nodes × n_nodes).
            users: User indices  [batch_size].
            items: Item indices  [batch_size].

        Returns:
            Tuple of
                preds:      Predicted scores           [batch_size].
                user_embs:  Final user embeddings      [batch_size, dim].
                item_embs:  Final item embeddings      [batch_size, dim].
        """
        all_user_embs, all_item_embs = self.propagate(adj_norm)

        # Layer combination (mean)
        user_final = torch.stack(all_user_embs, dim=0).sum(dim=0)
        item_final = torch.stack(all_item_embs, dim=0).sum(dim=0)

        # Gather for batch
        user_out = user_final[users]   # [batch_size, dim]
        item_out = item_final[items]   # [batch_size, dim]

        # Inner product → predicted score
        preds = (user_out * item_out).sum(dim=-1)

        return preds, user_out, item_out

    def propagate(
        self, adj_norm: torch.sparse.Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Propagate embeddings through the graph for all nodes.

        Args:
            adj_norm: Normalised sparse adjacency matrix.

        Returns:
            Tuple of (list_of_user_embs, list_of_item_embs).
            Each list has ``n_layers + 1`` entries (layer 0 .. layer K).
        """
        # Concatenate user & item embeddings → full node embeddings
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )  # [n_users + n_items, dim]

        ego_embeddings = self.dropout_layer(ego_embeddings)

        all_embeddings = [ego_embeddings]
        current = ego_embeddings

        for _ in range(self.n_layers):
            # Sparse matrix multiplication: E^{(k+1)} = A_norm @ E^{(k)}
            current = torch.sparse.mm(adj_norm, current)
            all_embeddings.append(current)

        # Split back into user and item parts
        user_embs_per_layer: List[Tensor] = []
        item_embs_per_layer: List[Tensor] = []

        for emb in all_embeddings:
            user_part, item_part = emb[:self.n_users], emb[self.n_users:]
            user_embs_per_layer.append(user_part)
            item_embs_per_layer.append(item_part)

        return user_embs_per_layer, item_embs_per_layer

    # ------------------------------------------------------------------
    # Full prediction matrix (for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_all_users(
        self, adj_norm: torch.sparse.Tensor
    ) -> Tensor:
        """Compute predicted scores for *all* user-item pairs.

        Returns shape ``[n_users, n_items]``.
        """
        all_user_embs, all_item_embs = self.propagate(adj_norm)

        user_final = torch.stack(all_user_embs, dim=0).sum(dim=0)
        item_final = torch.stack(all_item_embs, dim=0).sum(dim=0)

        return user_final @ item_final.T  # [n_users, n_items]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, map_location: Optional[str] = None) -> 'LightGCN':
        """Load model weights from disk."""
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)
        logger.info(f"Model loaded from {path}")
        return self

    def extra_repr(self) -> str:
        return (
            f"n_users={self.n_users}, n_items={self.n_items}, "
            f"dim={self.embedding_dim}, layers={self.n_layers}, "
            f"alpha={self.alpha}, dropout={self.dropout}"
        )
