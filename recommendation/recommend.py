"""
Recommendation engine for LightGCN.

Generates Top-N recommendations for a trained model.
Handles:
    - Scoring all candidate items for a user
    - Excluding items the user has already interacted with
    - Returning ranked lists with optional metadata
    - Batch recommendations across many users
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class Recommender:
    """Generates Top-N item recommendations using a trained LightGCN model.

    Args:
        model: Trained LightGCN model (in ``eval()`` mode).
        adj_norm: Normalised adjacency sparse tensor (same as used during
            training / evaluation).
        n_users: Total number of users.
        n_items: Total number of items.
        train_user_items: Mapping ``user_idx → list(item_idx)`` of
            interactions the user already had (will be excluded from
            recommendations).
        device: Torch device string.
        batch_size: Items are scored in batches when ``mode='batched'``.
        mode: ``'full'`` — score all items in one forward pass (fastest).
            ``'batched'`` — score in smaller batches (memory efficient).
    """

    def __init__(
        self,
        model: 'LightGCN',
        adj_norm: torch.sparse.Tensor,
        n_users: int,
        n_items: int,
        train_user_items: Dict[int, List[int]],
        device: str = 'cpu',
        batch_size: int = 512,
        mode: str = 'full',
    ):
        self.model = model
        self.adj_norm = adj_norm
        self.n_users = n_users
        self.n_items = n_items
        self.device = torch.device(device)

        # Build a set of seen items per user for fast exclusion
        self.seen_items: Dict[int, Set[int]] = {
            u: set(items) for u, items in train_user_items.items()
        }

        # Pre-compute full score matrix if using full mode
        self._score_matrix: Optional[Tensor] = None
        self._scores_computed = False

        logger.info(
            f"Recommender ready: {n_users} users × {n_items} items, "
            f"mode={mode}, batch_size={batch_size}"
        )

    # ── Public API ───────────────────────────────────────────────────

    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        return_scores: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """Get Top-N recommendations for a **single** user.

        Args:
            user_id: User index (0-based, encoded).
            top_k: Number of recommendations to return.
            exclude_seen: Whether to filter out already-interacted items.
            return_scores: If ``True``, also return the model scores.

        Returns:
            List of recommended item indices (ranked best first), or
            ``(items, scores)`` tuple if ``return_scores=True``.
        """
        self._ensure_user_valid(user_id)
        scores = self._get_user_scores(user_id)
        return self._rank(scores, user_id, top_k, exclude_seen, return_scores)

    @torch.no_grad()
    def recommend_batch(
        self,
        user_ids: List[int],
        top_k: int = 10,
        exclude_seen: bool = True,
        return_scores: bool = False,
    ) -> Dict[int, Union[List[int], Tuple[List[int], List[float]]]]:
        """Get Top-N recommendations for **multiple** users.

        Args:
            user_ids: List of user indices.
            top_k: Number of recommendations per user.
            exclude_seen: Whether to filter out already-interacted items.
            return_scores: If ``True``, also return scores.

        Returns:
            ``{user_id: items}`` or ``{user_id: (items, scores)}``.
        """
        if self._score_matrix is None:
            self._compute_score_matrix()

        results: Dict = {}
        for uid in user_ids:
            self._ensure_user_valid(uid)
            scores = self._score_matrix[uid].cpu().numpy()
            results[uid] = self._rank(scores, uid, top_k, exclude_seen, return_scores)
        return results

    @torch.no_grad()
    def recommend_all_users(
        self,
        top_k: int = 10,
        exclude_seen: bool = True,
        return_scores: bool = False,
    ) -> Dict[int, Union[List[int], Tuple[List[int], List[float]]]]:
        """Get Top-N recommendations for **every** user.

        Args:
            top_k: Number of recommendations per user.
            exclude_seen: Whether to filter out already-interacted items.
            return_scores: If ``True``, also return scores.

        Returns:
            ``{user_id: items}`` or ``{user_id: (items, scores)}``.
        """
        return self.recommend_batch(
            list(range(self.n_users)), top_k, exclude_seen, return_scores,
        )

    @torch.no_grad()
    def recommend_from_raw_id(
        self,
        raw_user_id: str,
        user2id: Dict[str, int],
        id2item: Dict[int, str],
        top_k: int = 10,
        exclude_seen: bool = True,
        return_scores: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """Recommend using **original (string) IDs** through the encoder mappings.

        Args:
            raw_user_id: Original user ID (before encoding).
            user2id: Mapping ``raw_user → encoded_idx`` from IDEncoder.
            id2item: Mapping ``encoded_item_idx → raw_item_id``.
            top_k: Number of recommendations.
            exclude_seen: Filter seen items.
            return_scores: Also return scores.

        Returns:
            List of original item IDs (strings), or
            ``(items, scores)`` tuple if ``return_scores=True``.
        """
        if raw_user_id not in user2id:
            raise KeyError(f"Unknown user '{raw_user_id}'")

        user_idx = user2id[raw_user_id]
        result = self.recommend(user_idx, top_k, exclude_seen, return_scores)

        if return_scores:
            items, scores = result
            raw_items = [id2item[i] for i in items]
            return raw_items, scores
        else:
            return [id2item[i] for i in result]

    # ── Internal helpers ─────────────────────────────────────────────

    def _compute_score_matrix(self) -> Tensor:
        """Compute and cache the full [n_users, n_items] score matrix."""
        logger.info("Computing full score matrix for recommendations...")
        self._score_matrix = self.model.predict_all_users(self.adj_norm)
        self._scores_computed = True
        logger.info("Score matrix computed.")
        return self._score_matrix

    def _ensure_user_valid(self, user_id: int) -> None:
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(
                f"User {user_id} out of range [0, {self.n_users - 1}]"
            )

    def _get_user_scores(self, user_id: int) -> np.ndarray:
        """Get scores for a single user (cached or computed on-demand)."""
        if self._score_matrix is not None:
            return self._score_matrix[user_id].cpu().numpy()

        # Compute full matrix on first access
        if not self._scores_computed:
            self._compute_score_matrix()
            return self._score_matrix[user_id].cpu().numpy()

        return self._score_matrix[user_id].cpu().numpy()

    def _rank(
        self,
        scores: np.ndarray,
        user_id: int,
        top_k: int,
        exclude_seen: bool,
        return_scores: bool,
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """Rank items for a user, returning top-k indices."""
        scores = scores.copy()

        if exclude_seen:
            seen = self.seen_items.get(user_id, set())
            if seen:
                scores[list(seen)] = -np.inf

        # Get top-k indices
        top_k = min(top_k, self.n_items)
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]

        # Sort these top-k by descending score
        top_scores = scores[top_indices]
        order = np.argsort(-top_scores)
        top_indices = top_indices[order]
        top_scores = top_scores[order]

        if return_scores:
            return top_indices.tolist(), top_scores.tolist()
        return top_indices.tolist()

    # ── Persistence ──────────────────────────────────────────────────

    def recommend_to_dict(
        self,
        user_ids: List[int],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[str, List[int]]:
        """Generate recommendations and return a plain dict (JSON-serialisable).

        Args:
            user_ids: List of user indices.
            top_k: Number of recommendations per user.
            exclude_seen: Filter seen items.

        Returns:
            ``{'user_0': [3, 5, ...], 'user_1': [2, 8, ...], ...}``
        """
        recs = self.recommend_batch(user_ids, top_k, exclude_seen)
        return {str(u): items for u, items in recs.items()}

    def save_recommendations(
        self,
        path: str,
        user_ids: Optional[List[int]] = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> None:
        """Generate and save recommendations to a JSON file.

        Args:
            path: Output JSON path.
            user_ids: Users to recommend for (default: all).
            top_k: Number of recommendations per user.
            exclude_seen: Filter seen items.
        """
        if user_ids is None:
            user_ids = list(range(self.n_users))

        data = self.recommend_to_dict(user_ids, top_k, exclude_seen)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Recommendations saved to {path} ({len(data)} users)")


# ══════════════════════════════════════════════════════════════════════
# Convenience function: full pipeline from saved model + graph
# ══════════════════════════════════════════════════════════════════════


def load_and_recommend(
    model_path: str,
    graph_path: str,
    user_id: int,
    top_k: int = 10,
    exclude_seen: bool = True,
) -> List[int]:
    """Load a saved model and graph, then recommend for one user.

    This is a quick-start convenience function. For production use, create
    a ``Recommender`` instance once and re-use it.

    Args:
        model_path: Path to ``best_model.pt`` checkpoint.
        graph_path: Path to ``graph.npz`` (GraphBuilder save).
        user_id: User index.
        top_k: Number of recommendations.
        exclude_seen: Filter seen items.

    Returns:
        List of recommended item indices.
    """
    from graph.build_graph import GraphBuilder
    from models.lightgcn import LightGCN

    builder = GraphBuilder()
    builder.load(graph_path)

    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('config', {})
    model = LightGCN(
        n_users=model_config.get('n_users', builder.n_users),
        n_items=model_config.get('n_items', builder.n_items),
        embedding_dim=model_config.get('embedding_dim', 64),
        n_layers=model_config.get('n_layers', 3),
        alpha=model_config.get('alpha', 'mean'),
        dropout=model_config.get('dropout', 0.0),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    adj_norm = builder.get_edge_index()  # edge_index for propagation

    # Build train_user_items from adjacency
    # (In practice you'd load this from the split data)
    train_user_items: Dict[int, List[int]] = {}
    edge_idx = builder.get_edge_index()
    for i in range(edge_idx.shape[1]):
        u, v = edge_idx[0, i].item(), edge_idx[1, i].item()
        if u < builder.n_users:  # user node
            train_user_items.setdefault(u, []).append(v - builder.n_users)

    recommender = Recommender(
        model=model,
        adj_norm=adj_norm,
        n_users=builder.n_users,
        n_items=builder.n_items,
        train_user_items=train_user_items,
    )
    return recommender.recommend(user_id, top_k, exclude_seen)
