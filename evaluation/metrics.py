"""
Evaluation metrics module for LightGCN Recommender System.

Implements standard top-N recommendation metrics:
    - Recall@K
    - Precision@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Hit Rate@K
"""

from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Metric functions
# ══════════════════════════════════════════════════════════════════════


def recall_at_k(
    recommended: List[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """Recall@K = |relevant ∩ retrieved| / |relevant|."""
    if not ground_truth:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def precision_at_k(
    recommended: List[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """Precision@K = |relevant ∩ retrieved| / K."""
    top_k = set(recommended[:k])
    hits = len(top_k & ground_truth)
    return hits / k


def ndcg_at_k(
    recommended: List[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """NDCG@K = DCG / IDCG.

    DCG  = ∑_{i=1}^{k} rel_i / log₂(i+1)
    IDCG = ∑_{i=1}^{min(k,|rel|)} 1 / log₂(i+1)

    rel_i = 1 if item at position i is relevant, else 0.
    """
    if not ground_truth:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log₂(1) = 0

    # IDCG (ideal: all relevant items at the top)
    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    recommended: List[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """Hit Rate@K = 1 if any relevant item in top-K, else 0."""
    top_k = set(recommended[:k])
    return 1.0 if (top_k & ground_truth) else 0.0


# ══════════════════════════════════════════════════════════════════════
# Evaluation runner
# ══════════════════════════════════════════════════════════════════════


class Evaluator:
    """Computes ranking metrics for a trained LightGCN model.

    For each user in the test set, the model scores **all** items, items
    the user already interacted with in the training set are masked out,
    and the remaining items are ranked. Metrics are averaged across users.

    Args:
        model: Trained LightGCN model (in evaluation mode).
        adj_norm: Normalised adjacency sparse tensor.
        n_users: Total number of users.
        n_items: Total number of items.
        train_user_items: Mapping ``user_idx → list(item_idx)`` of training
            interactions (used to exclude seen items from recommendations).
        val_user_items: Mapping for validation interactions.
        test_user_items: Mapping for test interactions.
        k_values: List of K values to evaluate (e.g., ``[5, 10, 20]``).
        device: Device for computation.
        batch_size: Items are scored in batches of this size when
            ``mode='batched'``.
        mode: ``'full'`` scores all items at once (fast, high memory);
            ``'batched'`` uses iterative scoring (lower memory).
    """

    def __init__(
        self,
        model: 'LightGCN',
        adj_norm: torch.sparse.Tensor,
        n_users: int,
        n_items: int,
        train_user_items: Dict[int, List[int]],
        val_user_items: Optional[Dict[int, List[int]]] = None,
        test_user_items: Optional[Dict[int, List[int]]] = None,
        k_values: Optional[List[int]] = None,
        device: str = 'cpu',
        batch_size: int = 512,
        mode: str = 'full',
    ):
        self.model = model
        self.adj_norm = adj_norm
        self.n_users = n_users
        self.n_items = n_items
        self.train_user_items = train_user_items
        self.val_user_items = val_user_items or {}
        self.test_user_items = test_user_items or {}
        self.k_values = k_values or [5, 10, 20]
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.mode = mode

        # Union of training + validation items (seen items to exclude)
        self.seen_items: Dict[int, Set[int]] = {}
        for u in range(n_users):
            seen = set(self.train_user_items.get(u, []))
            seen.update(self.val_user_items.get(u, []))
            self.seen_items[u] = seen

        logger.info(
            f"Evaluator: {n_users} users, {n_items} items, "
            f"K={self.k_values}, mode={mode}"
        )

    @torch.no_grad()
    def evaluate(
        self,
        user_subset: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Run evaluation and return averaged metrics.

        Args:
            user_subset: If provided, only evaluate these users.

        Returns:
            Dictionary like ``{'Recall@10': 0.123, 'NDCG@10': 0.456, ...}``.
        """
        self.model.eval()

        users = user_subset if user_subset is not None else list(range(self.n_users))
        n_eval_users = len(users)
        logger.info(f"Evaluating {n_eval_users} users...")

        # Get the full score matrix [n_users, n_items] or iterate by batches
        if self.mode == 'full':
            score_matrix = self._score_all()
        else:
            score_matrix = None  # batched iteration below

        # Accumulate metrics
        metric_sums: Dict[str, float] = {}
        for k in self.k_values:
            for metric in ['Recall', 'Precision', 'NDCG', 'HitRate']:
                metric_sums[f'{metric}@{k}'] = 0.0

        t_start = time.perf_counter()

        for user_idx in users:
            # Ground truth (test interactions)
            ground_truth = set(self.test_user_items.get(user_idx, []))
            if not ground_truth:
                continue

            # Scores for this user
            if self.mode == 'full':
                scores = score_matrix[user_idx].cpu().numpy()
            else:
                scores = self._score_user_batched(user_idx)

            # Mask seen items
            seen = self.seen_items.get(user_idx, set())
            scores[list(seen)] = -np.inf

            # Rank items by score (descending)
            ranked = np.argsort(-scores).tolist()

            # Compute metrics
            for k in self.k_values:
                metric_sums[f'Recall@{k}'] += recall_at_k(ranked, ground_truth, k)
                metric_sums[f'Precision@{k}'] += precision_at_k(ranked, ground_truth, k)
                metric_sums[f'NDCG@{k}'] += ndcg_at_k(ranked, ground_truth, k)
                metric_sums[f'HitRate@{k}'] += hit_rate_at_k(ranked, ground_truth, k)

        elapsed = time.perf_counter() - t_start
        logger.info(f"Evaluation took {elapsed:.1f}s ({elapsed / n_eval_users:.3f}s/user)")

        # Average
        results: Dict[str, float] = {}
        for key, total in metric_sums.items():
            results[key] = total / n_eval_users

        return results

    def _score_all(self) -> Tensor:
        """Compute the full [n_users, n_items] score matrix at once."""
        logger.info("Computing full score matrix...")
        t0 = time.perf_counter()
        scores = self.model.predict_all_users(self.adj_norm)
        elapsed = time.perf_counter() - t0
        logger.info(f"Full score matrix computed in {elapsed:.1f}s")
        return scores

    def _score_user_batched(self, user_idx: int) -> np.ndarray:
        """Score one user against all items in batches (memory efficient)."""
        user_t = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        scores = np.zeros(self.n_items, dtype=np.float32)

        for start in range(0, self.n_items, self.batch_size):
            end = min(start + self.batch_size, self.n_items)
            items_t = torch.arange(start, end, dtype=torch.long, device=self.device)

            preds, _, _ = self.model(self.adj_norm, user_t, items_t)
            scores[start:end] = preds.cpu().numpy()

        return scores

    def print_results(self, results: Dict[str, float]) -> None:
        """Pretty-print evaluation results."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        # Group by K
        for k in self.k_values:
            print(f"\n  Top-{k}:")
            for metric in ['Recall', 'Precision', 'NDCG', 'HitRate']:
                key = f'{metric}@{k}'
                val = results.get(key, 0.0)
                print(f"    {key:>12}: {val:.4f}")
        print("=" * 50 + "\n")

    def save_results(
        self, results: Dict[str, float], path: str
    ) -> None:
        """Save results to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {path}")


# ══════════════════════════════════════════════════════════════════════
# Module-level convenience
# ══════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: 'LightGCN',
    adj_norm: torch.sparse.Tensor,
    n_users: int,
    n_items: int,
    train_user_items: Dict[int, List[int]],
    test_user_items: Dict[int, List[int]],
    k_values: Optional[List[int]] = None,
    val_user_items: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, float]:
    """One-shot evaluation convenience function.

    Args:
        model: Trained LightGCN model.
        adj_norm: Normalised adjacency sparse tensor.
        n_users: Total users.
        n_items: Total items.
        train_user_items: Training interactions per user.
        test_user_items: Test interactions per user.
        k_values: K values to evaluate.
        val_user_items: Validation interactions per user (optional).

    Returns:
        Dictionary of metric results.
    """
    evaluator = Evaluator(
        model=model,
        adj_norm=adj_norm,
        n_users=n_users,
        n_items=n_items,
        train_user_items=train_user_items,
        val_user_items=val_user_items,
        test_user_items=test_user_items,
        k_values=k_values,
    )
    results = evaluator.evaluate()
    evaluator.print_results(results)
    return results
