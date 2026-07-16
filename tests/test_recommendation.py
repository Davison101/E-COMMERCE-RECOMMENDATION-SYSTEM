"""
Tests for recommendation engine and evaluation metrics.

Verifies:
    - Metric functions produce correct values for known inputs
    - Evaluator runs end-to-end on a trained model
    - Recommender generates valid Top-N lists
    - Seen items are correctly excluded
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    Evaluator,
    evaluate_model,
)
from recommendation.recommend import Recommender, load_and_recommend


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


def _build_minimal_model(n_users=10, n_items=20, dim=16, layers=2):
    """Create a minimal untrained LightGCN for testing shapes/logic."""
    from models.lightgcn import LightGCN
    model = LightGCN(
        n_users=n_users, n_items=n_items,
        embedding_dim=dim, n_layers=layers,
    )
    return model


def _build_random_adj(n_users=10, n_items=20, density=0.15):
    """Build a random bipartite adjacency edge_index."""
    from graph.build_graph import GraphBuilder
    rng = np.random.RandomState(42)
    n_interactions = max(1, int(n_users * n_items * density))

    users = rng.randint(0, n_users, size=n_interactions)
    items = rng.randint(0, n_items, size=n_interactions)

    builder = GraphBuilder()
    builder.build_from_arrays(
        users=users,
        items=items,
        n_users=n_users,
        n_items=n_items,
    )
    adj_norm = builder.get_torch_adj()
    train_user_items = {}
    for u, i in zip(users, items):
        train_user_items.setdefault(int(u), []).append(int(i))
    return adj_norm, train_user_items, builder


def _build_test_items(n_users=10) -> Dict[int, List[int]]:
    """Create test interactions (each user has 1-3 held-out items)."""
    rng = np.random.RandomState(123)
    test_items: Dict[int, List[int]] = {}
    for u in range(n_users):
        k = rng.randint(1, 4)
        test_items[u] = rng.randint(0, 20, size=k).tolist()
    return test_items


# ══════════════════════════════════════════════════════════════════════
# Unit tests: Metric functions
# ══════════════════════════════════════════════════════════════════════


class TestRecall:
    def test_perfect_recall(self):
        """All relevant items in top-K → recall = 1.0."""
        rec = [0, 1, 2, 3, 4]
        gt = {1, 3}
        assert recall_at_k(rec, gt, k=5) == 1.0

    def test_zero_recall(self):
        """No relevant items → recall = 0.0."""
        rec = [5, 6, 7, 8, 9]
        gt = {1, 2}
        assert recall_at_k(rec, gt, k=5) == 0.0

    def test_partial_recall(self):
        """Half of relevant items retrieved."""
        rec = [1, 10, 20]
        gt = {1, 2}
        assert recall_at_k(rec, gt, k=3) == 0.5

    def test_empty_ground_truth(self):
        """No ground truth → recall = 0.0."""
        assert recall_at_k([0, 1], set(), k=5) == 0.0


class TestPrecision:
    def test_perfect_precision(self):
        """All retrieved items are relevant."""
        rec = [1, 3, 5]
        gt = {1, 3, 5, 7}
        assert precision_at_k(rec, gt, k=3) == 1.0

    def test_zero_precision(self):
        """None retrieved are relevant."""
        rec = [2, 4, 6]
        gt = {1, 3}
        assert precision_at_k(rec, gt, k=3) == 0.0

    def test_partial_precision(self):
        """2 of 5 retrieved are relevant."""
        rec = [1, 10, 3, 20, 5]
        gt = {1, 3, 7}
        assert precision_at_k(rec, gt, k=5) == 0.4


class TestNDCG:
    def test_perfect_ndcg(self):
        """All relevant items retrieved in optimal order."""
        rec = [1, 2, 3]
        gt = {1, 2, 3}
        assert ndcg_at_k(rec, gt, k=3) == 1.0

    def test_zero_ndcg(self):
        """No relevant items → NDCG = 0."""
        rec = [5, 6, 7]
        gt = {1, 2}
        assert ndcg_at_k(rec, gt, k=3) == 0.0

    def test_ndcg_prefers_early_positions(self):
        """Same set but better ordering should give higher NDCG."""
        gt = {1, 5}
        rec_good = [1, 5, 10]
        rec_bad = [10, 5, 1]
        ndcg_good = ndcg_at_k(rec_good, gt, k=3)
        ndcg_bad = ndcg_at_k(rec_bad, gt, k=3)
        assert ndcg_good > ndcg_bad

    def test_ndcg_exact_value(self):
        """Verify against a known computation."""
        rec = [0, 1, 2, 3, 4]
        gt = {1, 3, 5}
        # DCG = 0/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5) + 0/log2(6)
        #     = 0 + 1/1.585 + 0 + 1/2.322 + 0 ≈ 0.6309 + 0.4307 = 1.0616
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        #      = 1 + 0.6309 + 0.5 = 2.1309
        # NDCG = 1.0616 / 2.1309 ≈ 0.498
        val = ndcg_at_k(rec, gt, k=5)
        assert abs(val - 1.0616 / 2.1309) < 0.01


class TestHitRate:
    def test_hit(self):
        """Relevant item in top-K → hit = 1.0."""
        assert hit_rate_at_k([1, 2, 3], {3, 5}, k=3) == 1.0

    def test_no_hit(self):
        """No relevant item → hit = 0.0."""
        assert hit_rate_at_k([4, 5, 6], {1, 2}, k=3) == 0.0

    def test_hit_in_top_k_only(self):
        """Relevant exists but beyond K → not a hit."""
        assert hit_rate_at_k([4, 5, 6], {6}, k=2) == 0.0


# ══════════════════════════════════════════════════════════════════════
# Unit tests: Recommender
# ══════════════════════════════════════════════════════════════════════


class TestRecommender:
    @classmethod
    def setup_class(cls):
        cls.n_users = 10
        cls.n_items = 20
        cls.model = _build_minimal_model(cls.n_users, cls.n_items)
        cls.adj, cls.train_items, cls.builder = _build_random_adj(
            cls.n_users, cls.n_items
        )
        cls.recommender = Recommender(
            model=cls.model,
            adj_norm=cls.adj,
            n_users=cls.n_users,
            n_items=cls.n_items,
            train_user_items=cls.train_items,
        )

    def test_recommend_returns_correct_length(self):
        """Recommend returns exactly top_k items (or fewer if exclude_seen)."""
        recs = self.recommender.recommend(user_id=0, top_k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5
        assert all(isinstance(i, int) for i in recs)

    def test_recommend_excludes_seen_items(self):
        """Seen items should not appear in recommendations."""
        recs = self.recommender.recommend(user_id=0, top_k=10)
        seen = self.train_items.get(0, [])
        for r in recs:
            assert r not in seen, f"Seen item {r} appeared in recommendations"

    def test_recommend_different_users_different_results(self):
        """Two different users should get different recommendations."""
        recs_0 = self.recommender.recommend(user_id=0, top_k=5)
        recs_1 = self.recommender.recommend(user_id=1, top_k=5)
        assert recs_0 != recs_1

    def test_recommend_with_scores(self):
        """Returning scores produces a tuple."""
        result = self.recommender.recommend(user_id=0, top_k=5, return_scores=True)
        assert isinstance(result, tuple)
        items, scores = result
        assert len(items) == len(scores)
        # Scores should be descending
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_recommend_batch_returns_dict(self):
        """Batch recommend returns a dict keyed by user."""
        results = self.recommender.recommend_batch([0, 1, 2], top_k=3)
        assert isinstance(results, dict)
        assert set(results.keys()) == {0, 1, 2}

    def test_recommend_all_users(self):
        """recommend_all_users covers all users."""
        results = self.recommender.recommend_all_users(top_k=3)
        assert len(results) == self.n_users

    def test_invalid_user_raises(self):
        """Out-of-range user raises ValueError."""
        import re
        try:
            self.recommender.recommend(user_id=999)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_recommend_to_dict_serialisable(self):
        """Dict output should be JSON-serialisable."""
        d = self.recommender.recommend_to_dict([0, 1], top_k=3)
        json_str = json.dumps(d)
        assert json_str

    def test_recommend_from_raw_id(self):
        """Raw ID path works with encoder mappings."""
        user2id = {f'u{i}': i for i in range(self.n_users)}
        id2item = {i: f'i{i}' for i in range(self.n_items)}
        recs = self.recommender.recommend_from_raw_id(
            raw_user_id='u0',
            user2id=user2id,
            id2item=id2item,
            top_k=5,
        )
        assert isinstance(recs, list)
        assert all(isinstance(r, str) and r.startswith('i') for r in recs)


# ══════════════════════════════════════════════════════════════════════
# Integration test: Evaluator
# ══════════════════════════════════════════════════════════════════════


class TestEvaluator:
    @classmethod
    def setup_class(cls):
        cls.n_users = 10
        cls.n_items = 20
        cls.model = _build_minimal_model(cls.n_users, cls.n_items)
        cls.adj, cls.train_items, cls.builder = _build_random_adj(
            cls.n_users, cls.n_items
        )
        cls.test_items = _build_test_items(cls.n_users)

    def test_evaluator_returns_all_metrics(self):
        """Evaluator returns all expected metric keys."""
        evaluator = Evaluator(
            model=self.model,
            adj_norm=self.adj,
            n_users=self.n_users,
            n_items=self.n_items,
            train_user_items=self.train_items,
            test_user_items=self.test_items,
            k_values=[5, 10],
        )
        results = evaluator.evaluate()

        expected_keys = [
            'Recall@5', 'Precision@5', 'NDCG@5', 'HitRate@5',
            'Recall@10', 'Precision@10', 'NDCG@10', 'HitRate@10',
        ]
        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"
            assert isinstance(results[key], float)

    def test_metrics_in_valid_range(self):
        """All metrics should be in [0, 1]."""
        evaluator = Evaluator(
            model=self.model,
            adj_norm=self.adj,
            n_users=self.n_users,
            n_items=self.n_items,
            train_user_items=self.train_items,
            test_user_items=self.test_items,
            k_values=[5, 10, 20],
        )
        results = evaluator.evaluate()
        for key, val in results.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} not in [0, 1]"

    def test_evaluate_model_convenience(self):
        """evaluate_model convenience function returns same keys."""
        results = evaluate_model(
            model=self.model,
            adj_norm=self.adj,
            n_users=self.n_users,
            n_items=self.n_items,
            train_user_items=self.train_items,
            test_user_items=self.test_items,
            k_values=[5, 10],
        )
        assert 'Recall@5' in results
        assert 'NDCG@10' in results

    def test_save_results(self):
        """save_results writes a valid JSON file."""
        import tempfile
        evaluator = Evaluator(
            model=self.model,
            adj_norm=self.adj,
            n_users=self.n_users,
            n_items=self.n_items,
            train_user_items=self.train_items,
            test_user_items=self.test_items,
        )
        results = evaluator.evaluate()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = f.name
        evaluator.save_results(results, tmp_path)
        with open(tmp_path) as f:
            loaded = json.load(f)
        assert loaded == results
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
