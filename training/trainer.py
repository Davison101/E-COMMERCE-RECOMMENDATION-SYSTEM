"""
Training module for LightGCN Recommender System.

Implements:
    - BPR (Bayesian Personalized Ranking) loss
    - Mini-batch negative sampling
    - Validation loop at configurable intervals
    - Early stopping
    - Model checkpointing
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import logging

from models.lightgcn import LightGCN
from graph.build_graph import GraphBuilder
from utils.config import get_config

logger = logging.getLogger(__name__)


def csr_to_sparse_tensor(adj_matrix) -> torch.sparse.Tensor:
    """Convert a scipy CSR adjacency matrix to a PyTorch sparse COO tensor.

    The resulting tensor is coalesced and moved to ``device``.
    """
    import scipy.sparse as sp
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)
    adj_coo = adj_matrix.tocoo()
    indices = np.vstack([adj_coo.row, adj_coo.col])
    indices = torch.tensor(indices, dtype=torch.long)
    values = torch.tensor(adj_coo.data, dtype=torch.float32)
    size = adj_coo.shape
    return torch.sparse_coo_tensor(indices, values, size).coalesce()


# ══════════════════════════════════════════════════════════════════════
# BPR Loss
# ══════════════════════════════════════════════════════════════════════


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking pairwise loss.

    ``L = - ∑ ln σ(ŷ_ui - ŷ_uj) + λ · ‖Θ‖²``

    where *i* is a positive item, *j* a negative item, and *λ* an L2
    regularisation coefficient.
    """

    def __init__(self, lambda_reg: float = 1e-5):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        pos_scores: Tensor,          # [batch_size]
        neg_scores: Tensor,          # [batch_size]
        user_emb: Tensor,            # [batch_size, dim]
        pos_item_emb: Tensor,        # [batch_size, dim]
        neg_item_emb: Tensor,        # [batch_size, dim]
    ) -> Tensor:
        """Compute BPR loss.

        Returns:
            Scalar loss tensor.
        """
        # BPR pairwise loss
        bpr_out = pos_scores - neg_scores                    # [batch_size]
        bpr_loss = -torch.log(torch.sigmoid(bpr_out) + 1e-8)  # [batch_size]
        bpr_loss = bpr_loss.mean()

        # L2 regularisation on trainable embeddings
        l2_loss = (
            user_emb.norm(2).pow(2)
            + pos_item_emb.norm(2).pow(2)
            + neg_item_emb.norm(2).pow(2)
        ) / user_emb.size(0)
        l2_loss = self.lambda_reg * l2_loss

        total = bpr_loss + l2_loss
        return total


# ══════════════════════════════════════════════════════════════════════
# Negative Sampling
# ══════════════════════════════════════════════════════════════════════


class NegativeSampler:
    """Samples negative items not interacted with by each user."""

    def __init__(
        self,
        n_items: int,
        train_interactions: np.ndarray,   # [n_train, 2] with (user, item) pairs
        num_negatives: int = 1,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        # Build set of positive items per user for fast lookup
        self.pos_items: Dict[int, np.ndarray] = {}
        for u, i in train_interactions:
            self.pos_items.setdefault(int(u), []).append(int(i))
        self.pos_items = {u: np.array(items) for u, items in self.pos_items.items()}

        logger.info(
            f"NegativeSampler(n_items={n_items}, num_neg={num_negatives})"
        )

    def sample(self, users: np.ndarray) -> np.ndarray:
        """For each user in ``users``, sample ``num_negatives`` negative items.

        Returns:
            Array of shape ``[len(users) * num_negatives]`` with item indices.
        """
        neg_items = []
        for u in users:
            u_int = int(u)
            pos = self.pos_items.get(u_int, np.array([], dtype=int))
            # Items that are NOT positive for this user
            candidates = np.setdiff1d(np.arange(self.n_items), pos, assume_unique=False)
            if len(candidates) < self.num_negatives:
                # Fallback (should rarely happen)
                neg = self.rng.integers(0, self.n_items, size=self.num_negatives)
            else:
                neg = self.rng.choice(candidates, size=self.num_negatives, replace=False)
            neg_items.extend(neg)

        return np.array(neg_items, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════


class Trainer:
    """LightGCN training loop with validation, early stopping, and checkpointing.

    Args:
        model: The LightGCN model to train.
        train_df: Training interactions DataFrame (must have ``user_idx`` and
            ``item_idx`` columns).
        val_df: Validation interactions DataFrame.
        test_df: Test interactions DataFrame.
        n_users: Total number of users.
        n_items: Total number of items.
        device: Device string (``'cpu'``, ``'cuda'``, or ``'auto'``).
        config: Override configuration. Falls back to ``config/settings.yaml``.
    """

    def __init__(
        self,
        model: LightGCN,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_users: int,
        n_items: int,
        device: str = 'auto',
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.n_users = n_users
        self.n_items = n_items

        # Config
        cfg = config or get_config().training
        self.epochs: int = cfg.get('epochs', 100)
        self.batch_size: int = cfg.get('batch_size', 2048)
        self.learning_rate: float = cfg.get('learning_rate', 0.001)
        self.weight_decay: float = cfg.get('weight_decay', 1e-5)
        self.early_stop_patience: int = cfg.get('early_stopping_patience', 10)
        self.eval_interval: int = cfg.get('eval_interval', 5)
        self.neg_sample_ratio: int = cfg.get('neg_sample_ratio', 1)
        self.save_dir: str = cfg.get('save_dir', 'saved_models')

        # Device
        self.device = self._resolve_device(device)

        # Graph — build from training data only
        logger.info("Building graph from training data...")
        graph_builder = GraphBuilder()
        graph_builder.build_from_dataframe(train_df, n_users, n_items)
        self.adj_norm = csr_to_sparse_tensor(
            graph_builder.get_normalized_adj()
        ).to(self.device)
        logger.info(f"Graph adjacency on {self.adj_norm.device}")

        # Optimiser & loss
        self.optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,  # L2 handled in BPRLoss
        )
        self.criterion = BPRLoss(lambda_reg=self.weight_decay).to(self.device)

        # Negative sampler
        train_pairs = train_df[['user_idx', 'item_idx']].values.astype(np.int64)
        self.neg_sampler = NegativeSampler(
            n_items=n_items,
            train_interactions=train_pairs,
            num_negatives=self.neg_sample_ratio,
        )

        # Move model
        self.model = self.model.to(self.device)

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'epoch': [],
        }

        logger.info(
            f"Trainer ready: epochs={self.epochs}, batch_size={self.batch_size}, "
            f"lr={self.learning_rate}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> LightGCN:
        """Run the full training loop.

        Returns:
            Trained model (best checkpoint loaded).
        """
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info("=" * 60)

        n_train = len(self.train_df)
        n_batches = max(1, n_train // self.batch_size)

        epoch_bar = tqdm(range(1, self.epochs + 1), desc='Epoch', unit='ep')

        for epoch in epoch_bar:
            t_start = time.perf_counter()

            # ── Training step ────────────────────────────────────────
            train_loss = self._train_epoch(n_batches)

            # ── Validation ───────────────────────────────────────────
            val_loss = None
            if epoch % self.eval_interval == 0 or epoch == 1:
                val_loss = self._validate()
                self.history['val_loss'].append(val_loss)
                self.history['epoch'].append(epoch)

                # Checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_loss)
                else:
                    self.patience_counter += 1

            # ── Logging ──────────────────────────────────────────────
            self.history['train_loss'].append(train_loss)
            elapsed = time.perf_counter() - t_start

            postfix = {
                'loss': f'{train_loss:.4f}',
                'time': f'{elapsed:.1f}s',
            }
            if val_loss is not None:
                postfix['val'] = f'{val_loss:.4f}'
            epoch_bar.set_postfix(postfix)

            # ── Early stopping ───────────────────────────────────────
            if self.patience_counter >= self.early_stop_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {self.early_stop_patience} checks)"
                )
                break

        # Restore best model
        best_path = Path(self.save_dir) / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Restored best model from epoch {checkpoint['epoch']} "
                f"(val_loss={checkpoint['val_loss']:.4f})"
            )

        self._save_history()
        logger.info("Training complete")
        return self.model

    # ------------------------------------------------------------------
    # Internal training step
    # ------------------------------------------------------------------

    def _train_epoch(self, n_batches: int) -> float:
        """Run one training epoch over the data.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_processed = 0

        # Shuffle training edges
        indices = torch.randperm(len(self.train_df))
        edge_users = torch.tensor(
            self.train_df['user_idx'].values, dtype=torch.long
        )[indices]
        edge_items = torch.tensor(
            self.train_df['item_idx'].values, dtype=torch.long
        )[indices]

        # We use edge_users and edge_items directly for positive pairs
        # and sample negatives separately
        n = len(edge_users)

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_users = edge_users[batch_start:batch_end]
            batch_pos_items = edge_items[batch_start:batch_end]
            batch_size_actual = len(batch_users)

            # Sample negative items
            neg_items_np = self.neg_sampler.sample(batch_users.numpy())
            batch_neg_items = torch.tensor(neg_items_np[:batch_size_actual], dtype=torch.long)

            # Move to device
            batch_users = batch_users.to(self.device)
            batch_pos_items = batch_pos_items.to(self.device)
            batch_neg_items = batch_neg_items.to(self.device)

            # Forward
            pos_scores, user_emb, pos_item_emb = self.model(
                self.adj_norm, batch_users, batch_pos_items
            )
            neg_scores, _, neg_item_emb = self.model(
                self.adj_norm, batch_users, batch_neg_items
            )

            # Loss
            loss = self.criterion(
                pos_scores, neg_scores,
                user_emb, pos_item_emb, neg_item_emb,
            )

            # Backward
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item() * batch_size_actual
            n_processed += batch_size_actual

        return total_loss / n_processed

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> float:
        """Compute validation loss.

        Uses a subset of validation edges for efficiency.

        Returns:
            Scalar validation loss.
        """
        self.model.eval()

        # Use up to 10k validation edges for efficiency
        val_size = min(len(self.val_df), 10000)
        val_subset = self.val_df.sample(n=val_size, random_state=42)

        edge_users = torch.tensor(val_subset['user_idx'].values, dtype=torch.long)
        edge_items = torch.tensor(val_subset['item_idx'].values, dtype=torch.long)

        total_loss = 0.0
        n = len(edge_users)

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_users = edge_users[batch_start:batch_end].to(self.device)
            batch_pos_items = edge_items[batch_start:batch_end].to(self.device)

            # Sample negatives
            neg_np = self.neg_sampler.sample(batch_users.cpu().numpy())
            batch_neg_items = torch.tensor(
                neg_np[:batch_end - batch_start], dtype=torch.long
            ).to(self.device)

            pos_scores, user_emb, pos_item_emb = self.model(
                self.adj_norm, batch_users, batch_pos_items
            )
            neg_scores, _, neg_item_emb = self.model(
                self.adj_norm, batch_users, batch_neg_items
            )

            loss = self.criterion(
                pos_scores, neg_scores,
                user_emb, pos_item_emb, neg_item_emb,
            )
            total_loss += loss.item() * (batch_end - batch_start)

        return total_loss / n

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / 'best_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'val_loss': val_loss,
            'config': {
                'n_users': self.n_users,
                'n_items': self.n_items,
                'embedding_dim': self.model.embedding_dim,
                'n_layers': self.model.n_layers,
            },
        }, path)
        logger.info(f"Checkpoint saved (epoch {epoch}, val_loss={val_loss:.4f})")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = Path(self.save_dir) / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {path}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)


# ══════════════════════════════════════════════════════════════════════
# Convenience function
# ══════════════════════════════════════════════════════════════════════


def train_model(
    model: LightGCN,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    config: Optional[Dict[str, Any]] = None,
) -> LightGCN:
    """One-shot convenience: instantiate a ``Trainer`` and run training.

    Returns:
        Trained model.
    """
    trainer = Trainer(
        model=model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_users=n_users,
        n_items=n_items,
        config=config,
    )
    return trainer.train()
