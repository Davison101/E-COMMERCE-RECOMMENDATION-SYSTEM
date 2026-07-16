"""
FastAPI application for LightGCN Recommender System.

Exposes REST endpoints for:
    - Health check
    - Top-N recommendations (single user, batch, all users)
    - Model evaluation
    - Model / dataset information

Start with::

    uvicorn api.main:app --reload

Or from the project root::

    python -m uvicorn api.main:app --reload --port 8000
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager

from api.schemas import (
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    BatchRecommendResponse,
    EvaluateResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from utils.config import get_config
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Application state (loaded once at startup)
# ══════════════════════════════════════════════════════════════════════


class AppState:
    """Holds lazy-loaded model, graph, and recommender instances."""

    def __init__(self):
        self.model = None
        self.recommender = None
        self.evaluator = None
        self.adj_norm = None
        self.n_users = 0
        self.n_items = 0
        self.embedding_dim = 64
        self.n_layers = 3
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load model and supporting artefacts from disk."""
        cfg = get_config()
        model_path = Path(cfg.paths.get('saved_models', 'saved_models'))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        model_path = model_path / 'best_model.pt'

        processed_dir = Path(cfg.data.get('processed_data_path', 'data/processed'))
        if not processed_dir.is_absolute():
            processed_dir = PROJECT_ROOT / processed_dir

        if not model_path.exists():
            logger.warning(f"Model checkpoint not found at {model_path}")
            logger.warning("Start training first, or place best_model.pt in saved_models/")
            return

        logger.info(f"Loading model from {model_path}...")
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=True)

        model_cfg = checkpoint.get('config', {})
        self.n_users = model_cfg.get('n_users', 0)
        self.n_items = model_cfg.get('n_items', 0)
        self.embedding_dim = model_cfg.get('embedding_dim', 64)
        self.n_layers = model_cfg.get('n_layers', 3)

        if self.n_users == 0 or self.n_items == 0:
            logger.error("Invalid model config — missing n_users or n_items")
            return

        # Build model
        from models.lightgcn import LightGCN
        self.model = LightGCN(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            alpha=model_cfg.get('alpha', 'mean'),
            dropout=model_cfg.get('dropout', 0.0),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info("Model loaded and in eval mode.")

        # Build adjacency from saved graph or data
        self._load_graph(processed_dir)

        # Build recommender
        self._build_recommender()

        self._loaded = True
        logger.info("AppState initialised successfully.")

    def _load_graph(self, processed_dir: Path) -> None:
        """Try loading graph; fall back to building from data if available."""
        graph_dir = processed_dir / 'graph'
        if graph_dir.exists() and (graph_dir / 'graph_adj.npz').exists():
            from graph.build_graph import GraphBuilder
            builder = GraphBuilder.load(str(graph_dir))
            self.adj_norm = builder.get_torch_adj()
            logger.info(f"Graph loaded from {graph_dir}")
            return

        # Fallback: build from processed training data
        train_path = processed_dir / 'train.parquet'
        if train_path.exists():
            import pandas as pd
            from graph.build_graph import GraphBuilder
            df = pd.read_parquet(train_path)
            builder = GraphBuilder().build_from_dataframe(
                df, n_users=self.n_users, n_items=self.n_items,
            )
            self.adj_norm = builder.get_torch_adj()
            logger.info(f"Graph built from training data ({len(df)} interactions)")
            return

        # Last resort: build a dummy adjacency so the model can propagate
        # (recommendations won't be meaningful without the real graph)
        logger.warning(
            "No graph data found — building a fallback adjacency. "
            "Run the full preprocessing + training pipeline for "
            "meaningful recommendations."
        )
        self._build_fallback_graph()

    def _build_fallback_graph(self) -> None:
        """Build a minimal bipartite adjacency for demo purposes."""
        from graph.build_graph import GraphBuilder

        # Generate random interactions so the model can propagate
        rng = np.random.RandomState(42)
        n_interactions = max(1, min(self.n_users * self.n_items // 10, 500))
        users = rng.randint(0, self.n_users, size=n_interactions)
        items = rng.randint(0, self.n_items, size=n_interactions)

        builder = GraphBuilder()
        builder.build_from_arrays(
            users=users,
            items=items,
            n_users=self.n_users,
            n_items=self.n_items,
        )
        self.adj_norm = builder.get_torch_adj()
        logger.info(
            f"Fallback graph built: {self.n_users} users, "
            f"{self.n_items} items, {n_interactions} random edges"
        )

    def _build_recommender(self) -> None:
        """Build Recommender instance from loaded model + graph."""
        if self.model is None or self.adj_norm is None:
            return

        # Build train_user_items from adjacency if we have graph
        train_user_items: Dict[int, List[int]] = {}
        # Default: no training items (seen items won't be excluded)
        # In production, load from the processed split data

        from recommendation.recommend import Recommender
        self.recommender = Recommender(
            model=self.model,
            adj_norm=self.adj_norm,
            n_users=self.n_users,
            n_items=self.n_items,
            train_user_items=train_user_items,
        )
        logger.info("Recommender instance created.")

    def unload(self) -> None:
        """Clean up loaded artefacts."""
        self.model = None
        self.recommender = None
        self.evaluator = None
        self.adj_norm = None
        self._loaded = False
        logger.info("AppState unloaded.")


# ══════════════════════════════════════════════════════════════════════
# Application factory
# ══════════════════════════════════════════════════════════════════════

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle handler: load model on startup, clean up on shutdown."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("LightGCN Recommender API starting...")
    logger.info("=" * 60)
    state.load()
    yield
    state.unload()
    logger.info("LightGCN Recommender API shut down.")


app = FastAPI(
    title="LightGCN Recommender System",
    description="Production-quality Top-N recommendation engine "
                "using Light Graph Convolution Network (LightGCN).",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ══════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health():
    """Return service status and basic model info."""
    return HealthResponse(
        status="ok" if state.loaded else "degraded",
        model_loaded=state.loaded,
        n_users=state.n_users if state.loaded else None,
        n_items=state.n_items if state.loaded else None,
    )


@app.get(
    "/info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Model information",
)
async def model_info():
    """Return model configuration and dataset dimensions."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        n_users=state.n_users,
        n_items=state.n_items,
        embedding_dim=state.embedding_dim,
        n_layers=state.n_layers,
        model_path="saved_models/best_model.pt",
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    tags=["Recommendation"],
    summary="Get Top-N recommendations for a user",
)
async def recommend_user(
    user_id: int,
    top_k: int = Query(10, ge=1, le=500, description="Number of recommendations"),
    exclude_seen: bool = Query(True, description="Exclude already-interacted items"),
    return_scores: bool = Query(False, description="Include model scores"),
):
    """Get top-K item recommendations for a single user.

    Args:
        user_id: Zero-based user index.
        top_k: Number of items to return.
        exclude_seen: Whether to filter out items the user has interacted with.
        return_scores: If true, include the raw model scores.

    Returns:
        ``RecommendResponse`` with ranked item indices.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if user_id < 0 or user_id >= state.n_users:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} out of range [0, {state.n_users - 1}]",
        )

    try:
        result = state.recommender.recommend(
            user_id=user_id,
            top_k=top_k,
            exclude_seen=exclude_seen,
            return_scores=return_scores,
        )
    except Exception as e:
        logger.exception(f"Recommendation failed for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

    if return_scores:
        items, scores = result
        return RecommendResponse(
            user_id=user_id, recommendations=items, scores=scores,
        )
    return RecommendResponse(user_id=user_id, recommendations=result)


@app.post(
    "/recommend/batch",
    response_model=BatchRecommendResponse,
    tags=["Recommendation"],
    summary="Get recommendations for multiple users",
)
async def recommend_batch(body: RecommendRequest):
    """Get Top-N recommendations for a list of users.

    Args:
        body: JSON body with ``user_ids``, ``top_k``, ``exclude_seen``,
              and ``return_scores``.

    Returns:
        ``BatchRecommendResponse`` mapping user IDs to their recommendations.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    invalid = [u for u in body.user_ids if u < 0 or u >= state.n_users]
    if invalid:
        raise HTTPException(
            status_code=404,
            detail=f"Invalid user IDs: {invalid}. Range is [0, {state.n_users - 1}]",
        )

    try:
        results = state.recommender.recommend_batch(
            user_ids=body.user_ids,
            top_k=body.top_k,
            exclude_seen=body.exclude_seen,
            return_scores=body.return_scores,
        )
    except Exception as e:
        logger.exception("Batch recommendation failed")
        raise HTTPException(status_code=500, detail=str(e))

    resp = {}
    for uid, result in results.items():
        if body.return_scores:
            items, scores = result
            resp[str(uid)] = RecommendResponse(
                user_id=uid, recommendations=items, scores=scores,
            )
        else:
            resp[str(uid)] = RecommendResponse(user_id=uid, recommendations=result)

    return BatchRecommendResponse(recommendations=resp)


@app.get(
    "/evaluate",
    response_model=EvaluateResponse,
    tags=["Evaluation"],
    summary="Run model evaluation",
)
async def evaluate(
    k: int = Query(10, ge=1, le=100, description="K value for metrics"),
):
    """Evaluate the model on all users.

    Returns Recall@K, Precision@K, NDCG@K, and HitRate@K.

    Note: Requires test interaction data to be loaded. If no test data is
    available, metrics will be zero.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if state.model is None or state.adj_norm is None:
        raise HTTPException(status_code=503, detail="Model or graph not available")

    # Build evaluator with empty test data (will return 0 metrics)
    # In production, load train/test splits from processed data
    from evaluation.metrics import Evaluator

    evaluator = Evaluator(
        model=state.model,
        adj_norm=state.adj_norm,
        n_users=state.n_users,
        n_items=state.n_items,
        train_user_items={},
        test_user_items={},
        k_values=[k],
    )
    results = evaluator.evaluate()

    return EvaluateResponse(
        metrics=results,
        n_users_evaluated=state.n_users,
        model_info={
            "n_users": state.n_users,
            "n_items": state.n_items,
            "embedding_dim": state.embedding_dim,
            "n_layers": state.n_layers,
        },
    )


@app.post(
    "/reload",
    tags=["System"],
    summary="Reload model from disk",
)
async def reload_model():
    """Reload the model checkpoint from disk (hot-reload)."""
    state.unload()
    state.load()
    return {"status": "ok", "model_loaded": state.loaded}


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    host = cfg.api.get('host', '0.0.0.0')
    port = int(cfg.api.get('port', 8000))
    reload_flag = cfg.api.get('reload', True)

    print(f"Starting LightGCN Recommender API on {host}:{port}")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload_flag,
        log_level="info",
    )
