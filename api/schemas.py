"""
Pydantic schemas for the LightGCN Recommender API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("ok", description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    n_users: Optional[int] = Field(None, description="Number of users in the model")
    n_items: Optional[int] = Field(None, description="Number of items in the model")


class RecommendRequest(BaseModel):
    """Request body for batch recommendation."""
    user_ids: List[int] = Field(..., description="List of user indices to recommend for")
    top_k: int = Field(10, ge=1, le=500, description="Number of recommendations per user")
    exclude_seen: bool = Field(True, description="Exclude already-interacted items")
    return_scores: bool = Field(False, description="Include model scores in response")


class RecommendResponse(BaseModel):
    """Single-user recommendation response."""
    user_id: int = Field(..., description="User index")
    recommendations: List[int] = Field(..., description="Recommended item indices (ranked)")
    scores: Optional[List[float]] = Field(None, description="Model scores if requested")


class BatchRecommendResponse(BaseModel):
    """Batch recommendation response."""
    recommendations: Dict[str, RecommendResponse] = Field(
        ..., description="User-keyed recommendation results"
    )


class EvaluateResponse(BaseModel):
    """Evaluation results response."""
    metrics: Dict[str, float] = Field(..., description="Metric name → value")
    n_users_evaluated: int = Field(..., description="Number of users in evaluation")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model metadata")


class ModelInfoResponse(BaseModel):
    """Model and dataset information."""
    n_users: int
    n_items: int
    n_interactions: Optional[int] = None
    embedding_dim: int
    n_layers: int
    model_path: str
    status: str = Field("loaded", description="Model status")


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error description")
