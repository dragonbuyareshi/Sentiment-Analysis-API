"""
api/schemas.py
──────────────
Pydantic v2 request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        example="This product exceeded all my expectations!",
    )
    model: Optional[str] = Field(
        default="distilbert-finetuned",
        description="Model ID to use for inference",
    )
    language: Optional[str] = Field(
        default="en",
        description="Language code (auto-detected if not set)",
    )


class ScoresMap(BaseModel):
    POSITIVE: float
    NEUTRAL:  float
    NEGATIVE: float


class Metadata(BaseModel):
    model_version:  str
    processing_ms:  float
    token_count:    int
    cached:         bool = False


class AnalyzeResponse(BaseModel):
    label:      str = Field(..., description="Predicted sentiment class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores:     ScoresMap
    metadata:   Metadata


class BatchRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        example=["Great product!", "Terrible service.", "It was okay."],
    )


class BatchResultItem(BaseModel):
    label:      str
    confidence: float
    scores:     ScoresMap


class BatchResponse(BaseModel):
    results:       List[BatchResultItem]
    count:         int
    processing_ms: float


class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    cache_connected: bool
    model_version:   str


class ModelsResponse(BaseModel):
    models: List[Dict]
