from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, example="This API is incredible!")
    model: Optional[str] = Field(default="distilbert-finetuned")
    language: Optional[str] = Field(default="en")

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
    label:      str
    confidence: float
    scores:     ScoresMap
    metadata:   Metadata

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)

class BatchResponse(BaseModel):
    results:       List[Dict]
    count:         int
    processing_ms: float

class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    cache_connected: bool
    model_version:   str