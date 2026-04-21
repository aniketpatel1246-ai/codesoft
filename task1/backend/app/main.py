"""
CineGenre — FastAPI Backend
============================
Routes
  GET  /           → service info
  GET  /health     → model status + genre list
  POST /predict    → single-movie classification
  POST /batch      → classify up to 100 movies
  GET  /genres     → all 27 genres with training counts
  GET  /stats      → dataset + model statistics
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# allow importing sibling module
sys.path.insert(0, os.path.dirname(__file__))
from ml_pipeline import (
    predict_single, load_artifacts, artifacts_exist,
    ALL_GENRES, GENRE_COUNTS,
)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CineGenre API",
    description=(
        "Classify movie genres from title + description. "
        "Trained on 54,214 IMDB movies across 27 genres."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared model state ─────────────────────────────────────────────────────────

_model      = None
_vectorizer = None
_classes: list[str] = []


@app.on_event("startup")
def _load():
    global _model, _vectorizer, _classes
    if artifacts_exist():
        _model, _vectorizer, _classes = load_artifacts()
        print("✅  Model loaded successfully")
    else:
        print("⚠️   No model found — run: python backend/app/ml_pipeline.py")


def _require_model():
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first: python backend/app/ml_pipeline.py",
        )


# ── Request / Response schemas ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    title:       str = Field(..., example="The Dark Knight")
    description: str = Field(..., example="When the Joker wreaks havoc on Gotham…")


class PredictResponse(BaseModel):
    title:      str
    genre:      str
    confidence: Optional[dict] = None
    latency_ms: float


class MovieItem(BaseModel):
    id:          Optional[str] = None
    title:       str
    description: str


class BatchRequest(BaseModel):
    movies: list[MovieItem] = Field(..., max_length=100)


class BatchItem(BaseModel):
    id:         Optional[str]
    title:      str
    genre:      str
    confidence: Optional[dict] = None


class BatchResponse(BaseModel):
    predictions: list[BatchItem]
    total:       int
    latency_ms:  float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "CineGenre API",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "ok",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "genres":        sorted(_classes) if _classes else ALL_GENRES,
        "genre_count":   len(_classes) or len(ALL_GENRES),
        "train_samples": 54_214,
        "test_samples":  54_200,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """Classify a single movie."""
    _require_model()
    if not req.title.strip() and not req.description.strip():
        raise HTTPException(400, "Provide at least a title or description.")

    t0     = time.perf_counter()
    result = predict_single(
        req.title, req.description,
        model=_model, vectorizer=_vectorizer,
    )
    ms = round((time.perf_counter() - t0) * 1000, 2)
    return PredictResponse(
        title=req.title,
        genre=result["genre"],
        confidence=result["confidence"],
        latency_ms=ms,
    )


@app.post("/batch", response_model=BatchResponse, tags=["Inference"])
def batch(req: BatchRequest):
    """Classify up to 100 movies in one call."""
    _require_model()
    if len(req.movies) > 100:
        raise HTTPException(400, "Maximum 100 movies per request.")

    t0   = time.perf_counter()
    preds: list[BatchItem] = []
    for m in req.movies:
        r = predict_single(
            m.title, m.description,
            model=_model, vectorizer=_vectorizer,
        )
        preds.append(BatchItem(
            id=m.id, title=m.title,
            genre=r["genre"], confidence=r["confidence"],
        ))
    ms = round((time.perf_counter() - t0) * 1000, 2)
    return BatchResponse(predictions=preds, total=len(preds), latency_ms=ms)


@app.get("/genres", tags=["Metadata"])
def genres():
    """All 27 genres with their training sample counts."""
    gs = sorted(_classes) if _classes else ALL_GENRES
    return {
        "genres": [{"name": g, "count": GENRE_COUNTS.get(g, 0)} for g in gs],
        "total":  len(gs),
    }


@app.get("/stats", tags=["Metadata"])
def stats():
    return {
        "dataset": {
            "source":        "ftp://ftp.fu-berlin.de/pub/misc/movies/database/",
            "train_samples": 54_214,
            "test_samples":  54_200,
            "genres":        27,
            "top_genre":     "drama (13,613)",
            "rarest_genre":  "war (132)",
        },
        "model": {
            "type":       "LinearSVC",
            "features":   "TF-IDF unigrams+bigrams, 60 000 features",
            "input":      "title + description (combined)",
            "selection":  "5-fold cross-validation",
            "loaded":     _model is not None,
        },
    }
