"""Geometric outlier scores and candidate selection.
Here, we implement angular distance and kNN on L2-normalised embeddings and Mahalanobis distance on raw embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

MEASURES = ["angle", "mahalanobis", "density"] # and a custom score that can be added

def angle_to_mean(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Angle between each patch and the mean direction of the reference sample."""
    mean = X[ref].mean(axis=0)
    cos = X @ (mean / np.linalg.norm(mean))
    return np.arccos(np.clip(cos, -1.0, 1.0))

def mahalanobis(X: np.ndarray, norms: np.ndarray, ref: np.ndarray, dim: int = 50, chunk: int = 100_000) -> np.ndarray:
    """Mahalanobis distance of each raw embedding (X * norms) in the top `dim` principal components of the reference."""
    R = X[ref] * norms[ref, None]
    mean = R.mean(axis=0)
    _, s, vt = np.linalg.svd(R - mean, full_matrices=False)
    to_standardised_pcs = vt[:dim].T / (s[:dim] / np.sqrt(len(ref) - 1))
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):                  
        raw = X[i:i + chunk] * norms[i:i + chunk, None]
        out[i:i + chunk] = np.linalg.norm((raw - mean) @ to_standardised_pcs, axis=1)
    return out

def knn_density(X: np.ndarray, ref: np.ndarray, k: int = 20) -> np.ndarray:
    """Mean angle to the k nearest other reference patches, for the reference patches only (NaN elsewhere).
    Larger = sparser neighbourhood. Only the reference sample is scored, so candidates always come from it.
    """
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X[ref]).kneighbors(X[ref])
    angles = 2 * np.arcsin(np.clip(dist[:, 1:] / 2, 0.0, 1.0))   # drop column 0 (the patch itself); chord -> angle
    out = np.full(len(X), np.nan)
    out[ref] = angles.mean(axis=1)
    return out

def score(X: np.ndarray, norms: np.ndarray, ref: np.ndarray, mahal_dim: int = 50, knn: int = 20,
          custom: dict | None = None) -> pd.DataFrame:
    """All scores, one row per patch.

    custom: optional {name: function} of your own scores. Each function takes (X, norms, ref) and returns one
        value per patch, larger = more unusual. They become extra columns and count in `candidates` like the others.
    """
    scores = pd.DataFrame({
        "angle": angle_to_mean(X, ref),
        "mahalanobis": mahalanobis(X, norms, ref, mahal_dim),
        "density": knn_density(X, ref, knn),
    })
    for name, fn in (custom or {}).items():
        scores[name] = fn(X, norms, ref)
    return scores

def candidates(scores: pd.DataFrame, tail_pctl: float = 99.0, agree_min: int = 2, max_candidates: int = 400) -> np.ndarray:
    """Row indices of candidate artefacts, most extreme first.

    A score flags a patch above its tail_pctl percentile; candidates are flagged by at least agree_min scores.
    Every column of `scores` counts (custom ones too). Only patches with all scores are eligible. The max_candidates with the highest mean percentile rank are kept.
    """
    S = scores.dropna()
    flags = (S >= S.quantile(tail_pctl / 100)).sum(axis=1)
    mean_rank = S.rank(pct=True).mean(axis=1)
    return mean_rank[flags >= agree_min].sort_values(ascending=False).index.to_numpy()[:max_candidates]