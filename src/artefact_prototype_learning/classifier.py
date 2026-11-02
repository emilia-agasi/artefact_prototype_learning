"""From prototypes to every patch: the prototype margin, pseudo-labels from it, and a light classifier."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def margins(X: np.ndarray, art: np.ndarray, tissue: np.ndarray, chunk: int = 500_000) -> pd.DataFrame:
    """Per patch: cosine similarity to the closest artefact and tissue prototype, their difference and the closest type.

    margin = best_art - best_tissue: above 0 the patch is closer to an artefact prototype than to any tissue one.
    """
    parts = []
    for i in range(0, len(X), chunk):                   # in chunks: [patches x prototypes] for all patches is large
        sim_art, sim_tissue = X[i:i + chunk] @ art.T, X[i:i + chunk] @ tissue.T
        parts.append(pd.DataFrame({"best_art": sim_art.max(axis=1), "best_tissue": sim_tissue.max(axis=1),
                                   "nearest_art": sim_art.argmax(axis=1)}))
    m = pd.concat(parts, ignore_index=True)
    m["margin"] = m["best_art"] - m["best_tissue"]
    return m

def pseudo_labels(m: pd.DataFrame, cands: np.ndarray, types: np.ndarray, tau: float = 0.06, gap: float = 0.04,
                  art_cap: int = 3000, tissue_n: int = 20000, seed: int = 0) -> np.ndarray:
    """Training labels from the margin, leaving the uncertain band tau +- gap unlabelled.

    Returns one label per patch: -1 unlabelled, 0 tissue, k + 1 artefact type k.
    - artefact: up to art_cap patches per type with margin > tau + gap, labelled with their closest type,
      plus every typed candidate with its own type;
    - tissue: tissue_n random unlabelled patches with margin < tau - gap.
    """
    rng = np.random.default_rng(seed)
    labels = np.full(len(m), -1)
    margin, nearest = m["margin"].to_numpy(), m["nearest_art"].to_numpy()
    for k in np.unique(nearest[margin > tau + gap]):
        clear_k = np.where((margin > tau + gap) & (nearest == k))[0]
        labels[rng.choice(clear_k, min(art_cap, len(clear_k)), replace=False)] = k + 1
    labels[cands[types >= 0]] = types[types >= 0] + 1
    clear_tissue = np.where((margin < tau - gap) & (labels == -1))[0]
    labels[rng.choice(clear_tissue, min(tissue_n, len(clear_tissue)), replace=False)] = 0
    return labels

def fit_classifier(X: np.ndarray, labels: np.ndarray, C: float = 1.0) -> LogisticRegression:
    """Logistic regression, artefact (any type) vs tissue, on the labelled patches (label >= 0).

    class_weight="balanced" stops the many tissue labels from drowning out the few artefact ones.
    """
    rows = np.where(labels >= 0)[0]
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X[rows], labels[rows] > 0)