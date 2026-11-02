"""Here we define artefact and tissue prototypes (L2-normalised reference vectors)."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def artefact_prototypes(X: np.ndarray, cands: np.ndarray, type_dist: float = 0.3,
                        min_type_size: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster candidates using hierarchical clustering. The mean of each cluster we define as prototype.
    Returns:
        prototypes: [n_types, d], the L2-normalised mean of each type's candidates.
        types: type of each candidate, numbered largest first; -1 = in a cluster smaller than min_type_size.
        Z: the dendrogram (scipy linkage), for plotting.
    """
    Z = linkage(X[cands], method="average", metric="cosine")
    clusters = fcluster(Z, type_dist, criterion="distance")
    ids, sizes = np.unique(clusters, return_counts=True)
    kept = ids[sizes >= min_type_size][np.argsort(-sizes[sizes >= min_type_size])]
    if not len(kept):
        raise ValueError("no artefact type has min_type_size candidates: raise type_dist or lower min_type_size")
    types = np.full(len(cands), -1)
    for k, cluster in enumerate(kept):
        types[clusters == cluster] = k
    prototypes = normalize(np.stack([X[cands[types == k]].mean(axis=0) for k in range(len(kept))]))
    return prototypes, types, Z

def tissue_prototypes(X: np.ndarray, mahalanobis: np.ndarray, n_tissue: int = 8, tissue_core_n: int = 20000,
                      seed: int = 0) -> np.ndarray:
    """k-means centroids of patches with Mahalanobis at or below the median.
    """
    core = np.where(mahalanobis <= np.nanmedian(mahalanobis))[0]
    core = np.random.default_rng(seed).choice(core, min(tissue_core_n, len(core)), replace=False)
    kmeans = KMeans(n_clusters=n_tissue, n_init=10, random_state=seed).fit(X[core])
    return normalize(kmeans.cluster_centers_)

def move_to_tissue(art: np.ndarray, tissue: np.ndarray, types: np.ndarray,
                   not_artefact: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Option to move prototypes to tissue after visual inspection."""
    move = sorted({int(k) for k in not_artefact})
    if not move:
        return art, tissue, types
    if move[-1] >= len(art):
        raise ValueError(f"not_artefact {move}: there are only {len(art)} artefact types")
    keep = np.setdiff1d(np.arange(len(art)), move)
    new_number = np.full(len(art), -1)
    new_number[keep] = np.arange(len(keep))
    return art[keep], np.vstack([tissue, art[move]]), np.where(types >= 0, new_number[types], -1)