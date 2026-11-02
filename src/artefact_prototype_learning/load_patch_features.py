"""Load patch embeddings from one .h5 file per slide"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


def load_features(feature_dir: str | Path, cache: str | Path | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all slides' patch embeddings from a folder of .h5 files.

    Args:
        feature_dir: folder with one .h5 file per slide, each with datasets 'features' [n, d] and
            'coords' [n, 2] (level-0 x, y of each patch).
        cache: optional path without extension, e.g. "cache/uni". The first run saves the features
            there; later runs load them in seconds without reading the .h5 files again.

    Returns:
        X: [n_patches, d] float32 embeddings, each row L2-normalised.
        meta: one row per patch with slide, x, y and norm (the embedding's original L2 norm).
    """
    # find the files
    files = sorted(Path(feature_dir).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files in {feature_dir}")
    
    # check the cache and load it if it's up to date
    if cache:
        npy, pkl = Path(f"{cache}.npy"), Path(f"{cache}_meta.pkl") # cache contains matrix X of L2-norm features and metadata
        if pkl.exists() and pkl.stat().st_mtime > max(p.stat().st_mtime for p in files):
            meta = pd.read_pickle(pkl)
            if list(meta["slide"].unique()) == [p.stem for p in files]:
                return np.load(npy, mmap_mode="r"), meta
        npy.parent.mkdir(parents=True, exist_ok=True)

    # check h5 files
    shapes = []
    for p in files:
        # check features and coordinates are stored as expected
        with h5py.File(p, "r") as f:
            if "features" not in f or "coords" not in f:
                raise KeyError(f"{p} needs datasets 'features' and 'coords', found {list(f)}")
            fs, cs = f["features"].shape, f["coords"].shape
        # check shapes are compatible
        if len(fs) != 2 or cs != (fs[0], 2):
            raise ValueError(f"{p}: features {fs} and coords {cs}, expected [n, d] and [n, 2]")
        shapes.append(fs)
    # check consistency in embedding dimensions
    if len({d for _, d in shapes}) > 1:
        raise ValueError(f"{feature_dir} mixes different feature sizes {sorted({d for _, d in shapes})}; use one folder per encoder")

    # load the features and coordinates
    n, d = sum(n for n, _ in shapes), shapes[0][1] # n=num patches, d=embedding dimension
    X = np.lib.format.open_memmap(npy, "w+", np.float32, (n, d)) if cache else np.empty((n, d), np.float32)
    meta, i = [], 0
    for p in tqdm(files, desc="loading features"):
        with h5py.File(p, "r") as f:
            feats, coords = f["features"][:].astype(np.float32), f["coords"][:]
        feats, norm = normalize(feats, norm="l2", return_norm=True) # L2-normalise each patch
        X[i:i + len(feats)] = feats
        meta.append(pd.DataFrame({"slide": p.stem, "x": coords[:, 0], "y": coords[:, 1], "norm": norm}))
        i += len(feats)
    meta = pd.concat(meta, ignore_index=True)

    if cache:
        X.flush()
        meta.to_pickle(pkl)
    return X, meta
