"""Plots for looking at the results: patch images, prototypes, the type dendrogram, and slide heatmaps.

Patch images and slide thumbnails need OpenSlide: pip install ".[slides]".
"""
import math
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def patch_grid(meta: pd.DataFrame, rows, wsi_dir: str | Path, patch_size: int = 512, mag: float = 20,
               ncols: int = 8, title: str = "") -> plt.Figure:
    """Images of the given rows of the patch table, e.g. the candidates or the flagged patches.

    Args:
        meta: patch table with slide, x, y (from load_features, or the table from fit).
        rows: row numbers to show, e.g. np.where(table["candidate"])[0][:32].
        wsi_dir: folder of the slide files, named <slide>.<extension>.
        patch_size, mag: size in pixels and magnification the patches were extracted at.
    """
    rows = [int(i) for i in rows]
    nrows = max(1, math.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), rows):
        patch = meta.iloc[i]
        ax.imshow(read_patch(open_slide(wsi_dir, patch["slide"]), patch["x"], patch["y"], patch_size, mag))
        ax.set_title(str(i), fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def prototype_patches(X: np.ndarray, meta: pd.DataFrame, model: dict, wsi_dir: str | Path, n: int = 8,
                      patch_size: int = 512, mag: float = 20) -> plt.Figure:
    """One row per prototype (artefact types first, then tissue) showing its n most similar patches.

    Use it to see what each artefact type is, and to decide which are really tissue (--not-artefact).
    """
    protos = [(f"artefact type {k}", p) for k, p in enumerate(model["art"])]
    protos += [(f"tissue {k}", p) for k, p in enumerate(model["tissue"])]
    fig, axes = plt.subplots(len(protos), n, figsize=(1.5 * n, 1.6 * len(protos)), squeeze=False)
    for row, (name, p) in zip(axes, protos):
        nearest = np.argsort(-(X @ p))[:n]                 # cosine similarity: X and prototypes are L2-normalised
        for ax, i in zip(row, nearest):
            patch = meta.iloc[i]
            ax.imshow(read_patch(open_slide(wsi_dir, patch["slide"]), patch["x"], patch["y"], patch_size, mag))
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    return fig


def dendrogram_cut(Z: np.ndarray, type_dist: float) -> plt.Figure:
    """The candidates' dendrogram with the type cut (type_dist) as a dashed line. It should sit in a clear gap."""
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(7, 4))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=40, color_threshold=type_dist, no_labels=True,
               above_threshold_color="0.6")
    ax.axhline(type_dist, color="crimson", lw=1, ls="--")
    ax.set_ylabel("cosine distance")
    fig.tight_layout()
    return fig


def slide_heatmap(table: pd.DataFrame, slide: str, column: str = "classifier", wsi_dir: str | Path | None = None,
                  vmin: float | None = 0, vmax: float | None = 1, cmap: str = "turbo") -> plt.Figure:
    """One slide's patches coloured by a score, next to the slide thumbnail if wsi_dir is given.

    Args:
        table: rows with slide, x, y and the score column (the table from fit, or apl_scores.csv).
        slide: which slide to show.
        column: the score, e.g. "classifier" (0 to 1) or "margin" (with vmin=None, vmax=None).
        wsi_dir: folder of the slide files, to show the thumbnail.
        vmin, vmax: colour range; None takes it from the slide's values.
    """
    t = table[table["slide"] == slide]
    step = np.diff(np.unique(t["x"])).min()            # patch width in level-0 pixels = smallest x spacing
    squares = PatchCollection([Rectangle((x, y), step, step) for x, y in zip(t["x"], t["y"])], cmap=cmap)
    squares.set_array(t[column].to_numpy())
    squares.set_clim(vmin, vmax)

    fig, axes = plt.subplots(1, 2 if wsi_dir else 1, figsize=(12 if wsi_dir else 6.5, 6), squeeze=False)
    axes[0, -1].add_collection(squares)
    fig.colorbar(squares, ax=axes[0, -1], shrink=0.7, label=column)
    if wsi_dir:
        s = open_slide(wsi_dir, slide)
        axes[0, 0].imshow(s.get_thumbnail((2000, 2000)), extent=(0, s.dimensions[0], s.dimensions[1], 0))
    for ax in axes[0]:
        ax.set_xlim(t["x"].min(), t["x"].max() + step)
        ax.set_ylim(t["y"].max() + step, t["y"].min())
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(slide, fontsize=9)
    fig.tight_layout()
    return fig


@lru_cache(maxsize=32)                                   # keep recently used slides open
def open_slide(wsi_dir: str | Path, slide: str):
    """Open the slide file named <slide>.<any extension> in wsi_dir."""
    import openslide

    files = sorted(Path(wsi_dir).glob(f"{slide}.*"))
    if not files:
        raise FileNotFoundError(f"no slide file {slide}.* in {wsi_dir}")
    return openslide.OpenSlide(str(files[0]))


def read_patch(slide, x: int, y: int, patch_size: int = 512, mag: float = 20):
    """RGB image of the patch with level-0 top-left (x, y), patch_size px wide at magnification mag."""
    import openslide

    objective = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if objective:
        downsample = float(objective) / mag
    else:                                                 # no objective power: use microns per pixel (20x ~ 0.5)
        downsample = (10.0 / mag) / float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5))
    level = slide.get_best_level_for_downsample(downsample) if downsample > 1 else 0
    size = round(patch_size * downsample / slide.level_downsamples[level])
    image = slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB")
    return image.resize((patch_size, patch_size))
