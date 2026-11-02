"""The whole method in two calls: fit on a cohort's embeddings, then score any patches.

    model, table = fit(X, meta, cfg)      # candidates -> prototypes -> margin -> pseudo-labels -> classifier
    scores = score(model, X_new)          # margin and classifier score for new patches
    save_model(model, "model.npz"); model = load_model("model.npz")

Both scores are confidence scores: higher = more likely an artefact. `flagged` applies a cut-off to the classifier
score, set by fit so that flag_fraction (default 10%) of the fitting patches are flagged.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import classifier, prototypes, scoring


def fit(X: np.ndarray, meta: pd.DataFrame, cfg: dict, fit_mask: np.ndarray | None = None,
        custom: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Fit APL on a cohort and score every patch.

    Args:
        X, meta: from load_features.
        cfg: settings from make_settings.
        fit_mask: True for patches used to fit (default: all). The others (held-out slides) are only scored.
        custom: optional {name: function} of extra candidate scores, see scoring.score.

    Returns:
        model: dict of arrays (prototypes and classifier weights), for score and save_model.
        table: one row per patch: slide, x, y, the geometric scores, candidate, type, fit, pseudo_label, margin,
            nearest_art, classifier.
    """
    sc, pc, lc, seed = cfg["scoring"], cfg["prototypes"], cfg["learning"], cfg["seed"]
    fit_mask = np.ones(len(X), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    fit_idx = np.where(fit_mask)[0]

    # 1. geometric scores against a reference sample of fitting patches; candidates where they agree
    ref = np.sort(np.random.default_rng(seed).choice(fit_idx, min(sc["sample_n"], len(fit_idx)), replace=False))
    geo = scoring.score(X, meta["norm"].to_numpy(), ref, sc["mahal_dim"], sc["knn"], custom)
    cands = scoring.candidates(geo.iloc[fit_idx], sc["tail_pctl"], sc["agree_min"], sc["max_candidates"])

    # 2. artefact prototypes (one per candidate type) and tissue prototypes (typical fitting patches)
    art, types, _ = prototypes.artefact_prototypes(X, cands, pc["type_dist"], pc["min_type_size"])
    mahal = np.where(fit_mask, geo["mahalanobis"], np.nan)        # held-out patches never become tissue prototypes
    tissue = prototypes.tissue_prototypes(X, mahal, pc["n_tissue"], pc["tissue_core_n"], seed)
    art, tissue, types = prototypes.move_to_tissue(art, tissue, types, pc["not_artefact"] or [])

    # 3. margin for every patch, and pseudo-labels where it is clear (fitting patches only)
    margins = classifier.margins(X, art, tissue)
    labels = classifier.pseudo_labels(margins, cands, types, lc["tau"], lc["gap"], lc["art_cap"], lc["tissue_n"], seed)
    labels[~fit_mask] = -1

    # 4. light classifier, artefact vs tissue
    clf = classifier.fit_classifier(X, labels, lc["C"])
    model = {"art": art, "tissue": tissue, "coef": clf.coef_[0].astype(np.float32),
             "intercept": np.float32(clf.intercept_[0]), "cut": np.float32(np.inf)}

    # 5. cut-off: flag the top flag_fraction of the fitting patches by classifier score
    scores = score(model, X)
    model["cut"] = np.float32(np.quantile(scores["classifier"][fit_mask], 1 - cfg["flag_fraction"]))
    scores["flagged"] = scores["classifier"] > model["cut"]

    table = meta[["slide", "x", "y"]].copy()
    table[list(geo.columns)] = geo.to_numpy()
    table["candidate"] = False
    table.loc[cands, "candidate"] = True
    table["type"] = -1
    table.loc[cands, "type"] = types
    table["fit"] = fit_mask
    table["pseudo_label"] = labels                      # -1 none, 0 tissue, k + 1 artefact type k
    return model, pd.concat([table, scores], axis=1)


def score(model: dict, X: np.ndarray, chunk: int = 500_000) -> pd.DataFrame:
    """Score patches with a fitted model: margin, nearest_art (closest artefact type), classifier and flagged.

    classifier is the logistic regression's P(artefact). It ranks patches well but is not calibrated: it was
    trained on clear cases only, so milder artefacts often score well below 0.5. flagged = classifier above the
    model's cut-off, which fit sets so that flag_fraction of the fitting patches are flagged.
    """
    m = classifier.margins(X, model["art"], model["tissue"], chunk)
    logits = np.concatenate([X[i:i + chunk] @ model["coef"] for i in range(0, len(X), chunk)]) + model["intercept"]
    p = expit(logits)
    return pd.DataFrame({"margin": m["margin"], "nearest_art": m["nearest_art"], "classifier": p,
                         "flagged": p > model["cut"]})


def save_model(model: dict, path: str | Path) -> None:
    """Save as a plain .npz of numpy arrays (no pickle), so it is safe to share and loads anywhere."""
    np.savez(path, **model)


def load_model(path: str | Path) -> dict:
    """Load a model saved by save_model."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}