"""Command line: `apl fit` on a cohort, `apl score` for new slides. Run `apl fit --help` for all options."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, pipeline
from .config import DEFAULTS, make_settings
from .load_patch_features import load_features


def fit_cmd(args):
    cfg = make_settings(seed=args.seed, flag_fraction=args.flag_fraction,
                        prototypes={"type_dist": args.type_dist, "not_artefact": args.not_artefact})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, meta = load_features(args.features, args.cache)
    fit_mask = np.ones(len(meta), bool)
    if args.holdout:                                         # these slides are scored but never used for fitting
        fit_mask = ~meta["slide"].isin(pd.read_csv(args.holdout)["slide"]).to_numpy()
    print(f"{len(meta):,} patches from {meta['slide'].nunique()} slides, {fit_mask.sum():,} used for fitting")

    model, table = pipeline.fit(X, meta, cfg, fit_mask)
    pipeline.save_model(model, out / "model.npz")
    table.to_csv(out / "patch_scores.csv", index=False)
    summary = run_summary(args, cfg, table, model)
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["results"], indent=2))
    if args.plots:
        plots(X, meta, table, model, cfg, args)
    print(f"wrote {out}")


def run_summary(args, cfg, table, model) -> dict:
    """Everything needed to repeat and report a run: version, date, command-line options, settings, results."""
    types = table.loc[table["candidate"], "type"]
    labels = table["pseudo_label"]
    return {
        "apl_version": __version__,
        "date": datetime.now().isoformat(timespec="seconds"),
        "options": {k: v for k, v in vars(args).items() if k not in ("func", "command")},
        "settings": cfg,
        "results": {
            "slides": int(table["slide"].nunique()),
            "patches": len(table),
            "patches_used_for_fitting": int(table["fit"].sum()),
            "candidates": len(types),
            "artefact_types": len(model["art"]),
            "candidates_per_type": np.bincount(types[types >= 0], minlength=len(model["art"])).tolist(),
            "tissue_prototypes": len(model["tissue"]),
            "pseudo_labels_artefact": int((labels > 0).sum()),
            "pseudo_labels_tissue": int((labels == 0).sum()),
            "classifier_cut": float(model["cut"]),
            "percent_flagged_fit": round(100 * float(table.loc[table["fit"], "flagged"].mean()), 2),
            "percent_flagged_all": round(100 * float(table["flagged"].mean()), 2),
        },
    }


def plots(X, meta, table, model, cfg, args):
    """The candidate dendrogram and, with --wsi-dir, the patches most similar to each prototype."""
    import matplotlib
    matplotlib.use("Agg")
    from scipy.cluster.hierarchy import linkage

    from . import visualisation as viz

    out = Path(args.out)
    Z = linkage(X[table["candidate"].to_numpy()], method="average", metric="cosine")
    viz.dendrogram_cut(Z, cfg["prototypes"]["type_dist"]).savefig(out / "dendrogram.png", dpi=150)
    if args.wsi_dir:                                         # look at these to decide not_artefact
        fig = viz.prototype_patches(X, meta, model, args.wsi_dir, patch_size=args.patch_size, mag=args.mag)
        fig.savefig(out / "prototypes.png", dpi=150)


def score_cmd(args):
    model = pipeline.load_model(args.model)
    X, meta = load_features(args.features, args.cache)
    scores = pd.concat([meta[["slide", "x", "y"]], pipeline.score(model, X)], axis=1)
    scores.to_csv(args.out, index=False)
    print(f"scored {len(scores):,} patches from {meta['slide'].nunique()} slides -> {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="apl", description="Artefact prototype learning on patch embeddings.")
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit on a cohort and score all its patches")
    f.add_argument("features", help="folder of per-slide .h5 files (datasets 'features' and 'coords')")
    f.add_argument("--out", default="apl_output", help="output folder (default: apl_output)")
    f.add_argument("--holdout", help="csv with a 'slide' column: these slides are scored but not used for fitting")
    f.add_argument("--cache", help="save the features here once (path without extension) for fast reloads")
    f.add_argument("--type-dist", type=float, default=DEFAULTS["prototypes"]["type_dist"],
                   help="where the candidate dendrogram is cut into artefact types (default %(default)s, see dendrogram.png)")
    f.add_argument("--not-artefact", type=int, nargs="+", default=[], metavar="TYPE",
                   help="artefact types that are really tissue, e.g. --not-artefact 5 (see prototypes.png)")
    f.add_argument("--flag-fraction", type=float, default=DEFAULTS["flag_fraction"],
                   help="share of fitting patches flagged as artefact, sets the cut-off (default %(default)s)")
    f.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="random seed (default %(default)s)")
    f.add_argument("--plots", action="store_true", help="also save the candidate dendrogram (dendrogram.png)")
    f.add_argument("--wsi-dir",
                   help="with --plots: folder of the slides, to save the patches nearest each prototype (prototypes.png)")
    f.add_argument("--patch-size", type=int, default=512, help="with --wsi-dir: patch size in px (default 512)")
    f.add_argument("--mag", type=float, default=20, help="with --wsi-dir: magnification of the patches (default 20)")
    f.set_defaults(func=fit_cmd)

    s = sub.add_parser("score", help="score new slides with a fitted model")
    s.add_argument("model", help="model.npz from apl fit")
    s.add_argument("features", help="folder of per-slide .h5 files from the same encoder")
    s.add_argument("--out", default="apl_scores.csv", help="output csv (default: apl_scores.csv)")
    s.add_argument("--cache", help="as for apl fit")
    s.set_defaults(func=score_cmd)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
