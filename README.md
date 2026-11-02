# Artefact prototype learning (APL)

Here we introduce APL, an unsupervised artefact detection tool for histopathology based on pre-trained patch embeddings.

## How it works

1. **Find candidates.** Patches that are extreme on at least two of three geometric scores are likely artefacts. There is the option here to implement and test your own assumptions.
2. **Build prototypes.** Candidates are clustered into artefact prototypes. Independently, tissue anchors are identified.
3. **Score every patch.** We score a patch by its `margin` = closeness to the nearest artefact prototype minus closeness to the nearest tissue
   prototype. A classifier trained on the resulting pseudo-labels outputs P(artefact).

`flagged` marks the patches above a cut-off on `classifier`, set so that
10% of the fitting patches are flagged (adapt with `--flag-fraction`). We recommend to visually inspect flagged patches.

## Getting started

```bash
git clone https://github.com/emilia-agasi/apl.git
cd apl
pip install .
pip install ".[slides]" #optional: for visual inspection
```

## Input

APL expects patch embeddings to be in `.h5` format with one file per slide, named after the slide, with two datasets:

| dataset    | shape    | content                                  |
| ---------- | -------- | ---------------------------------------- |
| `features` | `[n, d]` | patch embeddings from one encoder        |
| `coords`   | `[n, 2]` | level-0 `x, y` of each patch's top-left  |

## CLI use

### Step 1: fit on cohort

```bash
apl fit /path/to/features
```

This fits APL on every slide in the folder, scores every patch and writes to `apl_output/`:

| file               | content                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `patch_scores.csv` | one row per patch: `slide`, `x`, `y`, `margin`, `classifier`, `flagged` and the intermediate scores |
| `model.npz`        | the fitted model, for step 2                                                              |
| `run.json`         | version, date, options, every setting, and a summary (candidates, prototypes, ...)      |

Options:

| option               | what it does                                                                          |
| -------------------- | ------------------------------------------------------------------------------------- |
| `--out DIR`          | output folder (default `apl_output`)                                                  |
| `--holdout CSV`      | csv with a `slide` column: those slides are scored but never used for fitting         |
| `--cache PATH`       | save the features once (e.g. `cache/uni`); later runs load in seconds                 |
| `--type-dist X`      | where the candidates are split into artefact types (default 0.3), see [Settings](#settings) |
| `--not-artefact K..` | artefact types that are really tissue, e.g. `--not-artefact 5`, see [Settings](#settings)   |
| `--flag-fraction F`  | share of fitting patches flagged as artefact, sets the cut-off (default 0.1)          |
| `--seed N`           | random seed (default 0)                                                               |
| `--plots`            | also save the candidate dendrogram (`dendrogram.png`)                                 |
| `--wsi-dir DIR`      | with `--plots`: also save the patches most similar to each prototype (`prototypes.png`) |
| `--patch-size`, `--mag` | with `--wsi-dir`: patch size in px (default 512) and magnification (default 20)    |

For example:

```bash
apl fit /path/to/features --out results --holdout test_slides.csv --cache cache/uni --plots
```

### Step 2 (optional): score new slides

Score slides that were not part of the fit, with features from the **same encoder**:

```bash
apl score results/model.npz /path/to/new_features --out new_scores.csv
```

`new_scores.csv` has one row per patch: `slide`, `x`, `y`, `margin`, `nearest_art`, `classifier`, `flagged` (same cut-off as the fit).

Run `apl fit --help` or `apl score --help` for all options.

## Python

```python
from artefact_prototype_learning import fit, load_features, make_settings, save_model

X, meta = load_features("path/to/features") # L2-normalised embeddings + patch table
model, table = fit(X, meta, make_settings())
save_model(model, "model.npz")
```

You can add your own candidate scores: a function of `(X, norms, ref)` returning one value per patch, larger = more
unusual. It counts towards the candidates like the built-in scores:

```python
model, table = fit(X, meta, cfg, custom={"my_score": my_score_function})
```

### Looking at the results

`visualisation` has a few plots (patch images and thumbnails need `pip install ".[slides]"`):

```python
from artefact_prototype_learning import visualisation as viz

viz.patch_grid(meta, np.where(table["candidate"])[0][:32], wsi_dir) # any patches, e.g. the candidates
viz.prototype_patches(X, meta, model, wsi_dir) # the patches nearest each prototype
viz.slide_heatmap(table, "slide_1", "classifier", wsi_dir) # artefact scores
```

## Citation

If you use APL, please consider citing our package (see Zenodo DOI: ). A paper describing the method is in preparation.

## Licence

MIT, see [LICENSE](LICENSE)