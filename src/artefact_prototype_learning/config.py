"""Method settings: the defaults below, which work as they are, and make_settings to change a few."""
import copy

DEFAULTS = {
    "seed": 0,
    "flag_fraction": 0.1,        # share of fitting patches flagged as artefact: sets the classifier cut-off
    "scoring": {
        "sample_n": 40000,       # reference sample for Mahalanobis and kNN density (assumed mostly tissue)
        "mahal_dim": 50,         # PCA subspace for Mahalanobis
        "knn": 20,               # neighbours for local density
        "tail_pctl": 99.0,       # a score flags a patch above this percentile
        "agree_min": 2,          # candidate if >= this many scores flag it
        "max_candidates": 400,   # keep the most extreme (by mean rank)
    },
    "prototypes": {
        "type_dist": 0.3,        # cosine cut on the candidates' dendrogram; one artefact prototype per cluster
        "min_type_size": 5,      # drop smaller clusters
        "n_tissue": 8,           # k-means tissue prototypes
        "tissue_core_n": 20000,  # patches sampled from the central half (Mahalanobis <= median) for k-means
        "not_artefact": [],      # artefact types that are really tissue after looking at them (moved to tissue)
    },
    "learning": {
        "tau": 0.06,             # margin (best artefact - best tissue similarity) separating the pseudo-labels
        "gap": 0.04,             # pseudo-labels skip the uncertain band tau +- gap
        "art_cap": 3000,         # max extra pseudo-labelled patches per artefact type
        "tissue_n": 20000,       # pseudo-labelled tissue patches
        "C": 1.0,                # inverse regularisation strength of the logistic classifier
    },
}


def make_settings(**changes) -> dict:
    """The default settings with your changes, e.g. make_settings(prototypes={"type_dist": 0.25}).

    Unknown settings raise an error, so a typo is never silently ignored.
    """
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in changes.items():
        if key not in cfg:
            raise KeyError(f"unknown setting {key!r}; known settings: {sorted(cfg)}")
        if isinstance(cfg[key], dict):
            unknown = set(value) - set(cfg[key])
            if unknown:
                raise KeyError(f"unknown {key} setting(s) {sorted(unknown)}; known: {sorted(cfg[key])}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg