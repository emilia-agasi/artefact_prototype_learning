"""Artefact prototype learning (APL): unsupervised artefact detection in histopathology patch embeddings."""
from .config import make_settings
from .load_patch_features import load_features
from .pipeline import fit, load_model, save_model, score

__version__ = "0.1.0"
__all__ = ["fit", "load_features", "load_model", "make_settings", "save_model", "score"]