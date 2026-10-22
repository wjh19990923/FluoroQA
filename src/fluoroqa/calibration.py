"""Threshold fit and evaluation are separate; larger scores mean failure."""
import numpy as np
from scipy.stats import rankdata


def _arrays(scores, mtre_mm):
    scores, errors = np.asarray(scores, dtype=float), np.asarray(mtre_mm, dtype=float)
    if scores.ndim != 1 or scores.shape != errors.shape or not len(scores):
        raise ValueError("Expected equal nonempty vectors")
    if not np.isfinite(scores).all() or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Invalid scores or errors")
    return scores, errors >= 2


def fit_threshold(scores, mtre_mm, *, split, mode="raw"):
    """Max balanced accuracy, largest threshold on a tie.

    Raw: train only. Local: val only, matching the separate local protocol.
    The caller must provide genuine split provenance; a string is not proof.
    """
    if mode not in ("raw", "local") or split != ("train" if mode == "raw" else "val"):
        raise ValueError("Raw requires train; Local requires val")
    scores, failed = _arrays(scores, mtre_mm)
    if not failed.any() or failed.all():
        raise ValueError("Threshold fitting requires both classes")
    thresholds = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    ba = np.array([((scores[failed] >= t).mean() + (scores[~failed] < t).mean())/2 for t in thresholds])
    return float(thresholds[np.flatnonzero(np.abs(ba-ba.max()) <= 1e-12)[-1]])


def evaluate(scores, mtre_mm, threshold):
    scores, failed = _arrays(scores, mtre_mm)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    predicted = scores >= threshold
    recall = float(predicted[failed].mean()) if failed.any() else None
    specificity = float((~predicted[~failed]).mean()) if (~failed).any() else None
    n, m = int(failed.sum()), int((~failed).sum())
    auc = float((rankdata(scores)[failed].sum()-n*(n+1)/2)/(n*m)) if n and m else None
    return dict(n=len(scores), accuracy=float((predicted == failed).mean()),
                failure_recall=recall, success_specificity=specificity,
                balanced_accuracy=(recall+specificity)/2 if n and m else None, auroc=auc)


def local_score(center, neighbors, mode="mean"):
    """Neighbors must be rendered around the CURRENT pose, never GT.

    Population standard deviation is used for z scoring. Geometry and renderer
    are the caller's responsibility; scores cannot verify their provenance.
    """
    center, neighbors = np.asarray(center, dtype=float), np.asarray(neighbors, dtype=float)
    if neighbors.shape != center.shape + (12,) or mode not in ("mean", "z"):
        raise ValueError("Expected exactly 12 neighbors per center; mean or z mode")
    if not np.isfinite(center).all() or not np.isfinite(neighbors).all():
        raise ValueError("Invalid local score input")
    value = center - neighbors.mean(axis=-1)
    std = neighbors.std(axis=-1)
    return value if mode == "mean" else value / np.where(std > 0, std, 1.)
