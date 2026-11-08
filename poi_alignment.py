"""POI frequency ranking and zero padding used by SIFT."""

import numpy as np

ALIGNMENT_VERSION = "SIFT-frequency-rank"


def _validate_poi(poi):
    matrix = np.asarray(poi, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("POI matrix must be two-dimensional and non-empty")
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise ValueError("POI matrix contains invalid values")
    return matrix


def frequency_rank(poi, output_dim):
    """Reproduce the original stable frequency ranking exactly."""
    matrix = _validate_poi(poi)
    output_dim = int(output_dim)
    if matrix.shape[1] > output_dim:
        raise ValueError(
            "POI width {} exceeds output_dim {}".format(matrix.shape[1], output_dim)
        )
    order = np.argsort(-matrix.sum(axis=0), kind="stable")
    aligned = matrix[:, order]
    if aligned.shape[1] < output_dim:
        aligned = np.pad(
            aligned,
            ((0, 0), (0, output_dim - aligned.shape[1])),
            mode="constant",
        )
    return aligned.astype(np.float32)


def make_frequency_alignment(output_dim, source_cities=()):
    return {
        "version": ALIGNMENT_VERSION,
        "mode": "frequency_rank",
        "output_dim": int(output_dim),
        "source_cities": list(source_cities),
    }


def canonicalize_poi(poi, alignment, return_diagnostics=False):
    matrix = _validate_poi(poi)
    if alignment.get("mode") != "frequency_rank":
        raise ValueError("This release supports frequency_rank POI alignment")
    canonical = frequency_rank(matrix, alignment["output_dim"])
    diagnostics = {
        "input_dim": int(matrix.shape[1]),
        "output_dim": int(canonical.shape[1]),
        "mass_relative_error": float(
            abs(float(canonical.sum()) - float(matrix.sum()))
            / max(float(matrix.sum()), 1e-12)
        ),
    }
    return (canonical, diagnostics) if return_diagnostics else canonical


def alignment_summary(alignment):
    return dict(alignment)
