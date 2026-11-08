"""Data loading, preprocessing, and evaluation for Beijing, Chengdu, and Xi'an."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

import poi_alignment as poi_tools


CITIES = ("bj", "cd", "xa")
TASKS = ("population", "gdp", "co2")
TASK_FILES = {
    "population": "population_sum.npy",
    "gdp": "gdp_sum.npy",
    "co2": "co2_sum.npy",
}
DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
POI_CATEGORY_DIM = 162


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cpu" or torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_labels(data_path):
    labels = {}
    for task, filename in TASK_FILES.items():
        path = data_path.parent.parent / "tasks" / data_path.name / filename
        if path.exists():
            labels[task] = np.load(path).astype(np.float32).reshape(-1)
    return labels


def city_data_path(city, root=None):
    if city not in CITIES:
        raise ValueError("Unknown city {}".format(city))
    return (Path(root) / "dataset" if root is not None else DATASET_ROOT) / city


def load_unaligned_poi(city, root=None):
    """Load and validate the released anonymous POI matrix without alignment."""
    path = city_data_path(city, root) / "poi_f.npy"
    raw = np.load(path).astype(np.float32)
    if raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError("{} has unsupported shape {}".format(path, raw.shape))
    if np.any(raw < 0) or not np.isfinite(raw).all():
        raise ValueError("{} contains invalid POI values".format(path))
    return raw


def legacy_poi_alignment(output_dim=POI_CATEGORY_DIM, source_cities=()):
    return poi_tools.make_frequency_alignment(output_dim, source_cities)


def fit_poi_alignment(cities, root, mode="frequency_rank", output_dim=None, **kwargs):
    if mode != "frequency_rank":
        raise ValueError("This release supports frequency_rank POI alignment")
    return legacy_poi_alignment(
        POI_CATEGORY_DIM if output_dim is None else int(output_dim), list(cities)
    )


def poi_alignment_summary(alignment):
    return poi_tools.alignment_summary(alignment)


def checkpoint_poi_alignment(checkpoint):
    """Read alignment state while retaining compatibility with old artifacts."""
    alignment = checkpoint.get("poi_alignment")
    if alignment is not None:
        return alignment
    return legacy_poi_alignment(
        checkpoint.get("model_config", {}).get("poi_category_dim", POI_CATEGORY_DIM),
        checkpoint.get("source_cities", ()),
    )


def apply_poi_alignment(poi, alignment, return_diagnostics=False):
    return poi_tools.canonicalize_poi(
        poi, alignment, return_diagnostics=return_diagnostics
    )


def load_raw_poi(
    city,
    root=None,
    poi_alignment=None,
    return_diagnostics=False,
):
    """Load a taxonomy-free POI matrix in a fixed canonical feature space."""
    raw = load_unaligned_poi(city, root)
    if poi_alignment is None:
        poi_alignment = legacy_poi_alignment()
    return apply_poi_alignment(raw, poi_alignment, return_diagnostics)


def compute_poi_statistics(cities, root, poi_alignment=None):
    totals = []
    for city in cities:
        poi = load_raw_poi(city, root, poi_alignment)
        totals.append(np.log1p(poi.sum(axis=1)))
    values = np.concatenate(totals)
    return {
        "log_total_mean": float(values.mean()),
        "log_total_std": float(max(values.std(), 1e-6)),
    }


def poi_arrays(poi, statistics):
    totals = np.maximum(poi.sum(axis=1, keepdims=True), 1e-9)
    composition = poi / totals
    log_total = np.log1p(totals)
    log_categories = np.log1p(poi)
    scale = (log_total - float(statistics["log_total_mean"])) / float(
        statistics["log_total_std"]
    )
    features = np.concatenate(
        [
            composition,
            scale,
            log_categories / 4.0,
        ],
        axis=1,
    )
    return composition.astype(np.float32), features.astype(np.float32)


def poi_city_context(poi):
    """City-wide category spectrum available from POI input alone."""
    mass = np.asarray(poi, dtype=np.float64).sum(axis=0)
    context = mass / max(float(mass.sum()), 1e-12)
    return context.astype(np.float32)


def row_normalize(matrix, eps=1e-9):
    denominator = np.maximum(matrix.sum(axis=1, keepdims=True), eps)
    return (matrix / denominator).astype(np.float32)


def column_standardize(values, eps=1e-6):
    values = np.asarray(values, dtype=np.float32)
    return (
        (values - values.mean(axis=0, keepdims=True))
        / np.maximum(values.std(axis=0, keepdims=True), eps)
    ).astype(np.float32)


def normalized_entropy(probabilities, eps=1e-9):
    n = probabilities.shape[1]
    entropy = -(probabilities * np.log(np.maximum(probabilities, eps))).sum(axis=1)
    return (entropy / max(np.log(max(n, 2)), eps)).astype(np.float32)


def mobility_arrays(mobility):
    mobility = np.asarray(mobility, dtype=np.float32).copy()
    if mobility.ndim != 2 or mobility.shape[0] != mobility.shape[1]:
        raise ValueError("Mobility must be square, got {}".format(mobility.shape))
    if np.any(mobility < 0) or not np.isfinite(mobility).all():
        raise ValueError("Mobility contains invalid values")

    np.fill_diagonal(mobility, 0.0)
    out_mass = mobility.sum(axis=1)
    in_mass = mobility.sum(axis=0)
    total = float(max(mobility.sum(), 1e-9))
    a_out = row_normalize(mobility)
    a_in = row_normalize(mobility.T)

    log_mass = column_standardize(
        np.column_stack([np.log1p(out_mass), np.log1p(in_mass)])
    )
    out_entropy = normalized_entropy(a_out)
    in_entropy = normalized_entropy(a_in)
    out_density = (mobility > 0).mean(axis=1).astype(np.float32)
    in_density = (mobility > 0).mean(axis=0).astype(np.float32)
    dynamic = np.column_stack(
        [
            log_mass[:, 0],
            log_mass[:, 1],
            out_entropy,
            in_entropy,
            out_density,
            in_density,
        ]
    ).astype(np.float32)

    expected = np.outer(out_mass, in_mass) / total
    residual = np.tanh((np.log1p(mobility) - np.log1p(expected)) / 2.0).astype(
        np.float32
    )
    np.fill_diagonal(residual, 0.0)

    return {
        "mobility_np": mobility,
        "a_out_np": a_out,
        "a_in_np": a_in,
        "dynamic_np": dynamic,
        "residual_np": residual,
    }


def load_source_city(
    city,
    root,
    device,
    poi_statistics,
    poi_alignment=None,
    load_labels=False,
):
    """Load source POI and mobility; task labels are optional and disabled in training."""
    data_path = city_data_path(city, root)
    poi, alignment_diagnostics = load_raw_poi(
        city, root, poi_alignment, return_diagnostics=True
    )
    composition, features = poi_arrays(poi, poi_statistics)
    city_context = poi_city_context(poi)
    source = np.load(data_path / "source_matrix_norm.npy").astype(np.float32)
    source_sum = np.load(data_path / "source_sum.npy").astype(np.float32)
    mobility = source * source_sum
    if mobility.shape != (len(poi), len(poi)):
        raise ValueError(
            "{} POI/OD mismatch: {} versus {}".format(city, poi.shape, mobility.shape)
        )
    movement = mobility_arrays(mobility)
    labels = _load_labels(data_path) if load_labels else {}

    result = {
        "city": city,
        "N": len(poi),
        "labels": labels,
        "poi_raw_np": poi,
        "poi_composition_np": composition,
        "poi_features_np": features,
        "poi_city_context_np": city_context,
        "poi_alignment_diagnostics": alignment_diagnostics,
        "data_access": {
            "representation_input": [
                "poi_f.npy",
                "source_matrix_norm.npy",
                "source_sum.npy",
            ],
            "downstream_evaluation": list(TASK_FILES[task] for task in labels),
        },
        "poi_composition": torch.tensor(composition, device=device),
        "poi_features": torch.tensor(features, device=device),
        "poi_city_context": torch.tensor(city_context, device=device),
    }
    result.update(movement)
    for key in ("a_out", "a_in", "dynamic", "residual"):
        result[key] = torch.tensor(result[key + "_np"], device=device)
    return result


def load_target_city(
    city,
    root,
    device,
    poi_statistics,
    poi_alignment=None,
    load_labels=True,
):
    """Load target POI and labels without opening any target mobility file."""
    data_path = city_data_path(city, root)
    poi, alignment_diagnostics = load_raw_poi(
        city, root, poi_alignment, return_diagnostics=True
    )
    composition, features = poi_arrays(poi, poi_statistics)
    city_context = poi_city_context(poi)
    labels = _load_labels(data_path) if load_labels else {}
    return {
        "city": city,
        "N": len(poi),
        "labels": labels,
        "poi_raw_np": poi,
        "poi_composition_np": composition,
        "poi_features_np": features,
        "poi_city_context_np": city_context,
        "poi_alignment_diagnostics": alignment_diagnostics,
        "poi_composition": torch.tensor(composition, device=device),
        "poi_features": torch.tensor(features, device=device),
        "poi_city_context": torch.tensor(city_context, device=device),
        "data_access": {
            "representation_input": ["poi_f.npy"],
            "downstream_evaluation": list(TASK_FILES[task] for task in labels),
        },
    }


def compute_poi_initializers(source_data, role_count, seed):
    all_composition = np.concatenate(
        [data["poi_composition_np"] for data in source_data.values()],
        axis=0,
    )
    model = MiniBatchKMeans(
        n_clusters=role_count,
        random_state=seed,
        n_init=10,
        batch_size=max(64, role_count * 8),
    )
    return model.fit(all_composition).cluster_centers_.astype(np.float32)


def split_region_masks(source_data, validation_fraction, seed, device):
    generator = np.random.RandomState(seed)
    masks = {}
    for city, data in source_data.items():
        order = generator.permutation(data["N"])
        n_validation = max(1, int(round(data["N"] * validation_fraction)))
        validation_index = order[:n_validation]
        train = np.ones(data["N"], dtype=bool)
        train[validation_index] = False
        masks[city] = {
            "train": torch.tensor(train, device=device),
            "validation": torch.tensor(~train, device=device),
        }
    return masks


def downstream_metrics(embedding, label):
    """Ordered five-fold Ridge(alpha=1), with nonnegative predictions."""
    splitter = KFold(n_splits=5)
    predictions = []
    truths = []
    for train_index, test_index in splitter.split(embedding):
        model = Ridge(alpha=1.0)
        model.fit(embedding[train_index], label[train_index])
        prediction = model.predict(embedding[test_index])
        prediction[prediction < 0] = 0
        predictions.append(prediction)
        truths.append(label[test_index])
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
    }


def evaluate_embedding(city, embedding, labels):
    results = {}
    for task in TASKS:
        if task not in labels:
            continue
        metrics = downstream_metrics(embedding, labels[task])
        if not np.isfinite(metrics["r2"]):
            continue
        results[task] = metrics
    return results


def mean_r2(metrics):
    values = [item["r2"] for item in metrics.values()]
    return float(np.mean(values)) if values else float("nan")


def representation_diagnostics(embedding, role_outputs, grammar=None):
    embedding = np.asarray(embedding, dtype=np.float64)
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    variance_sum = float(max(variance.sum(), 1e-12))
    effective_rank = float(
        variance_sum**2 / max(float(np.square(variance).sum()), 1e-12)
    )
    diagnostics = {
        "embedding_effective_rank": effective_rank,
        "embedding_top1_ratio": float(variance[0] / variance_sum),
        "embedding_top10_ratio": float(
            variance[: min(10, len(variance))].sum() / variance_sum
        ),
        "embedding_feature_std_mean": float(centered.std(axis=0).mean()),
    }
    for name, assignments in role_outputs.items():
        assignments = np.asarray(assignments, dtype=np.float64)
        normalized_entropy = -(
            assignments * np.log(np.maximum(assignments, 1e-12))
        ).sum(axis=1).mean() / np.log(assignments.shape[1])
        usage = assignments.mean(axis=0)
        diagnostics["{}_entropy".format(name)] = float(normalized_entropy)
        diagnostics["{}_usage_min".format(name)] = float(usage.min())
        diagnostics["{}_usage_max".format(name)] = float(usage.max())
    if "outgoing" in role_outputs and "incoming" in role_outputs:
        outgoing = np.asarray(role_outputs["outgoing"], dtype=np.float64)
        incoming = np.asarray(role_outputs["incoming"], dtype=np.float64)
        diagnostics["outgoing_incoming_l1"] = float(np.abs(outgoing - incoming).mean())
    if "semantic" in role_outputs and "outgoing" in role_outputs:
        semantic = np.asarray(role_outputs["semantic"], dtype=np.float64)
        outgoing = np.asarray(role_outputs["outgoing"], dtype=np.float64)
        diagnostics["outgoing_semantic_l1"] = float(np.abs(outgoing - semantic).mean())
    if "semantic" in role_outputs and "incoming" in role_outputs:
        semantic = np.asarray(role_outputs["semantic"], dtype=np.float64)
        incoming = np.asarray(role_outputs["incoming"], dtype=np.float64)
        diagnostics["incoming_semantic_l1"] = float(np.abs(incoming - semantic).mean())
    if grammar is not None:
        grammar = np.asarray(grammar, dtype=np.float64)
        grammar_norm = max(float(np.linalg.norm(grammar)), 1e-12)
        diagnostics.update(
            {
                "grammar_std": float(grammar.std()),
                "grammar_abs_mean": float(np.abs(grammar).mean()),
                "grammar_rank_1e3": int(np.linalg.matrix_rank(grammar, tol=1e-3)),
                "grammar_asymmetry_ratio": float(
                    np.linalg.norm(grammar - grammar.T) / grammar_norm
                ),
            }
        )
    return diagnostics
