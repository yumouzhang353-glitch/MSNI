from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np

SCHEMA = "SNI_COMPACT_QUADRATIC_V1"


def decompose_centered(centered, normalized_weights, *, spectral_tolerance, ridge=0.1):
    centered = np.asarray(centered, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    if centered.ndim != 2 or min(centered.shape) < 1:
        raise ValueError("Expected a nonempty k by d centred cloud")
    if weights.shape != (len(centered),):
        raise ValueError("Weight/cloud shapes differ")
    if not np.isfinite(centered).all() or not np.isfinite(weights).all():
        raise ValueError("Nonfinite cloud or weights")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=0.0):
        raise ValueError("Weights must be nonnegative and already normalized")
    tau = float(spectral_tolerance)
    ridge = float(ridge)
    if not np.isfinite(tau) or not 0.0 <= tau < 1.0:
        raise ValueError("Invalid statistical spectral tolerance")
    if not np.isfinite(ridge) or not 0.0 < ridge <= 1.0:
        raise ValueError("Invalid inverse ridge")
    ambient = centered.shape[1]
    weighted = np.sqrt(weights)[:, None] * centered
    if not np.any(weighted):
        values = np.empty(0, dtype=np.float64)
        basis = np.empty((0, ambient), dtype=np.float64)
    else:
        (_, singular, basis) = np.linalg.svd(weighted, full_matrices=False)
        values = singular * singular
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite covariance spectrum")
    eigenvalues = np.zeros(ambient, dtype=np.float64)
    if len(values):
        eigenvalues[-len(values) :] = values[::-1]
    largest = float(eigenvalues[-1])
    positive = eigenvalues > tau * largest
    kept = np.where(positive, eigenvalues, 0.0)
    total = float(kept.sum())
    squares = float((kept * kept).sum())
    effective_dimension = total * total / squares if squares > 0.0 else 1.0
    spectral_rank = int(positive.sum())
    confidence = (
        float(np.clip(1.0 - effective_dimension / spectral_rank, 0.0, 1.0))
        if spectral_rank
        else 0.0
    )
    scale = total / effective_dimension if total > 0.0 else 1.0
    floor = ridge * largest
    isotropic = scale / floor if floor > 0.0 else 1.0
    regularized = np.maximum(values, floor)
    ratios = np.divide(scale, regularized, out=np.ones_like(regularized), where=regularized > 0.0)
    correction = ratios - isotropic
    statistics = np.asarray([effective_dimension, confidence, scale, isotropic, total])
    if (
        not np.isfinite(statistics).all()
        or not np.isfinite(correction).all()
        or scale <= 0.0
        or (isotropic <= 0.0)
    ):
        raise FloatingPointError("Invalid spectral statistics or inverse scale")
    return dict(
        ambient_dimension=ambient,
        basis=basis,
        basis_eigenvalues=values,
        inverse_correction=correction,
        isotropic_inverse=float(isotropic),
        eigenvalues=eigenvalues,
        largest_eigenvalue=largest,
        effective_dimension=float(effective_dimension),
        spectral_rank=spectral_rank,
        confidence=confidence,
        scale=float(scale),
        positive_spectral_mass=total,
        query_residual_defined=bool(total > 0.0),
        spectral_tolerance=tau,
        ridge=ridge,
    )


def fit_weighted_node(points, weights, *, spectral_tolerance, ridge=0.1):
    points = np.asarray(points, dtype=np.float64)
    raw = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or min(points.shape) < 1 or raw.shape != (len(points),):
        raise ValueError("Invalid neighbour/weight shapes")
    if not np.isfinite(points).all() or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid neighbour coordinates or weights")
    mass = float(raw.sum())
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Zero normalization mass is not the zero-spectrum branch")
    normalized = raw / mass
    mean = np.einsum("nk,nkd->nd", normalized[None, :], points[None, :, :])[0]
    result = decompose_centered(
        points - mean, normalized, spectral_tolerance=spectral_tolerance, ridge=ridge
    )
    result.update(mean=mean, normalized_weights=normalized)
    return result


def local_quadratic(node: Mapping, vectors, *, inverse=True):
    vectors = np.asarray(vectors, dtype=np.float64)
    ambient = int(node["ambient_dimension"])
    if vectors.ndim != 2 or vectors.shape[1] != ambient or (not np.isfinite(vectors).all()):
        raise ValueError("Expected finite m by d query vectors")
    basis = np.asarray(node["basis"], dtype=np.float64)
    coefficients = np.asarray(
        node["inverse_correction"] if inverse else node["basis_eigenvalues"], dtype=np.float64
    )
    if basis.shape != (len(coefficients), ambient):
        raise ValueError("Local spectral representation has inconsistent shapes")
    projection = vectors @ basis.T
    value = projection * projection @ coefficients
    if inverse:
        value += float(node["isotropic_inverse"]) * np.einsum("mi,mi->m", vectors, vectors)
    if not np.isfinite(value).all():
        raise FloatingPointError("Nonfinite compact quadratic form")
    return value


def pack(nodes: Sequence[Mapping]):
    if not nodes:
        raise ValueError("Expected at least one node")
    ambient = int(nodes[0]["ambient_dimension"])
    if any((int(node["ambient_dimension"]) != ambient for node in nodes)):
        raise ValueError("Mixed ambient dimensions")
    counts = np.asarray([len(node["basis"]) for node in nodes], dtype=np.int64)
    result = dict(
        schema=SCHEMA,
        ambient_dimension=ambient,
        n_nodes=len(nodes),
        indptr=np.r_[0, np.cumsum(counts)].astype(np.int64),
        basis=np.concatenate([node["basis"] for node in nodes], axis=0),
        basis_eigenvalues=np.concatenate([node["basis_eigenvalues"] for node in nodes]),
        inverse_correction=np.concatenate([node["inverse_correction"] for node in nodes]),
        isotropic_inverse=np.asarray(
            [node["isotropic_inverse"] for node in nodes], dtype=np.float64
        ),
        scale=np.asarray([node["scale"] for node in nodes], dtype=np.float64),
        ridge=np.asarray([node["ridge"] for node in nodes], dtype=np.float64),
        largest_eigenvalue=np.asarray(
            [node["largest_eigenvalue"] for node in nodes], dtype=np.float64
        ),
    )
    validate(result)
    return result


def validate(operator: Mapping):
    if operator.get("schema") != SCHEMA:
        raise ValueError("Unknown compact operator schema")
    (ambient, count) = (operator["ambient_dimension"], operator["n_nodes"])
    if not isinstance(ambient, (int, np.integer)) or not isinstance(count, (int, np.integer)):
        raise ValueError("Dimensions must be integers")
    if ambient < 1 or count < 1:
        raise ValueError("Invalid packed dimensions")
    ptr = np.asarray(operator["indptr"])
    basis = np.asarray(operator["basis"])
    values = np.asarray(operator["basis_eigenvalues"])
    correction = np.asarray(operator["inverse_correction"])
    isotropic = np.asarray(operator["isotropic_inverse"])
    scale = np.asarray(operator["scale"])
    ridge = np.asarray(operator["ridge"])
    largest = np.asarray(operator["largest_eigenvalue"])
    if ptr.dtype.kind not in "iu" or ptr.shape != (count + 1,) or ptr[0] != 0:
        raise ValueError("Invalid node pointers")
    if np.any(ptr[1:] < ptr[:-1]) or ptr[-1] != len(basis):
        raise ValueError("Nonmonotone or incomplete node pointers")
    if (
        basis.shape != (int(ptr[-1]), ambient)
        or values.shape != (len(basis),)
        or correction.shape != values.shape
    ):
        raise ValueError("Invalid packed spectral shapes")
    if any((array.shape != (count,) for array in (isotropic, scale, ridge, largest))):
        raise ValueError("Invalid per-node scalar arrays")
    for array in (basis, values, correction, isotropic, scale, ridge, largest):
        if array.dtype != np.dtype("float64") or not np.isfinite(array).all():
            raise ValueError("Expected finite float64 spectral arrays")
    if (values < 0.0).any() or (isotropic <= 0.0).any():
        raise ValueError("Invalid covariance or inverse spectrum")
    if (scale <= 0.0).any() or (ridge <= 0.0).any() or (ridge > 1.0).any() or (largest < 0.0).any():
        raise ValueError("Invalid inverse regularization parameters")
    floor = ridge * largest
    expected_isotropic = np.divide(scale, floor, out=np.ones_like(scale), where=floor > 0.0)
    hosts = np.repeat(np.arange(count, dtype=np.int64), np.diff(ptr).astype(np.int64))
    regularized = np.maximum(values, floor[hosts])
    ratios = np.divide(
        scale[hosts], regularized, out=np.ones_like(regularized), where=regularized > 0.0
    )
    expected_correction = ratios - expected_isotropic[hosts]
    if not np.array_equal(isotropic, expected_isotropic) or not np.array_equal(
        correction, expected_correction
    ):
        raise ValueError("Persisted inverse coefficients disagree with ridge and scale")
    return True


def quadratic(operator: Mapping, node_ids, vectors, *, inverse=True):
    validate(operator)
    vectors = np.asarray(vectors, dtype=np.float64)
    ids = np.asarray(node_ids)
    if ids.ndim == 1 and (not len(ids)):
        ids = np.empty(0, dtype=np.int64)
    (ambient, count) = (int(operator["ambient_dimension"]), int(operator["n_nodes"]))
    if ids.dtype.kind not in "iu" or ids.ndim != 1 or vectors.shape != (len(ids), ambient):
        raise ValueError("Invalid endpoint/vector shapes")
    if not np.isfinite(vectors).all() or np.any(ids < 0) or np.any(ids >= count):
        raise ValueError("Invalid endpoint index or vector")
    output = np.empty(len(ids), dtype=np.float64)
    if not len(ids):
        return output
    ptr = np.asarray(operator["indptr"])
    basis = np.asarray(operator["basis"])
    coefficients = np.asarray(
        operator["inverse_correction"] if inverse else operator["basis_eigenvalues"]
    )
    isotropic = np.asarray(operator["isotropic_inverse"])
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    stops = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1, len(ids)]
    for start, stop in zip(stops[:-1], stops[1:]):
        positions = order[start:stop]
        node = int(sorted_ids[start])
        (left, right) = (int(ptr[node]), int(ptr[node + 1]))
        query = vectors[positions]
        projection = query @ basis[left:right].T
        value = projection * projection @ coefficients[left:right]
        if inverse:
            value += isotropic[node] * np.einsum("mi,mi->m", query, query)
        output[positions] = value
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite packed quadratic form")
    return output
