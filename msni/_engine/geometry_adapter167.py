import hashlib
import math
import numpy as np
from .sni_single_native._config import EPS, PROTOCOL
from .sni_single_native._tie_neighborhood import RankNeighborhoods
from .geometry_operator167 import decompose_centered, local_quadratic, pack, quadratic, validate

GEOMETRY_SCHEMA = "SNI_CLOSED_COMPACT_GEOMETRY_V2"


def _digest(array):
    array = np.asarray(array)
    return dict(
        shape=list(array.shape),
        dtype=array.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    )


def _operator_identity(operator):
    keys = (
        "indptr",
        "basis",
        "basis_eigenvalues",
        "inverse_correction",
        "isotropic_inverse",
        "scale",
        "ridge",
        "largest_eigenvalue",
    )
    return dict(
        schema=operator["schema"],
        n=operator["n_nodes"],
        dimension=operator["ambient_dimension"],
        arrays={key: _digest(operator[key]) for key in keys},
    )


def _alignment(
    X,
    rank,
    ns,
    valid,
    weights,
    means,
    scales,
    dimensions,
    confidence,
    eigenvalues,
    spectral_ranks,
    residual,
    defined,
    operator,
):
    return dict(
        schema=GEOMETRY_SCHEMA,
        n=len(X),
        dimension=X.shape[1],
        rank=int(rank),
        X=_digest(X),
        neighbors=_digest(ns),
        valid=_digest(valid),
        weights=_digest(weights),
        means=_digest(means),
        scales=_digest(scales),
        dimensions=_digest(dimensions),
        confidence=_digest(confidence),
        eigenvalues=_digest(eigenvalues),
        spectral_ranks=_digest(spectral_ranks),
        query_residual=_digest(residual),
        query_residual_defined=_digest(defined),
        operator=_operator_identity(operator),
    )


def require_canonical_pairs(pairs, n):
    pairs = np.asarray(pairs)
    if pairs.dtype.kind not in "iu" or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Expected an integer E by 2 edge table")
    if (pairs < 0).any() or (pairs >= n).any() or (pairs[:, 0] >= pairs[:, 1]).any():
        raise ValueError("Fixed path requires canonical undirected endpoints i<j")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate undirected edge")
    return True


def prepare(X, r, reliability, weighted=False, tie_closed=False):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 2 or X.shape[1] < 1 or (not np.isfinite(X).all()):
        raise ValueError("Expected finite n by d geometry coordinates, n>=2")
    if not tie_closed:
        raise ValueError("Compact adapter only supports the fixed closed-shell route")
    (n, d) = X.shape
    if weighted:
        reliability = np.asarray(reliability, dtype=np.float64)
        if (
            reliability.shape != (n,)
            or not np.isfinite(reliability).all()
            or (reliability < 0).any()
        ):
            raise ValueError("Invalid neighbour reliability")
    else:
        reliability = np.ones(n, dtype=np.float64)
    ranks = sorted(set((min(n - 1, max(3, int(r * factor))) for factor in (1, 2))))
    ranked = RankNeighborhoods(X, max(ranks))
    closed = [ranked.rows(rank) for rank in ranks]
    width = max((len(row) for rows in closed for row in rows))
    outputs = {
        key: []
        for key in (
            "means",
            "weights",
            "dimensions",
            "confidence_per_rank",
            "eigenvalues",
            "spectral_ranks",
            "scales",
            "query_residual",
            "query_residual_defined",
            "actual_counts",
        )
    }
    (operators, neighborhoods, masks, alignments) = ([], [], [], [])
    tau = PROTOCOL["neighbor_geometry"]["spectral_tolerance"]
    for rank, rows in zip(ranks, closed):
        ns = np.broadcast_to(np.arange(n)[:, None], (n, width)).copy()
        valid = np.zeros((n, width), dtype=bool)
        for i, row in enumerate(rows):
            ns[i, : len(row)] = row
            valid[i, : len(row)] = True
        weights = reliability[ns] * valid
        sums = weights.sum(axis=1, keepdims=True)
        if (sums <= 0.0).any():
            raise ValueError("Zero normalization mass is outside the frozen geometry domain")
        weights = weights / sums
        means = np.empty((n, d), dtype=np.float64)
        dimensions = np.empty(n, dtype=np.float64)
        confidence = np.empty(n, dtype=np.float64)
        eigenvalues = np.empty((n, d), dtype=np.float64)
        spectral_ranks = np.empty(n, dtype=np.int64)
        scales = np.empty(n, dtype=np.float64)
        residual = np.zeros(n, dtype=np.float64)
        defined = np.empty(n, dtype=bool)
        nodes = []
        for i in range(n):
            points = X[ns[i]]
            mean = np.einsum("nk,nkd->nd", weights[i : i + 1], points[None, :, :])[0]
            node = decompose_centered(points - mean, weights[i], spectral_tolerance=tau, ridge=0.1)
            means[i] = mean
            dimensions[i] = node["effective_dimension"]
            confidence[i] = node["confidence"]
            eigenvalues[i] = node["eigenvalues"]
            spectral_ranks[i] = node["spectral_rank"]
            scales[i] = node["scale"]
            defined[i] = node["query_residual_defined"]
            if defined[i]:
                residual[i] = local_quadratic(node, (X[i] - mean)[None, :])[0] / scales[i]
            nodes.append(node)
        operator = pack(nodes)
        del nodes
        alignment = _alignment(
            X,
            rank,
            ns,
            valid,
            weights,
            means,
            scales,
            dimensions,
            confidence,
            eigenvalues,
            spectral_ranks,
            residual,
            defined,
            operator,
        )
        operator["geometry_alignment"] = alignment
        values = (
            means,
            weights,
            dimensions,
            confidence,
            eigenvalues,
            spectral_ranks,
            scales,
            residual,
            defined,
            valid.sum(axis=1),
        )
        for key, value in zip(outputs, values):
            outputs[key].append(value)
        operators.append(operator)
        neighborhoods.append(ns)
        masks.append(valid)
        alignments.append(alignment)
    result = {key: np.asarray(value) for (key, value) in outputs.items()}
    result.update(
        geometry_schema=GEOMETRY_SCHEMA,
        ranks=ranks,
        requested_rank=int(r),
        inverse=operators,
        covariance_operators=operators,
        operator_alignment=alignments,
        neighbors=neighborhoods[-1],
        neighbor_valid=masks[-1],
        neighbors_per_rank=np.asarray(neighborhoods),
        neighbor_valid_per_rank=np.asarray(masks),
        dimension=result["dimensions"].mean(axis=0),
        confidence=result["confidence_per_rank"].mean(axis=0),
        weighted=bool(weighted),
        tie_closed=True,
    )
    validate_geometry(X, result)
    return result


def validate_geometry(X, geometry):
    X = np.asarray(X, dtype=np.float64)
    if geometry["geometry_schema"] != GEOMETRY_SCHEMA:
        raise ValueError("Unknown compact geometry schema")
    (n, d) = X.shape
    if len(geometry["ranks"]) != len(geometry["inverse"]):
        raise ValueError("Rank/operator count mismatch")
    for level, rank in enumerate(geometry["ranks"]):
        operator = geometry["inverse"][level]
        validate(operator)
        if operator["n_nodes"] != n or operator["ambient_dimension"] != d:
            raise ValueError("Operator/geometry dimensions differ")
        ns = geometry["neighbors_per_rank"][level]
        valid = geometry["neighbor_valid_per_rank"][level]
        if ns.shape != valid.shape or ns.shape[0] != n:
            raise ValueError("Rank-shell neighbour/mask shapes differ")
        if not np.array_equal(valid.sum(axis=1), geometry["actual_counts"][level]):
            raise ValueError("Rank-shell counts differ")
        scales = geometry["scales"][level]
        eigenvalues = geometry["eigenvalues"][level]
        if not np.array_equal(scales, operator["scale"]) or not np.array_equal(
            eigenvalues[:, -1], operator["largest_eigenvalue"]
        ):
            raise ValueError("Sidecar scale/spectrum differs from the inverse")
        expected = _alignment(
            X,
            rank,
            ns,
            valid,
            geometry["weights"][level],
            geometry["means"][level],
            scales,
            geometry["dimensions"][level],
            geometry["confidence_per_rank"][level],
            eigenvalues,
            geometry["spectral_ranks"][level],
            geometry["query_residual"][level],
            geometry["query_residual_defined"][level],
            operator,
        )
        if (
            expected != geometry["operator_alignment"][level]
            or expected != operator["geometry_alignment"]
        ):
            raise ValueError("Operator/sidecar alignment identity mismatch")
    return True


def geometry_sidecar(geometry):
    return {
        key: value
        for (key, value) in geometry.items()
        if key not in ("inverse", "covariance_operators")
    }


def validate_shared_geometry(X, shared):
    if "geometry_compact_sidecar" not in shared:
        raise ValueError("New compact trace sidecar is required")
    geometry = dict(shared["geometry_compact_sidecar"])
    geometry["inverse"] = shared["geometry_inverse"]
    validate_geometry(X, geometry)
    aliases = {
        "geometry_ranks": "ranks",
        "geometry_neighbors": "neighbors",
        "geometry_dimensions": "dimensions",
        "geometry_scales": "scales",
        "geometry_neighbor_valid": "neighbor_valid",
        "geometry_actual_counts": "actual_counts",
        "geometry_neighbors_per_rank": "neighbors_per_rank",
    }
    for old_key, new_key in aliases.items():
        if old_key not in shared or not np.array_equal(shared[old_key], geometry[new_key]):
            raise ValueError("Shared geometry alias differs: " + old_key)
    return geometry


def factors(X, pairs, geometry):
    validate_geometry(X, geometry)
    require_canonical_pairs(pairs, len(X))
    (a, b) = (pairs[:, 0], pairs[:, 1])
    displacement = X[b] - X[a]
    length = np.linalg.norm(displacement, axis=1)
    unit = np.divide(
        displacement, length[:, None], out=np.zeros_like(displacement), where=length[:, None] > EPS
    )
    sides = []
    for operator in geometry["inverse"]:
        first = quadratic(operator, a, unit)
        second = quadratic(operator, b, unit)
        sides.append(
            np.stack([np.where(length > EPS, first, 1.0), np.where(length > EPS, second, 1.0)])
        )
    return np.asarray(sides)


def correct(X, base, ev, field, geometry, version, core):
    directional = factors(X, ev["pairs"], geometry)
    if version in ["LN_GE2", "LW_GE2"]:
        factor = directional[-1].mean(axis=0)
    elif version in ["LN_GM", "LW_GM"]:
        factor = directional[-1].max(axis=0)
    elif version == "LP_GM":
        factor = directional.max(axis=1).max(axis=0)
    else:
        assert version == "LN4"
        (a, b) = ev["pairs"].T
        factor = directional.mean(axis=1).mean(axis=0)
        credit = 0.5 * (geometry["confidence"][a] + geometry["confidence"][b])
        factor = 1 + credit * (factor - 1)
    assert np.isfinite(factor).all() and (factor > 0).all()
    t = field["t"] * np.sqrt(factor)
    values = []
    for k, (a, b) in enumerate(ev["pairs"]):
        alpha = core.adaptive_alpha(
            int(a), int(b), base["reliability"], alpha_min=1.0, alpha_max=2.0
        )
        beta = 2 - min(base["reliability"][a], base["reliability"][b])
        sigma = field["sigma"][k]
        values.append(
            sigma**beta * math.exp(-(float(t[k]) ** 2)) / (float(t[k]) + EPS) ** alpha
            if sigma > EPS
            else 0.0
        )
    values = np.asarray(values)
    G = {}
    for (a, b), v in zip(ev["pairs"], values):
        G[int(a), int(b)] = G[int(b), int(a)] = float(v)
    return {
        **field,
        "t": t,
        "G": G,
        "G_ij": values,
        "G_ji": values.copy(),
        "edge_direction_factor": factor,
    }
