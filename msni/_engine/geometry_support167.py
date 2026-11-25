import numpy as np
from .sni_single_native._config import EPS
from . import geometry_adapter167 as _geometry

local_geometry = _geometry.prepare


def continuation(X, pre, parent_mask, core, parameters, tie_geometry=False):
    n = len(X)
    r = int(pre["r"])
    graph = pre["graph"]
    reliability = core.compute_reliability_from_nb(graph["nb"])
    geometry = local_geometry(X, r, reliability, weighted=True, tie_closed=tie_geometry)
    inverse = geometry["inverse"][-1]
    scale = geometry["scales"][-1]
    dimension = geometry["dimensions"][-1]
    valid = geometry["query_residual_defined"][-1]
    residual = geometry["query_residual"][-1]
    k = int(geometry["ranks"][-1])
    neighbors = geometry["neighbors"] if tie_geometry else geometry["neighbors"][:, :k]
    centered = X[neighbors] - geometry["means"][-1, :, None, :]
    neighborhood_residual = (
        _geometry.quadratic(
            inverse,
            np.repeat(np.arange(n, dtype=np.int64), centered.shape[1]),
            centered.reshape(-1, X.shape[1]),
        ).reshape(centered.shape[:2])
        / scale[:, None]
    )
    fence = (
        np.asarray(
            [
                np.quantile(row[valid], parameters["fit_quantile"])
                for (row, valid) in zip(neighborhood_residual, geometry["neighbor_valid"])
            ]
        )
        if tie_geometry
        else np.quantile(neighborhood_residual, parameters["fit_quantile"], axis=1)
    )
    fit = valid & (residual <= fence + EPS)
    narrow = dimension <= parameters["maximum_effective_dimension"] + EPS
    enough_incoming = graph["nb"] >= parameters["minimum_nb"]
    opposing = np.zeros(n, bool)
    for i in np.flatnonzero(parent_mask & fit & narrow & enough_incoming):
        ns = np.asarray(sorted(graph["NN"][int(i)]), int)
        if len(ns) < 2:
            continue
        vectors = X[ns] - X[i]
        lengths = np.linalg.norm(vectors, axis=1)
        vectors = vectors[lengths > EPS] / lengths[lengths > EPS, None]
        if len(vectors) >= 2:
            opposing[i] = bool(
                np.any(
                    (vectors @ vectors.T)[np.triu_indices(len(vectors), 1)]
                    <= parameters["opposing_cosine_maximum"] + EPS
                )
            )
    ranks = sorted(
        set((min(n - 1, max(1, r * factor)) for factor in parameters["density_rank_factors"]))
    )
    (ranked, _) = core._query_knn_excluding_self(X, graph["tree"], max(ranks))
    ranked = np.asarray(ranked, int)
    rho = pre["ev"]["density"]
    ratios = np.asarray(
        [
            rho / np.maximum(np.median(rho[ranked[:, :rank]], axis=1), np.finfo(float).tiny)
            for rank in ranks
        ]
    )
    rank_to_row = {rank: index for (index, rank) in enumerate(ranks)}
    scores = {}
    for name, factors in [("L", [1]), ("M", [1, 2, 4]), ("H", [1, 2, 4, 8])]:
        rows = [rank_to_row[min(n - 1, max(1, r * factor))] for factor in factors]
        scores[name] = np.min(ratios[rows], axis=0)
    eligible = parent_mask & fit & narrow & enough_incoming & opposing
    protects = {
        name: eligible & (score >= parameters["minimum_density_ratio"] - EPS)
        for (name, score) in scores.items()
    }
    assert not (protects["H"] & ~protects["M"]).any() and (
        not (protects["M"] & ~protects["L"]).any()
    )
    arrays = dict(
        continuity_parent_mask=parent_mask,
        continuity_fit=fit,
        continuity_narrow=narrow,
        continuity_incoming_support=enough_incoming,
        continuity_opposing=opposing,
        continuity_query_residual=residual,
        continuity_fit_fence=fence,
        continuity_dimension=dimension,
        continuity_density_ranks=np.asarray(ranks),
        continuity_density_ratios=ratios,
        continuity_density_score_L=scores["L"],
        continuity_density_score_M=scores["M"],
        continuity_density_score_H=scores["H"],
        continuity_protect_L=protects["L"],
        continuity_protect_M=protects["M"],
        continuity_protect_H=protects["H"],
    )
    return (protects, arrays)


def fragment_prepare(X, pre, parent_mask, continuity_H, core, maximum_ratio, opposing_maximum):
    (ids, groups) = core.weak_connected_components(pre["H"])
    minimum_size = max(3, int(pre["r"]) + 1)
    sizes = np.asarray([len(group) for group in groups], dtype=int)
    ratios = np.full(len(groups), np.inf)
    singular_1 = np.zeros(len(groups))
    singular_2 = np.zeros(len(groups))
    for cid, group in enumerate(groups):
        points = X[np.asarray(sorted(group), dtype=int)]
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if len(singular):
            singular_1[cid] = singular[0]
        if len(singular) > 1:
            singular_2[cid] = singular[1]
        if singular_1[cid] > EPS:
            ratios[cid] = singular_2[cid] / singular_1[cid]
    qualified = (sizes >= minimum_size) & (ratios <= maximum_ratio + EPS)
    supported = qualified[ids]
    opposing = np.zeros(len(X), dtype=bool)
    for i in np.flatnonzero(continuity_H):
        neighbors = np.asarray(sorted(pre["graph"]["NN"][int(i)]), dtype=int)
        neighbors = neighbors[supported[neighbors]]
        vectors = X[neighbors] - X[i]
        lengths = np.linalg.norm(vectors, axis=1)
        vectors = vectors[lengths > EPS] / lengths[lengths > EPS, None]
        if len(vectors) >= 2:
            opposing[i] = bool(
                np.any(
                    (vectors @ vectors.T)[np.triu_indices(len(vectors), 1)]
                    <= opposing_maximum + EPS
                )
            )
    protects = parent_mask & continuity_H & opposing
    assert not (protects & ~continuity_H).any()
    arrays = dict(
        fragment_component_ids=np.asarray(ids, int),
        fragment_sizes=sizes,
        fragment_singular_1=singular_1,
        fragment_singular_2=singular_2,
        fragment_ratios=ratios,
        fragment_qualified=qualified,
        fragment_supported=supported,
        fragment_opposing=opposing,
        fragment_protect=protects,
        fragment_minimum_size=np.asarray(minimum_size),
        fragment_maximum_ratio=np.asarray(maximum_ratio),
    )
    return (protects, arrays)


def gated_support(
    X, pre, parent_mask, candidates, extra, core, protocol, stable, tie_geometry=False
):
    (_, fragments) = fragment_prepare(
        X,
        pre,
        parent_mask,
        candidates["H"],
        core,
        protocol["fragment_ratio"],
        protocol["continuity_parameters"]["opposing_cosine_maximum"],
    )
    extra.update(fragments)
    reliability = core.compute_reliability_from_nb(pre["graph"]["nb"])
    geometry = _geometry.prepare(
        X, int(pre["r"]), reliability, weighted=True, tie_closed=tie_geometry
    )
    dimensions = geometry["dimensions"][[0, -1]]
    ids = fragments["fragment_component_ids"]
    count = len(fragments["fragment_sizes"])
    anchors = ~parent_mask
    medians = np.full((2, count), np.inf)
    anchor_count = np.zeros(count, dtype=int)
    for cid in range(count):
        selected = anchors & (ids == cid)
        anchor_count[cid] = int(selected.sum())
        if selected.any():
            medians[:, cid] = np.median(dimensions[:, selected], axis=1)
    enough = anchor_count >= int(fragments["fragment_minimum_size"])
    limit = float(protocol["anchor_dimension_limit"])
    median_gate = enough & (medians[-1] <= limit + EPS)
    stable_gate = median_gate & (medians[0] <= limit + EPS)
    gate = stable_gate if stable else median_gate
    original_qualified = fragments["fragment_qualified"].copy()
    qualified = original_qualified & gate
    supported = qualified[ids]
    opposing = np.zeros(len(X), dtype=bool)
    for i in np.flatnonzero(candidates["H"]):
        neighbors = np.asarray(sorted(pre["graph"]["NN"][int(i)]), dtype=int)
        neighbors = neighbors[supported[neighbors]]
        vectors = X[neighbors] - X[i]
        lengths = np.linalg.norm(vectors, axis=1)
        vectors = vectors[lengths > EPS] / lengths[lengths > EPS, None]
        if len(vectors) >= 2:
            opposing[i] = bool(
                np.any(
                    (vectors @ vectors.T)[np.triu_indices(len(vectors), 1)]
                    <= protocol["continuity_parameters"]["opposing_cosine_maximum"] + EPS
                )
            )
    broad = parent_mask & candidates["H"] & opposing
    protected = broad & (extra["continuity_dimension"] <= limit + EPS)
    search_allowed = stable_gate[ids]
    extra.update(
        fragment_original_qualified=original_qualified,
        fragment_qualified=qualified,
        fragment_supported=supported,
        fragment_opposing=opposing,
        fragment_protect=broad,
        anchor_counts=anchor_count,
        anchor_median_dimensions=medians,
        anchor_full_dimensions=dimensions,
        anchor_full_geometry_ranks=np.asarray(geometry["ranks"])[[0, -1]],
        anchor_median_gate=median_gate,
        anchor_stable_gate=stable_gate,
        anchor_protection_gate=gate,
        support_search_allowed=search_allowed,
        support_applied_protect=protected,
        support_parent_mask=parent_mask,
    )
    return (protected, search_allowed, extra)
