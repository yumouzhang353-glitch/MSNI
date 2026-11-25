import numpy as np
from .recovery import CONFIGURATIONS, align_edges, edge_ids, evidence

EXCLUDED_TERMS = {"geometry", "reference", "target"}
AUXILIARY_TERMS = tuple((term for term in CONFIGURATIONS["transfer"] if term not in EXCLUDED_TERMS))


def strengths(pairs, weights, n):
    return np.bincount(pairs[:, 0], weights=weights, minlength=n) + np.bincount(
        pairs[:, 1], weights=weights, minlength=n
    )


def fuse(n, pairs, full_affinity, core, coupling, with_auxiliary):
    if not 0 < coupling <= 1:
        raise ValueError("coupling must be in (0, 1]")
    if not (np.isfinite(full_affinity).all() and (full_affinity > 0).all()):
        raise ValueError("full affinities must be finite and positive")
    core_mask = core["target"]
    if (core["reference"] & ~core_mask).any():
        raise ValueError("Core reference edges must belong to the target graph")
    if (core["bridge_removed"] & core_mask).any():
        raise ValueError("a bridge-removed edge remains in the final Core target graph")
    corrected = core["corrected_force"][core_mask]
    if not (np.isfinite(corrected).all() and (corrected >= 0).all()):
        raise ValueError("invalid corrected force on a selected Core edge")
    positive = corrected > 0
    core_affinity_pairs = core["pairs"][core_mask][positive]
    corrected = corrected[positive]
    core_strength = strengths(core_affinity_pairs, corrected, n)
    (u, v) = core_affinity_pairs.T
    normalized_core = corrected / (np.sqrt(core_strength[u]) * np.sqrt(core_strength[v]))
    normalized_strength = strengths(core_affinity_pairs, normalized_core, n)
    full_strength = strengths(pairs, full_affinity, n)
    scale = np.ones(n, dtype=float)
    np.divide(full_strength, normalized_strength, out=scale, where=normalized_strength > 0)
    edge_scale = np.minimum(1.0, np.minimum(scale[u], scale[v]))
    correction = coupling * normalized_core * edge_scale
    core_addition_strength = strengths(core_affinity_pairs, correction, n)
    if (core_addition_strength > coupling * full_strength + 1e-12).any():
        raise ValueError("Core addition exceeds the local incident-strength budget")
    enabled = AUXILIARY_TERMS if with_auxiliary else ()
    (aux_terms, _) = evidence(pairs, core, n)
    aux_shift = sum((aux_terms[term] for term in enabled), np.zeros(len(pairs)))
    base = full_affinity * np.exp(np.clip(0.5 * aux_shift, -np.log(3), np.log(2)))
    positive = correction > 0
    (core_affinity_pairs, correction) = (core_affinity_pairs[positive], correction[positive])
    (overlap, mapped) = align_edges(pairs, core_affinity_pairs, n)
    core_only = ~np.isin(edge_ids(core_affinity_pairs, n), edge_ids(pairs, n))
    weights = base.copy()
    weights[overlap] += correction[mapped]
    span_pairs = np.concatenate((pairs, core_affinity_pairs[core_only]))
    weights = np.concatenate((weights, correction[core_only]))
    ratios = np.divide(
        core_addition_strength, full_strength, out=np.zeros(n), where=full_strength > 0
    )
    audit = {
        "full_edges": len(pairs),
        "positive_core_layer_edges": len(correction),
        "overlapping_core_edges": int(overlap.sum()),
        "new_core_only_edges": int(core_only.sum()),
        "core_layer_nodes": int(np.count_nonzero(core_addition_strength)),
        "max_core_added_strength_ratio": float(ratios.max()),
        "mean_core_added_strength_ratio_on_core_nodes": float(
            ratios[core_addition_strength > 0].mean()
        )
        if (core_addition_strength > 0).any()
        else 0.0,
        "fraction_core_layer_by_total_weight": float(correction.sum() / weights.sum()),
        "auxiliary_terms": list(enabled),
        "auxiliary_reweighted_edges": int(np.count_nonzero(aux_shift)),
        "bridge_deleted_edges_in_input": int(
            np.count_nonzero(core["bridge_removed"] | core["first_bridge_removed"])
        ),
    }
    return (span_pairs, weights, audit)


def solve(n, pairs, weights, settings):
    import igraph
    import leidenalg

    if settings["partition"] != "RBConfigurationVertexPartition":
        raise ValueError("the registered partitioner differs")
    graph = igraph.Graph(n=n, edges=pairs.tolist(), directed=False)
    graph.es["w"] = weights.tolist()
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="w",
        initial_membership=settings["initial_membership"],
        n_iterations=settings["n_iterations"],
        seed=settings["seed"],
        resolution_parameter=settings["resolution_parameter"],
        max_comm_size=settings["max_comm_size"],
    )
    return (
        np.asarray(partition.membership, dtype=np.int64),
        {"igraph": igraph.__version__, "leidenalg": leidenalg.__version__},
    )
