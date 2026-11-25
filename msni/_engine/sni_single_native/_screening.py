from collections import Counter
import numpy as np
from ._config import EPS

P = {
    "weak_nb_ratio": 0.5,
    "short_edge_ratio": 1.5,
    "minimum_short_mutual_support": 2,
    "density_deficit_ratio": 2 / 3,
    "group_internal_ratio": 0.6,
    "group_density_ratio": 0.5,
    "anchor_minimum_edges": 2,
    "curvature_gap_smoothing": 5,
    "background_multiplier": 2.0,
    "angle_cosine_maximum": -0.5,
    "group_minimum_size": "max(3,r+1)",
    "rank_vote_fraction": 2 / 3,
    "raw_coordinates": True,
    "maximum_dimension": 10,
    "maximum_n": 10000,
    "workers": 4,
    "timeout_s": 1800,
    "technical_retries": 1,
    "true_labels_used": False,
}


def _build_full_graph(X, core, original=None):
    original = core.compute_original_graph_info(X) if original is None else original
    (M, S, gamma) = core.build_mng(original["NN"], original["RNN"])
    rel = core.compute_reliability_from_nb(original["nb"])
    scales = core.compute_local_scale_from_gamma(X, gamma)
    sigma = core.compute_structure_similarity(M, gamma, rel)
    G = core.compute_gravity(X, M, sigma, rel, scales)
    (rho, detail) = core.compute_composite_density(X, original["NN"], original["RNN"], M, S, G)
    strong = core.build_component_mng_by_strong_edges(X, M, S, G, scales)
    (raw_ids, raw) = core.weak_connected_components(M)
    (cut_ids, cut) = core.weak_connected_components(strong)
    return {
        **original,
        "M": M,
        "S": S,
        "gamma": gamma,
        "reliability": rel,
        "local_scale": scales,
        "G": G,
        "rho": rho,
        "rho_nn": detail["rho_nn"],
        "strong": strong,
        "raw_ids": np.asarray(raw_ids),
        "cut_ids": np.asarray(cut_ids),
        "raw_count": len(raw),
        "cut_count": len(cut),
    }


def scale_floor(values):
    positive = values[values > 0]
    return max(np.finfo(float).tiny, np.median(positive) * 1e-09 if len(positive) else 1.0)


def _prepare_density_evidence(X, graph, core):
    n = len(X)
    N = max(1, min(n - 1, (graph["r"] + int(np.max(graph["nb"]))) // 2))
    rho = core.compute_dsngcap_density(X, graph["tree"], N)
    (distance, index) = graph["tree"].query(X, k=N + 1)
    (distance, index) = (distance[:, 1:], index[:, 1:])
    scales = np.maximum(graph["local_scale"], scale_floor(graph["local_scale"]))
    log_rho = np.log(np.maximum(rho, np.finfo(float).tiny))
    peer_log = np.asarray(
        [np.median(log_rho[list(ns)]) if ns else log_rho[i] for (i, ns) in enumerate(graph["NN"])]
    )
    deficit = log_rho - peer_log
    relative = np.exp(np.clip(deficit, -50, 50))
    normalized = distance / np.sqrt(scales[:, None] * scales[index])
    pair = N / np.maximum(normalized.sum(axis=1), EPS)
    return {
        "N": N,
        "density": rho,
        "relative_density": relative,
        "pair_density": pair,
        "log_density_deficit": deficit,
        "scales": scales,
    }


def weak_evidence(X, graph, evidence):
    scales = evidence["scales"]
    nb = graph["nb"]
    peer_nb = []
    short = [set() for _ in X]
    for i, ns in enumerate(graph["NN"]):
        peer_nb.append(np.median(nb[list(ns)]) if ns else nb[i])
    ratio = (nb + 1) / (np.asarray(peer_nb) + 1)
    for i, ns in enumerate(graph["S"]):
        for j in ns:
            if (
                np.linalg.norm(X[i] - X[j]) / np.sqrt(scales[i] * scales[j])
                <= P["short_edge_ratio"]
            ):
                short[i].add(int(j))
    support = np.asarray(list(map(len, short)))
    deficient = (
        (ratio < P["weak_nb_ratio"])
        | (evidence["relative_density"] < P["density_deficit_ratio"])
        | (nb == 0)
    )
    weak = (support < P["minimum_short_mutual_support"]) & deficient
    return (
        weak,
        short,
        {"relative_nb": ratio, "short_mutual_support": support, "weak_structure": weak},
    )


def protect_groups(mask, short, evidence, r, core):
    candidate_indices = set(map(int, np.flatnonzero(mask)))
    outside_indices = set(map(int, np.flatnonzero(~mask)))
    candidate = [
        ns.intersection(candidate_indices) if mask[i] else set() for (i, ns) in enumerate(short)
    ]
    (_, groups) = core.weak_connected_components(candidate)
    outside = [
        ns.intersection(outside_indices) if not mask[i] else set() for (i, ns) in enumerate(short)
    ]
    (outside_ids, outside_groups) = core.weak_connected_components(outside)
    minimum = max(3, int(r) + 1)
    anchors = {cid for (cid, points) in enumerate(outside_groups) if len(points) >= minimum}
    protect = np.zeros(len(mask), bool)
    records = []
    for group in groups:
        idx = np.asarray(sorted(group), int)
        if not mask[idx[0]] or len(idx) < minimum:
            continue
        ratios = [len(short[i].intersection(group)) / max(1, len(short[i])) for i in idx]
        internal = float(np.median(ratios))
        density = float(np.median(evidence["relative_density"][idx]))
        contacts = Counter(
            (
                int(outside_ids[j])
                for i in idx
                for j in short[i]
                if not mask[j] and outside_ids[j] in anchors
            )
        )
        substantial = sum((count >= P["anchor_minimum_edges"] for count in contacts.values()))
        eligible = (
            internal >= P["group_internal_ratio"]
            and density >= P["group_density_ratio"]
            and (substantial <= 1)
        )
        if eligible:
            protect[idx] = True
        records.append(
            {
                "points": idx.tolist(),
                "size": len(idx),
                "internal_support": internal,
                "relative_density": density,
                "anchor_regions": int(substantial),
                "protected": bool(eligible),
            }
        )
    return (mask & ~protect, protect, records)


def restricted(graph, active):
    ids = set(map(int, np.flatnonzero(active)))
    return [ns & ids if active[i] else set() for (i, ns) in enumerate(graph)]


def anchors(graph, active, r, core):
    (ids, groups) = core.weak_connected_components(restricted(graph, active))
    accepted = {
        k for (k, g) in enumerate(groups) if len(g) >= max(3, int(r) + 1) and active[next(iter(g))]
    }
    return (np.asarray(ids), accepted, groups)


def two_sided_protection(X, candidate, short, evidence, r, core):
    two = np.zeros(len(X), bool)
    for i in np.flatnonzero(candidate):
        if evidence["relative_density"][i] < 0.5 or len(short[i]) < 2:
            continue
        vectors = X[np.asarray(sorted(short[i]), int)] - X[i]
        lengths = np.linalg.norm(vectors, axis=1)
        vectors = vectors[lengths > EPS] / lengths[lengths > EPS, None]
        if len(vectors) >= 2:
            two[i] = bool(
                np.any(
                    (vectors @ vectors.T)[np.triu_indices(len(vectors), 1)]
                    <= P["angle_cosine_maximum"]
                )
            )
    (ids, regions, _) = anchors(short, ~candidate, r, core)
    (_, groups) = core.weak_connected_components(restricted(short, two))
    protect = np.zeros(len(X), bool)
    records = []
    for group in groups:
        idx = np.asarray(sorted(group), int)
        if not two[idx[0]] or len(idx) < max(3, r + 1):
            continue
        contacts = Counter(
            (int(ids[j]) for i in idx for j in short[i] if not candidate[j] and ids[j] in regions)
        )
        substantial = sum((n >= 2 for n in contacts.values()))
        eligible = substantial <= 1
        if eligible:
            protect[idx] = True
        records.append(
            {
                "points": idx.tolist(),
                "size": len(idx),
                "anchor_regions": substantial,
                "protected": eligible,
            }
        )
    return (protect, two, records)


def pre_state(X, core):
    natural = core.compute_original_graph_info(X)
    graph = _build_full_graph(X, core, natural)
    ev = _prepare_density_evidence(X, graph, core)
    (weak, short, _) = weak_evidence(X, graph, ev)
    (ds, *_) = core.detect_low_density_points_by_dsngcap_density(
        X, natural["tree"], natural["r"], natural["nb"]
    )
    (bs, *_) = core.detect_low_density_points_by_nb_adaptive(natural["nb"], natural["r"])
    D = np.zeros(len(X), bool)
    B = D.copy()
    D[ds] = True
    B[bs] = True
    return {
        "graph": graph,
        "ev": ev,
        "H": short,
        "weak": weak,
        "D": D,
        "B": B,
        "r": int(natural["r"]),
    }


def screen(X, front, core, pre):
    assert front == "N4_B"
    (D, B) = (pre["D"], pre["B"])
    C = D | B
    (_, protected, _) = protect_groups(C, pre["H"], pre["ev"], pre["r"], core)
    (extra, _, _) = two_sided_protection(X, C, pre["H"], pre["ev"], pre["r"], core)
    mask = C & ~(protected | extra)
    if (~mask).sum() < 2:
        for value in np.unique(pre["ev"]["density"])[::-1]:
            mask[mask & (pre["ev"]["density"] == value)] = False
            if (~mask).sum() >= 2:
                break
    return mask
