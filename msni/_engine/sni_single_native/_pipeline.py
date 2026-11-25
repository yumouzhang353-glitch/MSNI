import hashlib
import numpy as np


def digest_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def edges(graph):
    return np.asarray(
        [(i, int(j)) for (i, ns) in enumerate(graph) for j in sorted(ns) if i < j], dtype=np.int64
    ).reshape(-1, 2)


def csr(graph):
    ptr = [0]
    indices = []
    for ns in graph:
        indices.extend(sorted(map(int, ns)))
        ptr.append(len(indices))
    return (np.asarray(ptr, np.int64), np.asarray(indices, np.int64))


def build_base(X, core):
    (r, NN, RNN, nb, tree) = core.natural_neighbor_search(X)
    (M, S, gamma) = core.build_mng(NN, RNN)
    reliability = core.compute_reliability_from_nb(nb)
    scale = core.compute_local_scale_from_gamma(X, gamma)
    sigma = core.compute_structure_similarity(M, gamma, reliability)
    G = core.compute_gravity(X, M, sigma, reliability, scale)
    (rho, rho_detail) = core.compute_composite_density(X, NN, RNN, M, S, G)
    (raw_ids, raw) = core.weak_connected_components(M)
    return {
        "r": r,
        "NN": NN,
        "RNN": RNN,
        "nb": nb,
        "M": M,
        "S": S,
        "gamma": gamma,
        "reliability": reliability,
        "local_scale": scale,
        "sigma": sigma,
        "G": G,
        "rho": rho,
        "rho_detail": rho_detail,
        "component_ids_raw": raw_ids,
        "components_raw": raw,
    }


def suffix(X, base, strong, core, caps_override=None):
    (ids, components) = core.weak_connected_components(strong)
    ref_size = core.robust_reference_component_size(components)
    (caps, info) = core.compute_raw_component_center_caps_v94(
        data=X,
        components_raw=base["components_raw"],
        components_cut=components,
        component_ids_raw=base["component_ids_raw"],
        ref_size=ref_size,
        local_scale=base["local_scale"],
    )
    calculated_caps = dict(caps)
    if caps_override is not None:
        caps = {int(k): int(v) for (k, v) in caps_override.items()}
        assert set(caps) == set(calculated_caps)
    selected = core.select_components_by_budget(
        components, base["component_ids_raw"], base["components_raw"], caps
    )
    labels = np.full(len(X), -1, dtype=int)
    selected_mask = np.zeros(len(X), dtype=bool)
    for label, record in enumerate(selected):
        idx = np.asarray(list(components[record["cut_cid"]]), dtype=int)
        labels[idx] = label
        selected_mask[idx] = True
    labels = core.dpc_assign_remaining_clean_points_v94(
        data=X,
        labels=labels,
        rho=base["rho"],
        component_ids_cut=ids,
        component_ids_raw=base["component_ids_raw"],
        raw_parent=info["raw_parent"],
    )
    assert (labels >= 0).all()
    summary = {
        "r": int(base["r"]),
        "raw_count": len(base["components_raw"]),
        "cut_count": len(components),
        "raw_sizes": [len(c) for c in base["components_raw"]],
        "cut_sizes": [len(c) for c in components],
        "reference_size": ref_size,
        "caps": caps,
        "calculated_caps": calculated_caps,
        "parents": info["raw_parent"],
        "significant_cut_min": info["significant_cut_min"],
        "selected": selected,
        "budget_total": int(sum(caps.values())),
        "budget_scale_digest": digest_array(base["local_scale"]),
        "selected_points": int(selected_mask.sum()),
        "fixed_caps_diagnostic": caps_override is not None,
    }
    return (
        labels,
        {
            "component_ids": ids,
            "active_labels": labels,
            "selected_mask": selected_mask,
            "strong_edges": edges(strong),
        },
        summary,
    )


def assign(X, active, active_labels, core):
    mask = np.ones(len(X), dtype=bool)
    mask[active] = False
    labels = np.full(len(X), -1, dtype=int)
    labels[active] = active_labels
    (labels, witness) = core.nearest_retained_labels(X, labels, mask)
    assert (labels >= 0).all() and (not mask[witness[mask]].any())
    return (labels, witness)


def base_arrays(base, active):
    arrays = {
        "active": active,
        "nb": base["nb"],
        "rho": base["rho"],
        "rho_nn": base["rho_detail"]["rho_nn"],
        "reliability": base["reliability"],
        "original_scale": base["local_scale"],
        "raw_ids": base["component_ids_raw"],
        "raw_edges": edges(base["M"]),
        "mutual_edges": edges(base["S"]),
    }
    for name in ["NN", "RNN", "M", "S", "gamma"]:
        (arrays[f"{name}_indptr"], arrays[f"{name}_indices"]) = csr(base[name])
    pairs = arrays["raw_edges"]
    arrays["G_ij"] = np.asarray([base["G"].get(tuple(e), 0.0) for e in pairs])
    arrays["G_ji"] = np.asarray([base["G"].get((int(e[1]), int(e[0])), 0.0) for e in pairs])
    arrays["sigma"] = np.asarray([base["sigma"].get(tuple(e), 0.0) for e in pairs])
    return arrays


def pre_arrays(pre):
    g = pre["graph"]
    out = {
        "r": np.asarray(pre["r"]),
        "relative_density": pre["ev"]["relative_density"],
        "density": pre["ev"]["density"],
        "D": pre["D"],
        "B": pre["B"],
        "weak": pre["weak"],
        "pre_raw_ids": g["raw_ids"],
    }
    for name in ["NN", "RNN", "M", "S", "gamma"]:
        (out[name + "_indptr"], out[name + "_indices"]) = csr(g[name])
    (out["H_indptr"], out["H_indices"]) = csr(pre["H"])
    return out


def derive_base(X, NN, RNN, r, core, M=None, S=None):
    (M0, S0, gamma) = core.build_mng(NN, RNN)
    M = M0 if M is None else M
    S = S0 if S is None else S
    gamma = [{i} | ns for (i, ns) in enumerate(M)]
    nb = np.asarray(list(map(len, RNN)))
    reliability = core.compute_reliability_from_nb(nb)
    mu = core.compute_local_scale_from_gamma(X, gamma)
    sigma = core.compute_structure_similarity(M, gamma, reliability)
    G = core.compute_gravity(X, M, sigma, reliability, mu)
    (rho, detail) = core.compute_composite_density(X, NN, RNN, M, S, G)
    (ids, groups) = core.weak_connected_components(M)
    return dict(
        r=r,
        NN=NN,
        RNN=RNN,
        nb=nb,
        M=M,
        S=S,
        gamma=gamma,
        reliability=reliability,
        local_scale=mu,
        sigma=sigma,
        G=G,
        rho=rho,
        rho_detail=detail,
        component_ids_raw=ids,
        components_raw=groups,
    )


def _original_variable_base(X, ranks, core):
    ranks = np.asarray(ranks, dtype=int)
    assert ranks.shape == (len(X),) and np.all((ranks >= 1) & (ranks < len(X)))
    (neighbors, _) = core._query_knn_excluding_self(X, core.KDTree(X), int(ranks.max()))
    NN = [set(map(int, row[:rank])) for (row, rank) in zip(neighbors, ranks)]
    RNN = [set() for _ in X]
    for i, neighbors_i in enumerate(NN):
        for j in neighbors_i:
            RNN[j].add(i)
    return derive_base(X, NN, RNN, int(ranks.max()), core)


def variable_base(X, ranks, core):
    from ._tie_neighborhood import variable_base as closed_base

    return closed_base(X, ranks, core, derive_base)
