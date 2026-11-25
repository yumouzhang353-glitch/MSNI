import numpy as np

RULES = ("NO_C5", "D25_ANY", "D50_ANY", "D25_QUORUM", "D50_QUORUM", "LOCAL_MEDIAN", "D25_LOCAL")
BOUNDARIES = ("C0", "W_RAW_MIN", "W_UNIT_MIN")
MODES = BOUNDARIES + tuple(("W_RAW_MIN__" + r for r in RULES))


def qualify(labels, seeds, rho, nn_indptr, nn_indices, pairs, source_pass, rule):
    if rule not in RULES:
        raise ValueError(rule)
    n = len(labels)
    m = len(pairs)
    if (
        labels.dtype != np.int64
        or seeds.dtype != np.int64
        or labels.shape != (n,)
        or (seeds.shape != (n,))
    ):
        raise ValueError("int64 label/seed vectors required")
    if (
        rho.dtype != np.float64
        or rho.shape != (n,)
        or (not np.isfinite(rho).all())
        or (rho < 0).any()
    ):
        raise ValueError("finite nonnegative float64 rho required")
    if pairs.dtype != np.int64 or pairs.shape != (m, 2) or (pairs < 0).any() or (pairs >= n).any():
        raise ValueError("invalid pairs")
    if source_pass.dtype != bool or source_pass.shape != (m,):
        raise ValueError("invalid source pass")
    if nn_indptr.dtype != np.int64 or nn_indices.dtype != np.int64 or nn_indptr.shape != (n + 1,):
        raise ValueError("invalid CSR")
    if nn_indptr[0] != 0 or nn_indptr[-1] != len(nn_indices) or (np.diff(nn_indptr) < 0).any():
        raise ValueError("invalid CSR offsets")
    if (nn_indices < 0).any() or (nn_indices >= n).any() or (labels < 0).any():
        raise ValueError("invalid node/label")
    selected = seeds >= 0
    if not np.array_equal(labels[selected], seeds[selected]):
        raise ValueError("fixed seeds disagree")
    groups = np.unique(labels)
    if m and (labels[pairs[:, 0]] == labels[pairs[:, 1]]).any():
        raise ValueError("noncrossing contact")
    (keys, domain) = np.unique(np.sort(labels[pairs], axis=1), axis=0, return_inverse=True)
    oriented = pairs.copy()
    swap = labels[pairs[:, 0]] > labels[pairs[:, 1]]
    oriented[swap] = oriented[swap, ::-1]
    percent = 50.0 if rule.startswith("D50") else 25.0
    floors = np.zeros(len(groups), np.float64)
    counts = np.zeros(len(groups), np.int64)
    group_valid = np.zeros(len(groups), bool)
    for k, g in enumerate(groups):
        values = rho[selected & (labels == g)]
        counts[k] = len(values)
        if len(values):
            floors[k] = float(np.percentile(values, percent, method="linear"))
            group_valid[k] = floors[k] > 0
    local = np.zeros(n, np.float64)
    local_valid = np.zeros(n, bool)
    for i in range(n):
        ns = nn_indices[nn_indptr[i] : nn_indptr[i + 1]]
        if len(ns):
            if i in ns or len(np.unique(ns)) != len(ns):
                raise ValueError("self/repeated NN")
            local[i] = float(np.median(rho[ns]))
            local_valid[i] = local[i] > 0
    gi = np.searchsorted(groups, labels[pairs])
    density_known = group_valid[gi].all(axis=1)
    density_pass = (rho[pairs] >= floors[gi]).all(axis=1)
    local_known = local_valid[pairs].all(axis=1)
    local_pass = (rho[pairs] >= local[pairs]).all(axis=1)
    if rule == "LOCAL_MEDIAN":
        (known, edge_pass) = (local_known, local_pass)
    elif rule == "D25_LOCAL":
        (known, edge_pass) = (density_known & local_known, density_pass & local_pass)
    else:
        (known, edge_pass) = (density_known, density_pass)
    effective = np.zeros(m, bool)
    admitted = np.zeros(len(keys), bool)
    unknown = np.zeros(len(keys), bool)
    qualified = np.zeros(len(keys), np.int64)
    endpoints = np.zeros((len(keys), 2), np.int64)
    source_counts = np.zeros(len(keys), np.int64)
    for k in range(len(keys)):
        ix = np.flatnonzero((domain == k) & source_pass)
        source_counts[k] = len(ix)
        if not len(ix):
            continue
        unknown[k] = not bool(known[ix].all())
        good = ix[known[ix] & edge_pass[ix]]
        qualified[k] = len(good)
        for side in range(2):
            endpoints[k, side] = len(np.unique(oriented[good, side]))
        if rule == "NO_C5":
            decision = False
        elif unknown[k]:
            decision = True
        elif rule.endswith("QUORUM"):
            decision = bool((endpoints[k] >= 2).all())
        else:
            decision = bool(len(good))
        admitted[k] = decision
        if decision:
            effective[ix] = True
    assert not (effective & ~source_pass).any()
    return dict(
        rule=rule,
        pairs=pairs.copy(),
        source_pass=source_pass.copy(),
        effective_pass=effective,
        groups=groups,
        group_seed_counts=counts,
        group_density_quantile=np.asarray(percent),
        group_density_floor=floors,
        group_density_valid=group_valid,
        local_NN_median=local,
        local_NN_valid=local_valid,
        edge_density_known=density_known,
        edge_density_pass=density_pass,
        edge_local_known=local_known,
        edge_local_pass=local_pass,
        edge_known=known,
        edge_qualified=edge_pass,
        domain_keys=keys,
        edge_domain=domain,
        oriented_endpoints=oriented,
        domain_source_edges=source_counts,
        domain_qualified_edges=qualified,
        domain_qualified_endpoint_counts=endpoints,
        domain_unknown_fallback=unknown,
        domain_admitted=admitted,
        quantile_method="linear",
        calibration_members="current_Q1_fixed_seeds",
        local_calibration="all_original_NN_members",
        unknown_fallback="any_unknown_endpoint_on_an_original_pass_edge_preserves_whole_domain",
        retained_edges="all_original_pass_edges_of_admitted_domain",
    )
