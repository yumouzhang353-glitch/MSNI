import numpy as np
from ._bridge import bridges

MODES = ["P3_MUTUAL", "P4_NB", "P5_RAW_DENSITY"]


def protected_bridges(base, H, active, pairs, candidate, mode, core):
    assert mode in MODES
    n = len(active)
    index = np.full(len(H), -1, np.int64)
    index[active] = np.arange(n)
    HA = [set((int(index[j]) for j in H[int(i)] if index[j] >= 0)) for i in active]
    values = base["nb"].astype(float) if mode == "P4_NB" else base["rho_detail"]["rho_nn"]
    cache = {}
    protected = np.zeros(len(pairs), bool)
    present_count = 0
    for k in np.flatnonzero(candidate):
        (a, b) = map(int, pairs[k])
        threshold = None if mode == "P3_MUTUAL" else float(min(values[a], values[b]))
        if threshold not in cache:
            if mode == "P3_MUTUAL":
                J = [neighbors & base["S"][i] for (i, neighbors) in enumerate(HA)]
            else:
                allowed = values >= threshold
                assert allowed[a] and allowed[b]
                J = [
                    set((j for j in neighbors if allowed[j])) if allowed[i] else set()
                    for (i, neighbors) in enumerate(HA)
                ]
            (ids, _) = core.weak_connected_components(J)
            cache[threshold] = (J, ids, bridges(J))
        (J, ids, critical) = cache[threshold]
        present = b in J[a]
        present_count += int(present)
        protected[k] = bool(ids[a] == ids[b] and (not present or (a, b) not in critical))
    assert not (protected & ~candidate).any()
    return (
        protected,
        dict(
            rule=mode,
            candidate_count=int(candidate.sum()),
            protected_count=int(protected.sum()),
            candidates_present_in_support_graph=present_count,
            distinct_support_graphs=len(cache),
            classification="connected_components_and_lowlink",
            graph_cache_scope="single_input_single_mode_single_call",
            reference_labels_received=False,
        ),
    )
