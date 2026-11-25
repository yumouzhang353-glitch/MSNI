import numpy as np

METHODS = ("R_NAT_DEG", "R_NAT_GSYM")
MAX_PAIRS = 2000000


class DataRejected(ValueError):
    pass


def need(value, message):
    if not value:
        raise DataRejected(message)


def check_pair_budget(count):
    need(
        type(count) is int and 1 <= count <= MAX_PAIRS,
        "complete natural graph exceeds fixed pair budget or is empty",
    )


def static_weights(pairs, gravity, n, method):
    need(method in METHODS, "fixed natural RCC method")
    degree = np.bincount(pairs.ravel(), minlength=n).astype(np.float64)
    need((degree > 0).all(), "full natural graph has no isolated input point")
    if method == "R_NAT_DEG":
        weights = degree.mean() / np.sqrt(degree[pairs[:, 0]] * degree[pairs[:, 1]])
        strength = degree.copy()
    else:
        need(
            gravity.dtype == np.float64
            and gravity.shape == (len(pairs),)
            and np.isfinite(gravity).all()
            and (gravity >= 0).all(),
            "finite nonnegative symmetric G",
        )
        strength = np.zeros(n, np.float64)
        np.add.at(strength, pairs[:, 0], gravity)
        np.add.at(strength, pairs[:, 1], gravity)
        denominator = np.sqrt(strength[pairs[:, 0]]) * np.sqrt(strength[pairs[:, 1]])
        weights = np.zeros(len(pairs), np.float64)
        np.divide(gravity, denominator, out=weights, where=denominator > 0)
    need(
        np.isfinite(strength).all()
        and np.isfinite(weights).all()
        and (weights >= 0).all()
        and np.any(weights > 0),
        "positive finite static graph weight required;no fallback",
    )
    return (weights, degree, strength)


def build_graph(X):
    from .sni_single_native import _tie_neighborhood as natural, _core as core

    (r, NN, RNN, nb, _) = natural.natural_neighbor_search(X)
    (M, S, gamma) = core.build_mng(NN, RNN)
    check_pair_budget(sum(map(len, M)) // 2)
    pairs = np.asarray(
        [(i, j) for (i, bag) in enumerate(M) for j in sorted(bag) if i < j], np.int64
    ).reshape(-1, 2)
    reliability = core.compute_reliability_from_nb(nb)
    scale = core.compute_local_scale_from_gamma(X, gamma)
    sigma_map = core.compute_structure_similarity(M, gamma, reliability)
    G = core.compute_gravity(X, M, sigma_map, reliability, scale)
    sigma = np.asarray([sigma_map[int(i), int(j)] for (i, j) in pairs], np.float64)
    forward = np.asarray([G[int(i), int(j)] for (i, j) in pairs], np.float64)
    reverse = np.asarray([G[int(j), int(i)] for (i, j) in pairs], np.float64)
    need(
        np.array_equal(forward, reverse),
        "frozen SNI field is symmetric on complete natural candidates",
    )
    need(
        np.isfinite(scale).all()
        and (scale > 0).all()
        and np.isfinite(sigma).all()
        and np.isfinite(forward).all()
        and (forward >= 0).all(),
        "finite full natural field domain",
    )
    return dict(
        r=int(r),
        NN=[sorted(v) for v in NN],
        RNN=[sorted(v) for v in RNN],
        M=[sorted(v) for v in M],
        S=[sorted(v) for v in S],
        gamma=[sorted(v) for v in gamma],
        nb=nb,
        reliability=reliability,
        local_scale=scale,
        sigma=sigma,
        gravity=forward,
        reverse_gravity=reverse,
        pairs=pairs,
        full_point_count=len(X),
        screening_calls=0,
        budget_calls=0,
        candidate_semantics="full_X_tie_closed_NN_union_RNN;no_strong_edge_predicate_or_MST",
    )
