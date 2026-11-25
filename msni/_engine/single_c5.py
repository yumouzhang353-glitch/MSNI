import numpy as np
from .sni_single_native import _core
from .sni_single_native._config import EPS
from .quotient_candidates import eligibility, merge
from .fragment_candidates import contact_candidates
from .fragment_quotient import coarsen
from .scale_candidates import threshold


def raw_statistics(X, q1, shared):
    groups = np.unique(q1["active_labels"])
    raw = shared["raw_edges"]
    strong = {tuple(map(int, pair)) for pair in q1["strong_edges"]}
    assert strong <= {tuple(map(int, pair)) for pair in raw}
    trusted = np.asarray([tuple(map(int, pair)) in strong for pair in raw], bool)
    trusted &= q1["selected_mask"][raw].all(axis=1)
    trusted &= q1["active_labels"][raw[:, 0]] == q1["active_labels"][raw[:, 1]]
    pairs = raw[trusted]
    distance = np.linalg.norm(X[pairs[:, 1]] - X[pairs[:, 0]], axis=1)
    edge_groups = np.searchsorted(groups, q1["active_labels"][pairs[:, 0]])
    rows = [distance[(edge_groups == g) & (distance > EPS)] for g in range(len(groups))]
    counts = np.asarray([len(row) for row in rows], np.int64)
    means = np.asarray([row.mean() if len(row) else 0.0 for row in rows])
    stds = np.asarray([row.std() if len(row) else 0.0 for row in rows])
    return dict(
        groups=groups,
        raw_trusted=trusted,
        pairs=pairs,
        distance=distance,
        edge_groups=edge_groups,
        raw_indptr=np.r_[0, np.cumsum(counts)].astype(np.int64),
        raw_values=np.concatenate(rows),
        raw_count=counts,
        raw_mean=means,
        raw_std=stds,
        raw_threshold=threshold(means, stds),
        raw_valid=counts >= 4,
    )


def suffix(X, q1, shared, target):
    (original, contacts) = contact_candidates(q1, shared, target)
    pairs = original["C4"]
    statistics = raw_statistics(X, q1, shared)
    endpoint_groups = np.searchsorted(statistics["groups"], q1["active_labels"][pairs])
    distance = np.linalg.norm(X[pairs[:, 1]] - X[pairs[:, 0]], axis=1)
    mean = statistics["raw_mean"][endpoint_groups]
    ratio = np.divide(distance[:, None], mean, out=np.full_like(mean, np.inf), where=mean > EPS)
    accepted = statistics["raw_valid"][endpoint_groups].all(axis=1) & (distance > EPS)
    accepted &= (ratio <= statistics["raw_threshold"][endpoint_groups]).all(axis=1)
    output = coarsen(q1, pairs[accepted])
    gate = dict(
        pairs=pairs,
        endpoint_groups=endpoint_groups,
        distance=distance,
        raw_ratio=ratio,
        admitted_C5=accepted,
    )
    return (output, dict(CONTACTS=contacts, TRUSTED=statistics, GATE=gate))
