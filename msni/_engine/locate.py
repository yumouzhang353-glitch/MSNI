from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def unit(v):
    return v / np.linalg.norm(v)


def normal(v):
    return np.array([-v[1], v[0]])


def neighborhood(X):
    k = max(6, int(np.ceil(np.log2(len(X)))))
    (dist, ids) = cKDTree(X).query(X, k=2 * k + 1)
    return (k, dist, ids)


def spectrum(X, ids):
    patches = X[ids]
    patches = patches - patches.mean(axis=1, keepdims=True)
    cov = patches.transpose(0, 2, 1) @ patches
    eig = np.linalg.eigvalsh(cov)
    return eig[:, 0] / np.maximum(eig[:, 1], np.finfo(float).tiny)


def get_bodies(X, k, ids):
    small = spectrum(X, ids[:, : k + 1])
    large = spectrum(X, ids)
    thick = (small > 0.16) & (large > 0.2)
    rows = np.repeat(np.arange(len(X)), k)
    cols = ids[:, 1 : k + 1].ravel()
    use = thick[rows] & thick[cols]
    graph = coo_matrix((np.ones(use.sum()), (rows[use], cols[use])), shape=(len(X), len(X)))
    (count, labels) = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    bodies = [np.flatnonzero(labels == i) for i in np.flatnonzero(sizes >= 3 * k)]
    return (bodies, (small < 0.16) & (large < 0.16))


def estimate_tip(X, inds, origin, direction, spacing):
    diff = X[inds] - origin
    along = diff @ direction
    across = diff @ normal(direction)
    sides = []
    for sign in [-1, 1]:
        flank = along[(sign * across >= 0.55 * spacing) & (sign * across <= 2.5 * spacing)]
        if len(flank) < 3:
            return None
        sides.append(np.mean(np.sort(flank)[-2:]))
    return origin + direction * np.mean(sides)


def find_route(X, pair, thin, radii, k):
    (left, right) = pair
    center = [X[ids].mean(axis=0) for ids in pair]
    if np.linalg.norm(center[1] - center[0]) == 0:
        return None
    direction = unit(center[1] - center[0])
    origin = (center[0] + center[1]) / 2
    spacing = [np.median(radii[ids]) for ids in pair]
    scale = np.mean(spacing)

    def bounds():
        p = estimate_tip(X, left, origin, direction, spacing[0])
        q = estimate_tip(X, right, origin, -direction, spacing[1])
        if p is None or q is None or (q - p) @ direction < 2 * scale:
            return None
        return (p, q)

    def witnesses(p, q, width, trim):
        span = (q - p) @ direction
        along = (X - p) @ direction
        across = (X - p) @ normal(direction)
        return thin & (abs(across) < width) & (along >= trim * span) & (along <= (1 - trim) * span)

    tips = bounds()
    if tips is None:
        return None
    mask = witnesses(*tips, 1.1 * scale, 0.12)
    if mask.sum() < k:
        return None
    mid = X[mask]
    origin = mid.mean(axis=0)
    (_, axes) = np.linalg.eigh((mid - origin).T @ (mid - origin))
    axis = axes[:, -1]
    if abs(axis @ direction) < 0.9:
        return None
    direction = axis * np.sign(axis @ direction)
    tips = bounds()
    if tips is None:
        return None
    (p, q) = tips
    length = np.linalg.norm(p - q)
    mask = witnesses(p, q, 0.65 * scale, 0.0)
    t = np.sort((X[mask] - p) @ direction)
    if len(t) < k or t[-1] - t[0] < 0.6 * length:
        return None
    gap = np.max(np.diff(np.r_[0, t, length]))
    if gap > min(0.4 * length, 2.8 * scale):
        return None
    return {
        "ends": [p.tolist(), q.tolist()],
        "witnesses": int(len(t)),
        "gap_fraction": float(gap / length),
    }


def measure_end(X, p, inward, k, dist, ids):
    delta = X - p
    radius = np.linalg.norm(delta, axis=1)
    near = np.argsort(radius)[: 4 * k]
    scale = np.median(dist[near, k])
    along = delta @ inward
    across = delta @ normal(inward)
    use = np.flatnonzero(
        (along >= 0.1 * scale) & (along <= 2.4 * scale) & (abs(across) <= 1.25 * scale)
    )
    if not len(use):
        return None
    best = use[np.argmin(dist[use, k])]
    scores = []
    for size in [k + 1, 2 * k + 1]:
        neighbors = X[ids[best, :size]]
        patch = neighbors - neighbors.mean(axis=0)
        run = np.sum((patch @ inward) ** 2)
        transverse = np.sum((patch @ normal(inward)) ** 2)
        if run + transverse == 0:
            return None
        scores.append((run - transverse) / (run + transverse))
    return {"score": float(np.mean(scores)), "center": X[best].tolist()}


def role_at(X, ends, k, dist, ids):
    (p, q) = np.asarray(ends)
    toward = unit(q - p)
    readings = [measure_end(X, p, -toward, k, dist, ids), measure_end(X, q, toward, k, dist, ids)]
    if any((r is None for r in readings)):
        return {"role": "abstain", "readings": readings}
    role = "within" if all((r["score"] > 0 for r in readings)) else "cross"
    return {"role": role, "readings": readings}


def locate(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or (not np.all(np.isfinite(X))):
        raise ValueError("Expected a finite planar point cloud")
    if len(X) < 32:
        return {"bodies": 0, "routes": [], "role": "abstain"}
    (k, dist, ids) = neighborhood(X)
    (bodies, thin) = get_bodies(X, k, ids)
    routes = []
    for pair in combinations(bodies, 2):
        route = find_route(X, pair, thin, dist[:, k], k)
        if route is not None:
            route.update(role_at(X, route["ends"], k, dist, ids))
            routes.append(route)
    return {
        "bodies": len(bodies),
        "routes": routes,
        "role": routes[0]["role"] if len(routes) == 1 else "abstain",
    }
