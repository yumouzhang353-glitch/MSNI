import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ._config import EPS, PROTOCOL


def factors(X, pairs, geometry):
    (a, b) = (pairs[:, 0], pairs[:, 1])
    v = X[b] - X[a]
    length = np.linalg.norm(v, axis=1)
    unit = np.divide(v, length[:, None], out=np.zeros_like(v), where=length[:, None] > EPS)
    sides = []
    for inv in geometry["inverse"]:
        fa = np.einsum("ni,nij,nj->n", unit, inv[a], unit)
        fb = np.einsum("ni,nij,nj->n", unit, inv[b], unit)
        sides.append(np.stack([np.where(length > EPS, fa, 1.0), np.where(length > EPS, fb, 1.0)]))
    return np.asarray(sides)


def prepare(X, r, reliability, weighted=False, tie_closed=False):
    if tie_closed:
        from ._tie_geometry import prepare_closed

        return prepare_closed(X, r, reliability, weighted)
    (n, d) = X.shape
    p = PROTOCOL["neighbor_geometry"]
    ranks = sorted(set((min(n - 1, max(3, int(r * f))) for f in [1, 2])))
    k = max(ranks)
    (_, idx) = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(X).kneighbors(X)
    ns = np.asarray([row[row != i][:k] for (i, row) in enumerate(idx)])
    covs = []
    means = []
    weights = []
    inverses = []
    dimensions = []
    confidences = []
    eigenvalues = []
    sranks = []
    scales = []
    residuals = []
    defined = []
    for rank in ranks:
        points = X[ns[:, :rank]]
        w = reliability[ns[:, :rank]] if weighted else np.ones((n, rank))
        w = w / w.sum(axis=1, keepdims=True)
        mu = np.einsum("nk,nkd->nd", w, points)
        center = points - mu[:, None, :]
        cov = np.einsum("nk,nki,nkj->nij", w, center, center)
        (vals, vecs) = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        largest = vals[:, -1]
        positive = vals > p["spectral_tolerance"] * largest[:, None]
        kept = np.where(positive, vals, 0.0)
        total = kept.sum(axis=1)
        squares = (kept**2).sum(axis=1)
        dim = np.divide(total**2, squares, out=np.ones(n), where=squares > 0)
        srank = positive.sum(axis=1)
        confidence = 1 - np.divide(dim, srank, out=np.ones(n), where=srank > 0)
        confidence = np.clip(confidence, 0, 1)
        scale = np.divide(total, dim, out=np.ones(n), where=total > 0)
        reg = np.maximum(vals, 0.1 * largest[:, None])
        ratio = np.divide(scale[:, None], reg, out=np.ones_like(reg), where=reg > 0)
        inverse = np.einsum("nik,nk,njk->nij", vecs, ratio, vecs)
        assert np.isfinite(inverse).all() and (np.linalg.eigvalsh(inverse) > 0).all()
        v = X - mu
        valid = total > 0
        res = np.divide(
            np.einsum("ni,nij,nj->n", v, inverse, v), scale, out=np.zeros(n), where=valid
        )
        covs.append(cov)
        means.append(mu)
        weights.append(w)
        inverses.append(inverse)
        dimensions.append(dim)
        confidences.append(confidence)
        eigenvalues.append(vals)
        sranks.append(srank)
        scales.append(scale)
        residuals.append(res)
        defined.append(valid)
    return dict(
        ranks=ranks,
        neighbors=ns,
        covariances=np.asarray(covs),
        means=np.asarray(means),
        weights=weights,
        inverse=np.asarray(inverses),
        dimensions=np.asarray(dimensions),
        dimension=np.mean(dimensions, axis=0),
        confidence=np.mean(confidences, axis=0),
        eigenvalues=np.asarray(eigenvalues),
        spectral_ranks=np.asarray(sranks),
        scales=np.asarray(scales),
        query_residual=np.asarray(residuals),
        query_residual_defined=np.asarray(defined),
        weighted=weighted,
    )


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
