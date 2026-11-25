import numpy as np
from sklearn.neighbors import KDTree


def nearest_retained_labels(data, labels, mask):
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels, dtype=int).copy()
    mask = np.asarray(mask, dtype=bool)
    assert labels.shape == mask.shape == (len(data),)
    clean = np.flatnonzero(~mask)
    low = np.flatnonzero(mask)
    witness = np.full(len(data), -1, dtype=int)
    if not len(low):
        return (labels, witness)
    if not len(clean) or (labels[clean] < 0).any():
        raise ValueError("Nearest redistribution requires assigned retained points")
    tree = KDTree(data[clean])
    (distances, indices) = tree.query(data[low], k=min(2, len(clean)))
    selected = clean[indices[:, 0]]
    if len(clean) > 1:
        tied = np.flatnonzero(np.isclose(distances[:, 0], distances[:, 1], rtol=1e-12, atol=1e-12))
        for row in tied:
            point = int(low[row])
            radius = float(distances[row, 0]) + max(1e-12, float(distances[row, 0]) * 1e-12)
            candidates = clean[tree.query_radius(data[point : point + 1], r=radius)[0]]
            selected[row] = min(
                (int(j) for j in candidates),
                key=lambda j: (float(np.linalg.norm(data[point] - data[j])), j),
            )
    labels[low] = labels[selected]
    witness[low] = selected
    assert not mask[selected].any()
    return (labels, witness)
