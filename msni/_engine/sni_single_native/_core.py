import math
from collections import Counter, deque
import numpy as np
from sklearn.neighbors import KDTree
from ._assignment import nearest_retained_labels

EPS = 1e-12


def component_shape_info(data, comp):
    idx = np.asarray(list(comp), dtype=int)
    if len(idx) <= 3:
        return {
            "thin_ratio": 1.0,
            "axis": np.ones(data.shape[1], dtype=float),
            "center": np.mean(data[idx], axis=0),
        }
    X = data[idx] - np.mean(data[idx], axis=0)
    try:
        (_, s, vt) = np.linalg.svd(X, full_matrices=False)
        if len(s) < 2 or s[0] <= EPS:
            thin_ratio = 0.0
        else:
            thin_ratio = float(s[1] / (s[0] + EPS))
        axis = vt[0]
        axis = axis / (np.linalg.norm(axis) + EPS)
    except Exception:
        thin_ratio = 1.0
        axis = np.ones(data.shape[1], dtype=float)
        axis = axis / (np.linalg.norm(axis) + EPS)
    return {"thin_ratio": thin_ratio, "axis": axis, "center": np.mean(data[idx], axis=0)}


def build_raw_to_cut_component_map(component_ids_raw, components_cut):
    raw_to_cut = {}
    for cut_cid, comp in enumerate(components_cut):
        raw_ids = [int(component_ids_raw[int(i)]) for i in comp]
        raw_cid = Counter(raw_ids).most_common(1)[0][0]
        if raw_cid not in raw_to_cut:
            raw_to_cut[raw_cid] = []
        raw_to_cut[raw_cid].append(cut_cid)
    return raw_to_cut


def min_inter_component_distance(data, comp_a, comp_b):
    idx_a = np.asarray(list(comp_a), dtype=int)
    idx_b = np.asarray(list(comp_b), dtype=int)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.inf
    tree = KDTree(data[idx_b])
    (dist, _) = tree.query(data[idx_a], k=1)
    return float(np.min(dist))


def axis_similarity(axis_a, axis_b):
    axis_a = np.asarray(axis_a, dtype=float)
    axis_b = np.asarray(axis_b, dtype=float)
    return float(
        abs(np.dot(axis_a, axis_b))
        / ((np.linalg.norm(axis_a) + EPS) * (np.linalg.norm(axis_b) + EPS))
    )


def compute_raw_component_center_caps_v94(
    data, components_raw, components_cut, component_ids_raw, ref_size, local_scale=None
):
    raw_to_cut = build_raw_to_cut_component_map(component_ids_raw, components_cut)
    raw_shapes = {}
    raw_center_caps = {}
    raw_parent = {}
    for cid in range(len(components_raw)):
        raw_parent[cid] = cid
        raw_shapes[cid] = component_shape_info(data, components_raw[cid])
    significant_cut_min = max(25.0, 0.5 * float(ref_size))
    for raw_cid, comp in enumerate(components_raw):
        size = len(comp)
        shape = raw_shapes[raw_cid]
        size_cap = int(round(size / (ref_size + EPS)))
        size_cap = max(1, size_cap)
        cut_ids = raw_to_cut.get(raw_cid, [])
        cut_sizes = [len(components_cut[cid]) for cid in cut_ids]
        significant_cut_count = sum((1 for s in cut_sizes if s >= significant_cut_min))
        significant_cut_count = max(1, significant_cut_count)
        cap = min(size_cap, significant_cut_count)
        if shape["thin_ratio"] <= 0.08 and size >= 1.2 * ref_size:
            cap = 1
        raw_center_caps[raw_cid] = cap
    main_raw_ids = [
        cid for (cid, comp) in enumerate(components_raw) if len(comp) >= 0.75 * ref_size
    ]
    for raw_cid, comp in enumerate(components_raw):
        size = len(comp)
        if size >= 0.45 * ref_size:
            continue
        if len(main_raw_ids) == 0:
            continue
        best_parent = None
        best_dist = np.inf
        for pid in main_raw_ids:
            if pid == raw_cid:
                continue
            d = min_inter_component_distance(data, components_raw[raw_cid], components_raw[pid])
            if d < best_dist:
                best_dist = d
                best_parent = pid
        if best_parent is None:
            continue
        shape_small = raw_shapes[raw_cid]
        shape_big = raw_shapes[best_parent]
        same_axis = axis_similarity(shape_small["axis"], shape_big["axis"]) >= 0.9
        both_line_like = shape_small["thin_ratio"] <= 0.12 and shape_big["thin_ratio"] <= 0.12
        if local_scale is not None:
            idx_small = np.asarray(list(comp), dtype=int)
            local_gap_scale = float(np.median(local_scale[idx_small]))
        else:
            local_gap_scale = 1.0
        near_parent = best_dist <= 3.5 * (local_gap_scale + EPS)
        if both_line_like and same_axis and near_parent:
            raw_center_caps[raw_cid] = 0
            raw_parent[raw_cid] = best_parent
    cap_info = {
        "raw_shapes": raw_shapes,
        "raw_to_cut": raw_to_cut,
        "raw_parent": raw_parent,
        "significant_cut_min": significant_cut_min,
    }
    return (raw_center_caps, cap_info)


def safe_set(values):
    return set((int(v) for v in values if v is not None))


def minmax_normalize(arr):
    arr = np.asarray(arr, dtype=float)
    (mn, mx) = (np.min(arr), np.max(arr))
    if abs(mx - mn) < EPS:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn + EPS)


def euclidean_distance(data, i, j):
    return float(np.linalg.norm(data[i] - data[j]))


def _query_knn_excluding_self(data, tree, k_neighbors):
    n = len(data)
    k_query = min(n, k_neighbors + 1)
    (distances, indices) = tree.query(data, k=k_query)
    knn_indices = []
    knn_distances = []
    for i in range(n):
        row_idx = []
        row_dist = []
        for d, j in zip(distances[i], indices[i]):
            j = int(j)
            if j == i:
                continue
            row_idx.append(j)
            row_dist.append(float(d))
            if len(row_idx) >= k_neighbors:
                break
        knn_indices.append(row_idx)
        knn_distances.append(row_dist)
    return (knn_indices, knn_distances)


def _original_natural_neighbor_search(data):
    data = np.asarray(data, dtype=float)
    n = len(data)
    tree = KDTree(data)
    if n <= 1:
        return (
            0,
            [set() for _ in range(n)],
            [set() for _ in range(n)],
            np.zeros(n, dtype=int),
            tree,
        )
    kmax = min(n - 1, max(8, int(math.sqrt(n) * 4)))
    (knn_indices, _) = _query_knn_excluding_self(data, tree, kmax)
    NN = [set() for _ in range(n)]
    RNN = [set() for _ in range(n)]
    zero_history = []
    r = 0
    while True:
        r += 1
        if r > kmax:
            if kmax >= n - 1:
                break
            kmax = min(n - 1, kmax * 2)
            (knn_indices, _) = _query_knn_excluding_self(data, tree, kmax)
        for i in range(n):
            if r - 1 < len(knn_indices[i]):
                j = int(knn_indices[i][r - 1])
                NN[i].add(j)
                RNN[j].add(i)
        nb = np.asarray([len(RNN[i]) for i in range(n)], dtype=int)
        zero_count = int(np.sum(nb == 0))
        if zero_count == 0:
            break
        zero_history.append(zero_count)
        if len(zero_history) >= 3 and zero_history[-1] == zero_history[-2] == zero_history[-3]:
            break
        if r >= n - 1:
            break
    nb = np.asarray([len(RNN[i]) for i in range(n)], dtype=int)
    return (r, NN, RNN, nb, tree)


def build_mng(NN, RNN):
    n = len(NN)
    M = []
    S = []
    gamma = []
    for i in range(n):
        nn_i = safe_set(NN[i])
        rnn_i = safe_set(RNN[i])
        m_i = nn_i | rnn_i
        s_i = nn_i & rnn_i
        M.append(m_i)
        S.append(s_i)
        gamma.append(set([i]) | m_i)
    return (M, S, gamma)


def compute_reliability_from_nb(nb, eps=EPS):
    nb = np.asarray(nb, dtype=float)
    nb_log = np.log1p(nb)
    max_v = np.max(nb_log)
    if max_v <= eps:
        return np.ones_like(nb_log)
    reliability = nb_log / (max_v + eps)
    reliability = np.maximum(reliability, eps)
    return reliability


def compute_local_scale_from_gamma(data, gamma, eps=EPS):
    n = len(data)
    local_scale = np.zeros(n, dtype=float)
    all_scales = []
    for i in range(n):
        dists = []
        for u in gamma[i]:
            u = int(u)
            if u == i:
                continue
            d = euclidean_distance(data, i, u)
            if d > eps:
                dists.append(d)
        if len(dists) > 0:
            local_scale[i] = float(np.mean(dists))
            all_scales.append(local_scale[i])
        else:
            local_scale[i] = 0.0
    if len(all_scales) > 0:
        fallback = float(np.median(all_scales))
    else:
        fallback = 1.0
    fallback = max(fallback, eps)
    local_scale[local_scale <= eps] = fallback
    return local_scale


def weighted_common_neighbor_similarity(gamma_i, gamma_j, reliability, eps=EPS):
    common = gamma_i & gamma_j
    if len(common) == 0:
        return 0.0
    common_weight = sum((reliability[int(u)] for u in common))
    weight_i = sum((reliability[int(u)] for u in gamma_i))
    weight_j = sum((reliability[int(u)] for u in gamma_j))
    return float(common_weight / (math.sqrt(weight_i * weight_j) + eps))


def adaptive_alpha(i, j, reliability, alpha_min=1.3, alpha_max=2.5):
    ri = reliability[int(i)]
    rj = reliability[int(j)]
    alpha_i = alpha_min + (1.0 - ri) * (alpha_max - alpha_min)
    alpha_j = alpha_min + (1.0 - rj) * (alpha_max - alpha_min)
    return float((alpha_i + alpha_j) / 2.0)


def compute_structure_similarity(M, gamma, reliability):
    sigma = {}
    n = len(M)
    for i in range(n):
        for j in M[i]:
            j = int(j)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key not in sigma:
                sigma[key] = weighted_common_neighbor_similarity(gamma[i], gamma[j], reliability)
    return sigma


def get_sigma(sigma, i, j):
    if i == j:
        return 1.0
    return float(sigma.get((min(i, j), max(i, j)), 0.0))


def _distance_density(data, i, neighbors):
    neighbors = list(neighbors)
    if len(neighbors) == 0:
        return 0.0
    dist_sum = 0.0
    for j in neighbors:
        dist_sum += euclidean_distance(data, i, j)
    return len(neighbors) / (dist_sum + EPS)


def compute_dsngcap_density(data, tree, N):
    data = np.asarray(data, dtype=float)
    n = len(data)
    N = int(max(1, min(N, n - 1)))
    density = np.zeros(n, dtype=float)
    for i in range(n):
        (dist, idx) = tree.query([data[i]], k=N + 1)
        dist_list = dist[0][1:]
        dist_sum = np.sum(dist_list)
        if dist_sum <= EPS:
            density[i] = 100000000.0
        else:
            density[i] = N / (dist_sum + EPS)
    return density


def compute_diff1_by_density(density):
    sorted_density = np.sort(density)
    diff1 = []
    for i in range(len(sorted_density) - 1):
        diff1.append(sorted_density[i + 1] - sorted_density[i])
    return np.asarray(diff1, dtype=float)


def dsngcap_var_compute(diff1, scale):
    diff1 = list(diff1)
    if len(diff1) == 0:
        return np.asarray([0.0])
    diff1.append(diff1[-1])
    step = max(1, len(diff1) // scale)
    var = []
    for i in range(scale):
        start = i * step
        end = min((i + 1) * step, len(diff1))
        if start >= len(diff1):
            var.append(0.0)
        else:
            temp_diff1 = diff1[start:end]
            var.append(float(np.var(temp_diff1)))
    return np.asarray(var, dtype=float)


def dsngcap_get_alpha(var, scale):
    if len(var) == 0:
        return 0
    standard = scale // 2
    var = list(var[:standard])
    if len(var) == 0:
        return 0
    if max(var) <= EPS:
        return 0
    max_index = int(np.argmax(var))
    var_range = max(var) - var[-1]
    k = max(1, standard - max_index)
    over = []
    for i in range(1, k + 1):
        step_value = var_range / k * i
        p = 0
        for j in range(standard - 1, max_index - 1, -1):
            if step_value > var[j]:
                p += 1
            else:
                break
        over.append(p)
    if len(over) == 0:
        return 0
    enhance = []
    enhance.append(over[0])
    for i in range(1, len(over)):
        if over[i] == 0 or over[i - 1] == 0:
            enhance.append(1.0)
        else:
            enhance.append(over[i] / (over[i - 1] + EPS))
    a = int(np.argmax(enhance))
    alpha = standard - over[a]
    if max_index != 0:
        alpha -= 1
    alpha = max(0, alpha)
    if scale == 100:
        return int(alpha)
    else:
        return int(alpha * 10)


def detect_low_density_points_by_dsngcap_density(data, tree, r, nb):
    n = len(data)
    if n <= 5:
        return (np.asarray([], dtype=int), np.arange(n), 0, None)
    N = int((r + int(np.max(nb))) // 2)
    N = max(1, min(N, n - 1))
    density = compute_dsngcap_density(data, tree, N)
    diff1 = compute_diff1_by_density(density)
    if n >= 500:
        scale = 100
    else:
        scale = 10
    var = dsngcap_var_compute(diff1, scale)
    alpha = dsngcap_get_alpha(var, scale)
    sorted_idx = np.argsort(density)
    low_len = int(n * alpha / 100)
    low_len = max(0, min(low_len, n - 1))
    low_indices = np.asarray(sorted_idx[:low_len], dtype=int)
    clean_indices = np.asarray(sorted_idx[low_len:], dtype=int)
    denoise_info = {
        "N": N,
        "density": density,
        "diff1": diff1,
        "var": var,
        "scale": scale,
        "alpha": alpha,
    }
    return (low_indices, clean_indices, alpha, denoise_info)


def detect_low_density_points_by_nb_adaptive(nb, r, eps=EPS):
    nb = np.asarray(nb, dtype=float)
    n = len(nb)
    if n <= 5:
        return (np.asarray([], dtype=int), np.arange(n), 0, {"k_auto": 0})
    nb_positive = nb[nb > eps]
    if len(nb_positive) > 0:
        k_auto = max(1, int(np.percentile(nb_positive, 15)))
    else:
        k_auto = 0
    noise_mask = nb <= k_auto
    low_indices = np.where(noise_mask)[0].astype(int)
    max_remove = int(n * 0.3)
    if len(low_indices) > max_remove:
        sorted_idx = np.argsort(nb)
        low_indices = sorted_idx[:max_remove].astype(int)
    clean_indices = np.setdiff1d(np.arange(n, dtype=int), low_indices).astype(int)
    alpha = int(round(len(low_indices) / n * 100))
    denoise_info = {"k_auto": k_auto, "method": "nb_adaptive"}
    return (low_indices, clean_indices, alpha, denoise_info)


def weak_connected_components(M):
    n = len(M)
    undirected = [set() for _ in range(n)]
    for i in range(n):
        for j in M[i]:
            j = int(j)
            if i == j:
                continue
            undirected[i].add(j)
            undirected[j].add(i)
    component_ids = np.full(n, -1, dtype=int)
    components = []
    comp_id = 0
    for start in range(n):
        if component_ids[start] != -1:
            continue
        q = deque([start])
        component_ids[start] = comp_id
        comp = set([start])
        while q:
            i = q.popleft()
            for j in undirected[i]:
                if component_ids[j] == -1:
                    component_ids[j] = comp_id
                    comp.add(j)
                    q.append(j)
        components.append(comp)
        comp_id += 1
    return (component_ids, components)


def robust_reference_component_size(components_cut):
    if len(components_cut) == 0:
        return 1.0
    sizes = np.asarray([len(c) for c in components_cut], dtype=float)
    if len(sizes) == 1:
        return max(float(sizes[0]), 1.0)
    q50 = float(np.percentile(sizes, 50))
    main_sizes = sizes[sizes >= q50]
    if len(main_sizes) == 0:
        main_sizes = sizes
    ref_size = float(np.median(main_sizes))
    return max(ref_size, 1.0)


def _scaled_edge_distance(data, local_scale, i, j):
    dij = euclidean_distance(data, i, j)
    return dij / (math.sqrt(local_scale[i] * local_scale[j]) + EPS)


def build_component_mng_by_strong_edges(data, M, S, G, local_scale):
    n = len(M)
    row_thr = np.full(n, np.inf, dtype=float)
    for i in range(n):
        vals = [G.get((i, int(j)), 0.0) for j in M[i] if G.get((i, int(j)), 0.0) > EPS]
        if len(vals) > 0:
            row_thr[i] = float(np.percentile(vals, 30))
    snn_dscaled = []
    for i in range(n):
        for j in S[i]:
            j = int(j)
            if i == j:
                continue
            snn_dscaled.append(_scaled_edge_distance(data, local_scale, i, j))
    if len(snn_dscaled) > 0:
        tau_mutual = float(np.percentile(snn_dscaled, 95))
        tau_strong = float(np.percentile(snn_dscaled, 85))
    else:
        tau_mutual = 1.8
        tau_strong = 1.5
    M_cc = [set() for _ in range(n)]
    for i in range(n):
        for j in M[i]:
            j = int(j)
            if i == j:
                continue
            d_scaled = _scaled_edge_distance(data, local_scale, i, j)
            raw_ij = G.get((i, j), 0.0)
            raw_ji = G.get((j, i), 0.0)
            mutual = j in S[i]
            keep_mutual = (
                mutual
                and d_scaled <= tau_mutual
                and (max(raw_ij, raw_ji) >= min(row_thr[i], row_thr[j]))
            )
            keep_strong = raw_ij >= row_thr[i] and raw_ji >= row_thr[j] and (d_scaled <= tau_strong)
            if keep_mutual or keep_strong:
                M_cc[i].add(j)
                M_cc[j].add(i)
    return M_cc


def dpc_assign_remaining_clean_points_v94(
    data, labels, rho, component_ids_cut, component_ids_raw, raw_parent=None
):
    labels = labels.copy()
    n = len(data)
    if raw_parent is None:
        raw_parent = {}
    order = np.argsort(-rho)
    for i in order:
        i = int(i)
        if labels[i] != -1:
            continue
        cut_i = int(component_ids_cut[i])
        raw_i = int(component_ids_raw[i])
        parent_i = int(raw_parent.get(raw_i, raw_i))
        same_cut = np.where(component_ids_cut == cut_i)[0]
        candidates = [int(j) for j in same_cut if rho[int(j)] > rho[i] and labels[int(j)] != -1]
        if len(candidates) > 0:
            nearest = min(candidates, key=lambda j: euclidean_distance(data, i, j))
            labels[i] = labels[nearest]
            continue
        same_cut_labeled = [int(j) for j in same_cut if labels[int(j)] != -1]
        if len(same_cut_labeled) > 0:
            nearest = min(same_cut_labeled, key=lambda j: euclidean_distance(data, i, j))
            labels[i] = labels[nearest]
            continue
        same_raw = np.where(component_ids_raw == raw_i)[0]
        same_raw_labeled = [int(j) for j in same_raw if labels[int(j)] != -1]
        if len(same_raw_labeled) > 0:
            nearest = min(same_raw_labeled, key=lambda j: euclidean_distance(data, i, j))
            labels[i] = labels[nearest]
            continue
        same_parent = np.where(component_ids_raw == parent_i)[0]
        same_parent_labeled = [int(j) for j in same_parent if labels[int(j)] != -1]
        if len(same_parent_labeled) > 0:
            nearest = min(same_parent_labeled, key=lambda j: euclidean_distance(data, i, j))
            labels[i] = labels[nearest]
            continue
        global_labeled = np.where(labels != -1)[0]
        if len(global_labeled) > 0:
            nearest = min(global_labeled, key=lambda j: euclidean_distance(data, i, int(j)))
            labels[i] = labels[nearest]
        else:
            labels[i] = 0
    return labels


def compute_original_graph_info(data):
    (r, NN, RNN, nb, tree) = natural_neighbor_search(data)
    return {"r": r, "NN": NN, "RNN": RNN, "nb": nb, "tree": tree}


def compute_composite_density(data, NN, RNN, M, S, G):
    rho_nn = np.asarray([_distance_density(data, i, NN[i]) for i in range(len(data))], dtype=float)
    rho = minmax_normalize(rho_nn)
    return (rho, {"rho_nn": rho_nn, "rho": rho})


def compute_gravity(data, M, sigma, reliability, local_scale):
    n = len(data)
    G = {}
    for i in range(n):
        for j in M[i]:
            j = int(j)
            if i == j:
                continue
            sij = get_sigma(sigma, i, j)
            if sij <= EPS:
                gij = 0.0
            else:
                dij = euclidean_distance(data, i, j)
                scale = math.sqrt(local_scale[i] * local_scale[j]) + EPS
                d_scaled = dij / scale
                alpha_ij = adaptive_alpha(i, j, reliability, alpha_min=1.0, alpha_max=2.0)
                rel_ij = min(reliability[i], reliability[j])
                beta_local = 1.0 + (1.0 - rel_ij)
                gij = sij**beta_local * math.exp(-(d_scaled**2)) / (d_scaled + EPS) ** alpha_ij
            G[i, j] = gij
    return G


def select_components_by_budget(components_cut, component_ids_raw, components_raw, raw_center_caps):
    raw_to_cut = build_raw_to_cut_component_map(component_ids_raw, components_cut)
    records = []
    for raw_cid in range(len(components_raw)):
        cut_ids = sorted(
            raw_to_cut.get(raw_cid, []), key=lambda cid: len(components_cut[cid]), reverse=True
        )
        quota = int(raw_center_caps.get(raw_cid, 1))
        assert 0 <= quota <= len(cut_ids)
        for cut_cid in cut_ids[:quota]:
            points = components_cut[cut_cid]
            records.append(
                {
                    "representative": int(min(points)),
                    "raw_cid": raw_cid,
                    "cut_cid": cut_cid,
                    "cut_size": len(points),
                }
            )
    assert all(
        (
            sum((rec["raw_cid"] == cid for rec in records)) == quota
            for (cid, quota) in raw_center_caps.items()
        )
    )
    return records


from ._tie_neighborhood import natural_neighbor_search
