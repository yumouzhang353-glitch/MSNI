import numpy as np


def deleted_bridges(M, strong, H, active, mode):
    assert mode in {"SB_DUAL", "SB_H_ACTIVE"}
    degree = np.asarray(list(map(len, strong)))
    candidates = {
        edge
        for edge in bridges(strong)
        if degree[edge[0]] >= 3
        and degree[edge[1]] >= 3
        and (len(M[edge[0]] - set(edge) & M[edge[1]] - set(edge)) <= 1)
    }
    if mode == "SB_DUAL" or not candidates:
        return candidates
    index = np.full(len(H), -1, np.int64)
    index[active] = np.arange(len(active))
    HA = [set((int(index[j]) for j in H[int(i)] if index[j] >= 0)) for i in active]
    ha_bridges = bridges(HA)
    protected = {edge for edge in candidates if edge[1] in HA[edge[0]] and edge not in ha_bridges}
    return candidates - protected


def bridges(graph):
    n = len(graph)
    discovered = np.full(n, -1, np.int64)
    low = np.full(n, -1, np.int64)
    parent = np.full(n, -1, np.int64)
    result = set()
    tick = 0
    for root in range(n):
        if discovered[root] >= 0:
            continue
        discovered[root] = low[root] = tick
        tick += 1
        stack = [(root, iter(sorted(graph[root])))]
        while stack:
            (v, iterator) = stack[-1]
            try:
                u = next(iterator)
            except StopIteration:
                stack.pop()
                p = int(parent[v])
                if p >= 0:
                    if low[v] > discovered[p]:
                        result.add(tuple(sorted((p, v))))
                    low[p] = min(low[p], low[v])
                continue
            if u == parent[v]:
                continue
            if discovered[u] < 0:
                parent[u] = v
                discovered[u] = low[u] = tick
                tick += 1
                stack.append((u, iter(sorted(graph[u]))))
            else:
                low[v] = min(low[v], discovered[u])
    return result
