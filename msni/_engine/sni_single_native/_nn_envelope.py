import numpy as np
from ._component_receiver import contacts


def envelope_evidence(
    pairs, ids, distance, forward, backward, eligible, support, relative, reciprocal, n
):
    (a, b) = pairs.T
    radius = np.zeros(n)
    np.maximum.at(radius, a[forward], distance[forward])
    np.maximum.at(radius, b[backward], distance[backward])
    wide = forward & (distance <= radius[b]) | backward & (distance <= radius[a])
    candidates = {
        "1": eligible & forward & backward & support,
        "3": eligible & forward & backward & support & relative,
        "5": eligible & wide & support,
    }
    arrays = dict(receiver_nn_envelope=radius, receiver_nn_envelope_direction=wide)
    quorums = {}
    for tag, candidate in candidates.items():
        (left, right, quorum) = contacts(pairs, ids, candidate)
        arrays["receiver_nn_candidate" + tag] = candidate
        arrays["receiver_nn_left" + tag] = left
        arrays["receiver_nn_right" + tag] = right
        arrays["receiver_nn_quorum" + tag] = quorum
        quorums[tag] = quorum
    extra = {
        "NN1": candidates["1"],
        "NN2": candidates["1"] & quorums["1"],
        "NN3": candidates["3"] & quorums["3"],
        "NN4": candidates["1"] & quorums["1"] & reciprocal,
        "NN5": candidates["5"] & quorums["5"],
    }
    for tag, mask in extra.items():
        arrays["receiver_nn_extra_" + tag] = mask
    return (arrays, extra)
