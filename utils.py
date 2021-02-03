import pickle
import numpy as np
import random
import torch


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def write_pickle(o, fname):
    with open(fname, "wb") as f:
        pickle.dump(o, f)


def arg_n_largest(n, arr):
    return np.argpartition(arr, -n)[-n:]


def argmax_2d(l, return_val=False):
    max_val = float("-inf")
    max_idx = (0, 0)
    for i, subl in enumerate(l):
        for j, x in enumerate(subl):
            if x > max_val:
                max_val = x
                max_idx = (i, j)
    if return_val:
        return max_idx, max_val
    else:
        return max_idx


def argmax_1d_like_2d_structure(arr, structure, return_val=False, fix_i=None):
    """
    find indices of max element of arr if it had the same shape as structure
    :param arr: 1d array of data
    :param structure: some 2d list
    :param return_val: True to also return max value
    :param fix_i: Only consider values in structure[fix_i] if not None
    :return: indices i, j
    """
    idx_arr = 0
    max_val = float("-inf")
    max_idx = (0, 0)
    for i, subl in enumerate(structure):
        if fix_i is not None and fix_i != i:
            idx_arr += len(subl)
            continue
        for j, _ in enumerate(subl):
            x = arr[idx_arr]
            if x > max_val:
                max_val = x
                max_idx = (i, j)
            idx_arr += 1

    if return_val:
        return max_idx, max_val
    else:
        return max_idx


def flatten(l):
    return [item for sublist in l for item in sublist]


def random_partition(G, k):
    nodes = list(G.nodes)
    assert nodes == list(range(len(G)))
    random.shuffle(nodes)
    node_2_partition = [x % k for x in nodes]
    return node_2_partition


def cut_weight_mapping(G, node_2_partition):
    return sum(node_2_partition[a] != node_2_partition[b] for a, b in G.edges)


def cut_weight_sets(G, partitions):
    assert len(partitions) == 2
    return sum((a in partitions[0]) != (b in partitions[0]) for a, b in G.edges)


def lofd_to_dofl(list_of_dict):
    if len(list_of_dict) == 0:
        raise ValueError("Empty list")
    return {k: [sample[k] for sample in list_of_dict] for k in list_of_dict[0].keys()}


def dict_to_device(d, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


def mean_max_std(vals, name):
    return {
        "mean " + name: np.mean(vals),
        "max " + name: np.max(vals),
        "min " + name: np.min(vals),
        "std " + name: np.std(vals),
    }


def pad_to_max_2d(l: list):
    assert len(l) > 0
    h_max, w_max = l[0].shape
    all_equal = True
    for t in l:
        h, w = t.shape
        all_equal = all_equal and h_max == h and w_max == w
        h_max = max(h, h_max)
        w_max = max(w, w_max)

    if all_equal:
        return torch.stack(l)
    else:
        out = torch.zeros((len(l), h_max, w_max))
        for i, t in enumerate(l):
            h, w = t.shape
            out[i, :h, :w] = t
        return out
