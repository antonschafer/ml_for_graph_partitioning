import os
from collections import defaultdict
import seaborn as sns
import numpy as np
from mongo.utils import get_run, get_metrics, get_results
from utils import load_pickle

data_dir = "data/synthetic"

graph_colors = {gt: c for gt, c in zip(["ER", "WS", "BA", "GEO"], sns.color_palette())}

gt_to_prefix = {
    "ER": "erdos_renyi",
    "WS": "watts_strogatz",
    "BA": "barabasi_albert",
    "GEO": "geometric"
}


def get_gt(name):
    for k, v in gt_to_prefix.items():
        if len(name) >= len(k) and name[:len(k)] == k:
            return k
    assert False


def get_density(dir_path, indices):
    densities = []
    for i in indices:
        G = load_pickle(os.path.join(dir_path, "G_{}.pkl".format(i)))
        densities.append(G.number_of_edges() / (len(G) * (len(G) - 1) / 2))
    return np.mean(densities)


