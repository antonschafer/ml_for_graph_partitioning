import math
import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from utils import write_pickle

BASE_DIR = "."
synthetic_dir = os.path.join(BASE_DIR, "data", "synthetic")


def generate_connected(generator, curr_dir, indices, max_attempts=10):
    for i in tqdm(indices):
        attempts = 1
        G = generator()
        while not nx.is_connected(G):
            if attempts >= max_attempts:
                raise Exception("Could not generate connected graph")
            else:
                G = generator()
                attempts += 1
        write_pickle(G, os.path.join(curr_dir, "G_{}.pkl".format(i)))


def float_fname(x):
    """
    make compatible to be part of file name
    :param x: number
    :return: string representation w/o .
    """
    return str(x).replace(".", "-")


def generate_erdos_renyi(nodes, probs, indices):
    for n in nodes:
        for p in probs:
            print("generating ER graphs n={}, p={}".format(n, p))
            curr_dir = os.path.join(synthetic_dir, "erdos_renyi_{}_{}".format(n, float_fname(p)))
            os.makedirs(curr_dir, exist_ok=True)
            generate_connected(lambda: nx.fast_gnp_random_graph(n, p), curr_dir, indices)


def generate_barabasi_albert(nms, indices):
    for n, m in nms:
        print("generating BA graphs n={}, m={}".format(n, m))
        curr_dir = os.path.join(synthetic_dir, "barabasi_albert_{}_{}".format(n, m))
        os.makedirs(curr_dir, exist_ok=True)
        generate_connected(lambda: nx.barabasi_albert_graph(n, m), curr_dir, indices)


def generate_watts_strogatz(nks, betas, indices):
    for n, k in nks:
        for beta in betas:
            print("generating WS graphs n={}, k={}, beta={}".format(n, k, beta))
            curr_dir = os.path.join(synthetic_dir, "watts_strogatz_{}_{}_{}".format(n, k, float_fname(beta)))
            os.makedirs(curr_dir, exist_ok=True)
            generate_connected(lambda: nx.watts_strogatz_graph(n, k, beta), curr_dir, indices)


def generate_powerlaw_cluster(nms, ps, indices):
    for n, m in nms:
        for p in ps:
            assert p % 0.1 == 0
            print("Generating Powerlaw Cluster graphs n={}, m={}, p={}".format(n, m, p))
            curr_dir = os.path.join(synthetic_dir, "powerlaw_cluster_{}_{}_{}".format(n, m, float_fname(p)))
            os.makedirs(curr_dir, exist_ok=True)
            generate_connected(lambda: nx.powerlaw_cluster_graph(n=n, m=m, p=p), curr_dir, indices)


def generate_geometric(ns, rs, indices, dim=2):
    for i, n in enumerate(ns):
        for r in rs[i]:
            print("Generating Geometric graphs n={}, radius={}".format(n, r))
            curr_dir = os.path.join(synthetic_dir, "geometric_{}_{}".format(n, float_fname(r)))
            os.makedirs(curr_dir, exist_ok=True)
            generate_connected(lambda: nx.random_geometric_graph(n=n, radius=r, dim=dim), curr_dir, indices)


def compute_radius_geo(d, dim=2):
    """
    Compute the radius for geometric graphs to reach given density
    :param d:
    :param dim:
    :return:
    """
    mid = math.sqrt(d / math.pi)
    def f(r, s=10):
        ds = []
        for _ in range(100):
            G = nx.random_geometric_graph(n=10, radius=r, dim=dim)
            ds.append(G.number_of_edges() / (len(G) * (len(G) - 1) / 2))
        return np.mean(ds)
    diff = f(mid) - d
    left = mid - 0.2
    right = mid + 0.2
    while abs(diff) > 0.01:
        if diff > 0:
            right = mid
        else:
            left = mid
        mid = (left + right) / 2
        diff = f(mid) - d
    r = mid
    digits_round = int(- math.log(r) / math.log(10)) + 2
    return round(r, digits_round)


def compute_m_ba(n, d):
    a = -1
    b = n
    c = -n * (n-1) * d / 2
    return round((-b+math.sqrt(b**2 - 4 * a * c)) / (2 * a))


if __name__ == "__main__":
    generate_erdos_renyi([100], [0.1, 0.2, 0.4, 0.6], range(100))
    generate_watts_strogatz([(100, 8), (100, 16), (100, 32), (100, 64)], [0.1, 0.2, 0.4, 0.8], range(100))
    generate_barabasi_albert([(100, 2), (100, 4), (100, 8), (100, 16), (100, 32), (100, 48)], range(100))
    generate_geometric(ns=[100], rs=[[0.2, 0.4, 0.6]], indices=range(100))

    # Generate graphs of various sizes with same density
    n_generate = 20
    density = 0.3  # <= ~0.5 for BA
    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]
    generate_erdos_renyi(sizes, [density], range(n_generate))
    generate_watts_strogatz([(s, round(density * (s-1))) for s in sizes], betas=[0.4], indices=range(n_generate))

    generate_barabasi_albert([(s, compute_m_ba(s, density)) for s in sizes], indices=range(n_generate))  # no s-1. this accounts for initial steps

    r = compute_radius_geo(density)
    generate_geometric(ns=sizes, rs=[[r] for s in sizes], indices=range(n_generate))
