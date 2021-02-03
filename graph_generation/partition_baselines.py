from utils import load_pickle
import tempfile
import shlex
import subprocess
import os
import networkx as nx
from utils import cut_weight_sets, cut_weight_mapping, random_partition
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

hmetis_exec = "hmetis-1.5-linux/hmetis"
greedy_exec = "greedy/run_greedy"

hmetis_cfg = dict(n_runs=1, ctype=1, vcycle=2, rtype=1, reconst=0, dbglvl=0, i=0)


def run_command(command):
    return subprocess.check_output(shlex.split(command)).decode("utf-8")


def networkx_to_hmetis(G, weighted=False):
    res = ""

    if weighted:
        res += "{} {} 1\n".format(G.number_of_edges(), len(G))
        for u, v, c in G.edges.data("weight"):
            res += "{} {} {}\n".format(c, u + 1, v + 1)
    else:
        res += "{} {}\n".format(G.number_of_edges(), len(G))
        for u, v in G.edges():
            res += "{} {}\n".format(u + 1, v + 1)
    return res


def partition_hmetis(G, g_file, imbalance_lim, k):
    cmd = hmetis_exec + " {file} {k} {imbalance} {n_runs} {ctype} {rtype} {vcycle} {reconst} {dbglvl}".format(
        file=g_file, k=k, imbalance=imbalance_lim, **hmetis_cfg,
    )
    out = run_command(cmd)
    cut_weight = None
    part_sizes = []
    for line in out.split("\n"):
        # extract result
        if "Hyperedge Cut" in line:
            for s in line.split():
                if s.isdigit():
                    cut_weight = float(s)
        elif "[" in line:
            uneven = True
            for s in line.replace("[", " ").replace("]", " ").split():
                if s.isdigit():
                    if uneven:
                        part_sizes.append(float(s))
                    uneven = not uneven
    ideal_size = len(G) / k
    imbalance = max(size / ideal_size - 1 for size in part_sizes)
    return cut_weight, imbalance


def partition_greedy(g_file):
    return float(run_command(greedy_exec + " {} 2".format(g_file)))


def partition_kl(G):
    partitions = nx.algorithms.community.kernighan_lin_bisection(G)
    return cut_weight_sets(G, partitions)


def write_to_file(G):
    g_file = os.path.join(temp_dir, "g_file")
    with open(g_file, "w") as f:
        f.write(networkx_to_hmetis(G))
    return g_file


temp_dir = tempfile.mkdtemp(dir="temp")

data_path = "data/synthetic"
runs = 5

hmetis_params = [(2, 1), (2, 5), (3, 1), (4, 1), (8, 1), (16, 1)]
random_params = [2, 3, 4, 8, 16]


if __name__ == "__main__":
    for graph_type in os.listdir(data_path):
        baseline_file = os.path.join(data_path, graph_type, "baselines.csv")
        n_graphs = len(os.listdir(os.path.join(data_path, graph_type)))
        if os.path.exists(baseline_file):
            df = pd.read_csv(baseline_file)
            n_graphs -= 1
        else:
            df = pd.DataFrame()
        print(graph_type)
        stats = defaultdict(list)
        for i in tqdm(range(n_graphs)):
            G = load_pickle(os.path.join(data_path, graph_type, "G_{}.pkl".format(i)))
            g_file = write_to_file(G)

            def compute_partitions(compute, metric_names):
                if metric_names[0] in df:
                    return

                metrics = [0 for _ in metric_names]
                for _ in range(runs):
                    res = compute()
                    for i, x in enumerate(res if len(metrics) > 1 else [res]):
                        metrics[i] += x/runs
                for name, res in zip(metric_names, metrics):
                    stats[name].append(res)

            compute_partitions(lambda: partition_kl(G), ["kl"])
            compute_partitions(lambda: partition_greedy(g_file), ["greedy"])
            for k in random_params:
                if k < len(G):
                    compute_partitions(lambda: cut_weight_mapping(G, random_partition(G, k)), ["random_{}".format(k)])
            for k, imb in hmetis_params:
                if k < len(G):
                    base_name = "hmetis_{}_{}".format(k, imb)
                    compute_partitions(lambda: partition_hmetis(G, g_file, imbalance_lim=imb, k=k), [base_name, base_name + "_imbalance"])

        for metric_name, values in stats.items():
            df[metric_name] = values
        df.to_csv(baseline_file)
