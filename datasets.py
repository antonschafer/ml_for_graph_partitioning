import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import networkx as nx
from utils import load_pickle
from feature_generators import degree_rand_features
from graph_generation.generate import float_fname


def choose_from_lists(idx, lists):
    choices = []
    for l in lists:
        choices.append(l[idx % len(l)])
        idx = int(idx / len(l))
    return choices


def erdos_renyi_dataset(
    base_dir,
    compute_features,
    sizes=[10, 100, 500],
    ps=[0.1, 0.2, 0.4],
    indices=range(100),
    sparse=False,
    adj_norm=None
):
    param_lists = [sizes, ps, indices]

    def p_to_f(x):
        dir_name = "erdos_renyi_{}_{}".format(x[0], float_fname(x[1]))
        dir_path = os.path.join(base_dir, dir_name)
        fname = os.path.join(dir_path, "G_{}.pkl".format(x[2]))
        return dir_path, dir_name, fname

    return GraphDataset(
        param_lists,
        p_to_f,
        sparse=sparse,
        graph_name="erdos_renyi",
        compute_features=compute_features,
        adj_norm=adj_norm,
    )


def watts_strogatz_dataset(
    base_dir,
    compute_features,
    nks,
    betas=[0.1, 0.2, 0.4, 0.8],
    indices=range(100),
    sparse=False,
    adj_norm=None
):
    param_lists = [nks, betas, indices]

    def p_to_f(x):
        dir_name = "watts_strogatz_{}_{}_{}".format(x[0][0], x[0][1], float_fname(x[1]))
        dir_path = os.path.join(base_dir, dir_name)
        fname = os.path.join(dir_path, "G_{}.pkl".format(x[2]))
        return dir_path, dir_name, fname

    return GraphDataset(
        param_lists,
        p_to_f,
        sparse=sparse,
        graph_name="watts_strogatz",
        compute_features=compute_features,
        adj_norm=adj_norm,
    )


def barabasi_albert_dataset(
    base_dir, compute_features, nms, indices=range(100), sparse=False, adj_norm=None
):
    param_lists = [nms, indices]

    def p_to_f(p):
        dir_name = "barabasi_albert_{}_{}".format(p[0][0], p[0][1])
        dir_path = os.path.join(base_dir, dir_name)
        fname = os.path.join(dir_path, "G_{}.pkl".format(p[1]))
        return dir_path, dir_name, fname

    return GraphDataset(
        param_lists,
        p_to_f,
        sparse=sparse,
        graph_name="barabasi_albert",
        compute_features=compute_features,
        adj_norm=adj_norm,
    )


def powerlaw_cluster_dataset(
        base_dir, compute_features, nms, ps, indices=range(100), sparse=False, adj_norm=None
):
    param_lists = [nms, ps, indices]

    def p_to_f(p):
        dir_name = "powerlaw_cluster_{}_{}_{}".format(p[0][0], p[0][1], float_fname(p[1]))
        dir_path = os.path.join(base_dir, dir_name)
        fname = os.path.join(dir_path, "G_{}.pkl".format(p[2]))
        return dir_path, dir_name, fname

    return GraphDataset(
        param_lists,
        p_to_f,
        sparse=sparse,
        graph_name="powerlaw_cluster",
        compute_features=compute_features,
        adj_norm=adj_norm,
    )


def geometric_dataset(
        base_dir, compute_features, nrs, indices=range(100), sparse=False, adj_norm=None
):
    param_lists = [nrs, indices]

    def p_to_f(p):
        dir_name = "geometric_{}_{}".format(p[0][0], float_fname(p[0][1]))
        dir_path = os.path.join(base_dir, dir_name)
        fname = os.path.join(dir_path, "G_{}.pkl".format(p[1]))
        return dir_path, dir_name, fname

    return GraphDataset(
        param_lists,
        p_to_f,
        sparse=sparse,
        graph_name="geometric",
        compute_features=compute_features,
        adj_norm=adj_norm,
    )

all_dataset_generators = [
    erdos_renyi_dataset,
    watts_strogatz_dataset,
    barabasi_albert_dataset,
    geometric_dataset
]


def combine(datasets):
    class CombDataset(ConcatDataset):
        def __str__(self):
            return "combined\n\t" + "\n\t".join(sorted([str(d) for d in datasets]))

    return CombDataset(datasets)


def all_combined_dataset(base_dir, compute_features, indices=range(100), sparse=False, adj_norm=None):
    datasets = [
        gen(
            base_dir=base_dir,
            indices=indices,
            compute_features=compute_features,
            sparse=sparse,
            adj_norm=adj_norm,
        )
        for gen in all_dataset_generators
    ]
    return combine(datasets)


class GraphDataset(Dataset):
    def __init__(
        self,
        param_lists,
        param_list_to_fname,
        compute_features=degree_rand_features,
        sparse=False,
        graph_name="Unknown",
        adj_norm=None,
    ):
        self.param_lists = param_lists
        self.param_list_to_fname = param_list_to_fname
        self.sparse = sparse
        self.compute_features = compute_features
        self.graph_name = graph_name
        self.adj_norm = adj_norm

    def __str__(self):
        return self.graph_name + "  with parameters  " + str(self.param_lists)

    def __len__(self):
        res = 1
        for l in self.param_lists:
            res *= len(l)
        return res

    def __getitem__(self, idx):
        param_list = choose_from_lists(idx, self.param_lists)
        dir_path, dir_name, fname = self.param_list_to_fname(param_list)
        G = load_pickle(fname)

        hmetis_part = None
        fname_hmetis = os.path.join(dir_path, "G_{}_hmetis_2_part_5_imbalance_part.txt".format(param_list[-1]))
        if os.path.exists(fname_hmetis):
            hmetis_part = []
            with open(fname_hmetis, "r") as f:
                for i, line in enumerate(f):
                    hmetis_part.append(int(line))

        kl_part = None
        kl_path = os.path.join(dir_path, "kernighan_lin_{}".format(param_list[-1]))
        if os.path.exists(kl_path):  # check if kl cut available
            cut_kl = load_pickle(kl_path)
            kl_part = [0 for _ in range(len(G))]
            for n in cut_kl[0]:
                kl_part[n] = 0
            for n in cut_kl[1]:
                kl_part[n] = 1

        adj = self.adj_tensor(G)
        return {
            "G": G,
            "node_features": self.compute_features(G),
            "adj": adj,
            "adj_gcn": self.normalize_adj(adj),
            "paths": {"dir_path": dir_path, "dir_name": dir_name, "idx": param_list[-1], "graph_fname": fname},
            "kl_part": kl_part,
            "hmetis_part": hmetis_part,
        }

    def adj_tensor(self, G):
        if self.sparse:
            adj = nx.to_scipy_sparse_matrix(G, format="coo")
            # convert sparse scipy coo to sparse pytorch tensor
            indices = torch.LongTensor([adj.row, adj.col])
            data = torch.from_numpy(adj.data).float()
            adj = torch.sparse.FloatTensor(indices, data, adj.shape)
        else:
            adj = torch.tensor(nx.to_numpy_matrix(G), dtype=torch.float32)
        return adj

    def normalize_adj(self, adj):
        if self.sparse:
            raise NotImplementedError()
        else:
            if self.adj_norm is None:
                return adj
            neighbors = self.adj_norm.split("-")[0]
            symmetric = self.adj_norm.split("-")[1]
            if neighbors == "neighbors":
                pass
            elif neighbors == "selfloop":
                adj = adj + torch.eye(adj.shape[0])
            else:
                raise ValueError("Invalid option for adjacency normalization")

            D = torch.sum(adj, dim=1)

            if symmetric == "sym":
                # symmetrically normalize
                root_D = torch.sqrt(1/D).diag()  # dense diagonal matrix inefficient here
                return root_D @ adj @ root_D
            elif symmetric == "mean":
                return adj / D.view(D.shape[0], 1)  # correct??
            else:
                raise ValueError("Invalid option for adjacency normalization")




