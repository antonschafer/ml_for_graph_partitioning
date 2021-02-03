import torch
from sacred import Ingredient
from datasets import (
    watts_strogatz_dataset,
    barabasi_albert_dataset,
    erdos_renyi_dataset,
    powerlaw_cluster_dataset,
    geometric_dataset,
    combine,
)
from torch.utils.data import DataLoader


dataset_ingredient = Ingredient("dataset")


@dataset_ingredient.config
def dataset_config():
    indices = [range(70), range(70, 80), range(80, 100)]  # graphs used for train, val, test


    adj_norm = None

    include_watts_strogatz = True
    include_barabasi_albert = True
    include_erdos_renyi = True
    include_geometric = True


    watts_strogatz_config = (
        None
        if not include_watts_strogatz
        else {
            "nks": [(100, 8), (100, 16), (100, 32), (100, 64)],
            "betas": [0.4],
            "adj_norm": adj_norm
        }
    )

    barabasi_albert_config = (
        None
        if not include_barabasi_albert
        else {
            "nms": [(100, 8), (100, 16), (100, 32), (100, 48)],
            "adj_norm": adj_norm
        }
    )

    erdos_renyi_config = (
        None
        if not include_erdos_renyi
        else {
            "sizes": [100],
            "ps": [0.1, 0.2, 0.4, 0.6],
            "adj_norm": adj_norm
        }
    )

    geometric_config = (
        None
        if not include_geometric
        else {
            "nrs": [(100, 0.2), (100, 0.4), (100, 0.6)],
            "adj_norm": adj_norm
        }
    )


@dataset_ingredient.capture
def build_dataset(
        data_dir,
        compute_features,
        sparse,
        watts_strogatz_config,
        barabasi_albert_config,
        erdos_renyi_config,
        geometric_config,
        indices
):
    datasets = []
    configs = [watts_strogatz_config, barabasi_albert_config, erdos_renyi_config, geometric_config]
    dataset_generators = [watts_strogatz_dataset, barabasi_albert_dataset, erdos_renyi_dataset, geometric_dataset]

    for split_indices in indices:
        split_datasets = []
        for config, data_gen in zip(configs, dataset_generators):
            if config is not None:
                split_datasets.append(data_gen(
                    base_dir=data_dir, compute_features=compute_features, sparse=sparse, indices=split_indices, **config
                ))
        split_dataset = combine(split_datasets)
        datasets.append(split_dataset)

    return datasets

