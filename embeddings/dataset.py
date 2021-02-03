from utils import lofd_to_dofl, pad_to_max_2d
import torch


def collate_function(batch):
    dict_of_param_lists = lofd_to_dofl(batch)
    dict_of_param_lists["adj"] = pad_to_max_2d(dict_of_param_lists["adj"])
    dict_of_param_lists["adj_gcn"] = pad_to_max_2d(dict_of_param_lists["adj_gcn"])
    dict_of_param_lists["features"] = pad_to_max_2d(dict_of_param_lists["node_features"])
    dict_of_param_lists["node_mask"] = None  # TODO enable training different sizes
    dict_of_param_lists["n"] = torch.tensor([len(G) for G in dict_of_param_lists["G"]], dtype=torch.float32)
    dict_of_param_lists["m"] = torch.tensor(
        [G.number_of_edges() for G in dict_of_param_lists["G"]], dtype=torch.float32
    )

    return dict_of_param_lists
