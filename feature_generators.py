import torch
import numpy as np


def ones_features(G):
    return torch.ones(len(G), 4)


def degree_ones_features(G):
    """
    feature = node degree / n, 1, 1, 1
    :param G: networkx graph
    :return: array of node features
    """
    degrees = [G.degree[n] for n in G]
    features = torch.ones(len(G), 4)
    features[:, 0] = torch.tensor(degrees, dtype=torch.float32) / len(G)
    return features


def degree_rand_features(G):
    """
    feature = node degree / n, 1, random, random
    :param G: networkx graph
    :return: array of node features
    """
    features = degree_ones_features(G)
    features[:, 2:4] = torch.rand(len(G), 2)
    return features


def argsort_ties_random(arr):
    rand = np.random.random(len(arr))
    return np.lexsort((rand, arr))


def degree_sorted_features(G):
    """
    feature = node degree / n, position of node in ordering by degree / (n-1), 1, 1
    :param G: networkx graph
    :return: array of node features
    """
    degrees = [G.degree[n] for n in G]
    order_id = np.zeros(len(G))
    for pos, n in enumerate(argsort_ties_random(degrees)):
        order_id[n] = pos * 1.0 / (len(G) - 1)

    features = degree_ones_features(G)
    features[:, 1] = torch.tensor(order_id, dtype=torch.float32) / len(G)
    return features


def degree_sorted_random_features(G):
    """
    feature = node degree / n, position of node in ordering by degree / (n-1), random, 1
    :param G: networkx graph
    :return: array of node features
    """
    features = degree_sorted_features(G)
    features[:, 2] = torch.rand(len(G))
    return features


all_generators = [degree_ones_features, degree_rand_features, degree_sorted_random_features, degree_sorted_features]
generator_dict = {g.__name__: g for g in all_generators}
