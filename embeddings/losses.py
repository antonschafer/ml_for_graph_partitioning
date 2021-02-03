import torch
import torch.nn as nn


def gap_cw_loss(Y, adj):
    """
    Expected cut weight normalized by partition volume (sum of degrees)
    :param Y: (b, n, k) matrix where Y[s, i, p] = P[node i in partition p] for sample s
    :param adj: (b, n, n) (sparse) adjacency matrix
    """
    B = torch.matmul(adj, (1 - Y))  # b x n x k: B[b, i, p] = sum over all of i's neighbors prob. that they're not in partition p
    D = torch.sum(adj, dim=2)  # b x n node degrees
    V = torch.matmul(Y.transpose(1, 2), D.view(*D.shape, 1)).squeeze(-1)  # b x k  expected sum of node degrees in partition
    CW = Y*B  # b x n x k : B[b, i, p] = sum over all of i's neighbors prob. that they're not in partition p but i is
    CP = torch.sum(CW, dim=1) # b x k expected edges between partition and rest that are cut
    NCP = CP / V # b x k normalized cut weight of partitions
    cut_weight = torch.sum(NCP, dim=1)
    return torch.mean(cut_weight)


def cut_weight_loss(Y, adj, m):
    """
    Expected cut weight
    :param Y: (b, n, k) matrix where Y[s, i, p] = P[node i in partition p] for sample s
    :param adj: (b, n, n) (sparse) adjacency matrix
    :param m: (b, 1) number of edges
    :return: E[weight of cut] / avg_deg

    b * O(n*k + m)
    """
    k = Y.shape[2]
    B = torch.matmul(adj, (1 - Y))  # B[i,p] = sum over all of i's neighbors prob. that they're not in partition p
    cut_weight = torch.sum(torch.sum(Y * B, dim=1), dim=1)/2
    # avg_deg = (2.*m)/ adj.shape[0] # TODO does shape also work for sparse tensors?
    # return cut_weight/avg_deg
    return torch.mean(cut_weight / (m * ((k-1)/k)))


def balance_loss(Y, n):
    """
    Expected balancing MSE
    :param Y: batch size times (n, k) matrix where Y[sample, i, p] = P[node i in partition p]
    :param n: Number of nodes in each graph.
    :return: mean of MSEs of expected #nodes in partition / n  and 1/k

    b * O(n*k)
    b * (n*k + 2k operations)
    """
    assert len(Y.shape) == 3
    k = Y.shape[2]
    ideal_weights = (n/k).view(-1, 1)
    eps = torch.sum(Y, dim=1)/ideal_weights - 1
    eps *= 10
    return torch.mean(eps * eps)
