import numpy as np
import os
from collections import defaultdict


class Graph(object):
    def __init__(self, n, m, adj_list, adj_indexed_by_node=True, weights=None, embeddings=None, nodes=None, nodes2index=None, combinator=None):
        """

        :param n: nodes
        :param m: number edges
        :param adj_list: adj_list[i] = collection containing neighbors of node i
        :param weights: None for uniform weight 1. else: for each edge {u,v} weights[(min(u,v), max(u,v))] is the weight
                        of the edge
        :param nodes: set of node ids (integers)
        :param embeddings: embeddings[i] = node embedding of node i
        :param combinator: function that combines two node embeddings
        """
        self.n = n
        self.m = m
        if nodes is None:
            self.nodes = set(range(n))
        else:
            self.nodes = nodes

        if nodes2index is None:
            self.nodes2index = {i: i for i in range(n)}
        else:
            self.nodes2index = nodes2index

        self.edges = dict()
        if adj_indexed_by_node:
            for i in self.nodes:
                self.edges[i] = set(adj_list[i])
        else:
            for i in self.nodes:
                self.edges[i] = set(adj_list[nodes2index[i]])

        self.embeddings = embeddings
        self.mask_embeddings = np.ones(self.n, dtype=bool) #mask_embeddings[i] = True if there is an n in nodes s.t. nodes2ind[n] = i

        if weights is not None:
            self.weights = weights
        else:
            self.weights = dict()
            for u, vs in self.edges.items():
                for v in vs:
                    self.weights[(min(u, v), max(u, v))] = 1
        self.combinator = combinator
        self.total_edge_weight = sum(self.weights.values())
        #import pdb; pdb.set_trace()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Graph) and self.same_value(other)

    def same_value(self, other: object) -> bool:
        return self.n == other.n and self.m == other.m and self.edges == other.edges and self.weights == other.weights \
               and self.embeddings == other.embeddings

    def set_weight(self, u: int, v: int, c:float) -> None:
        """
        set the weight of an edge. requires that edge not in weights
        :param u: node
        :param v: node != v
        :param c: weight
        :return: None
        """
        u, v = min(u, v), max(u, v)
        if (u, v) in self.weights:
            raise(Exception('attempted to set already assigned weight'))
        self.weights[(u, v)] = c
        self.total_edge_weight += c

    def increase_weight(self, u: int, v: int, c:float) -> None:
        u, v = min(u, v), max(u, v)
        self.weights[(u, v)] += c
        self.total_edge_weight += c

    def discard_weight(self, u: int, v: int) -> float:
        """
        removes {u,v} from weights and returns its weight
        :param u: node 1
        :param v: node 2
        :return: weight of {u,v}
        """
        u, v = min(u, v), max(u, v)
        res = self.weights[(u, v)]
        del self.weights[(u, v)]
        self.total_edge_weight -= res
        return res

    def contract_edge(self, v: int, u: int) -> None:
        """
        Contract an edge in O(n)
        :param v: first node, will stay in graph, embedding updated
        :param u: second node, will be removed
        :return: None
        """
        if v == u:
            raise(ValueError("No self edges allowed"))
        elif u not in self.edges[v]:
            raise(ValueError("edge does not exist"))

        self.combine_nodes(v, u)

    def combine_nodes(self, v: int, u: int) -> None:
        """
        Combines two nodes in O(n)
        :param v: first node, will stay in graph, embedding updated
        :param u: second node, will be removed
        :return: None
        """
        if v not in self.nodes or u not in self.nodes:
            raise(ValueError("node does not exist"))

        edges_less = 0
        # update edges
        for w in self.edges[u]:
            if w in self.edges[v]:
                self.increase_weight(w, v, self.discard_weight(u, w))
                edges_less += 1
            elif w != v:
                self.edges[v].add(w)
                self.edges[w].add(v)
                self.set_weight(w, v, self.discard_weight(w, u))
            else: # w = v
                self.discard_weight(w, u)
                edges_less += 1
            self.edges[w].remove(u)

        # update n, mu
        self.n -= 1
        self.m -= edges_less

        # remove u
        self.remove_node(u)
        del self.edges[u]

        if self.embeddings is not None:
            self.combine_embeddings(v, u)

    def remove_node(self, u: int) -> None:
        """
        Does not remove adjacent edges!
        :param u: node to remove
        :return: None
        """
        self.nodes.remove(u)
        self.mask_embeddings[self.nodes2index[u]] = False

    def combine_embeddings(self, v: int, u: int) -> None:
        """
        Changes embedding of v to be combined embedding of v and u
        :param v: first node, embedding updated
        :param u: second node, embedding unchanged
        :return: None
        """
        iv, iu = self.nodes2index[v], self.nodes2index[u]
        if self.combinator:
            self.embeddings[iv] = self.combinator(self.embeddings[iv], self.embeddings[iu])
        else:
            self.embeddings[iv] = (self.embeddings[iv] + self.embeddings[iu]) / 2.

    def get_embedding(self, u):
        return self.embeddings[self.nodes2index[u]]

    def edge_embeddings_iter(self):
        for u in self.edges.keys():
            emb_u = self.get_embedding(u)
            for v in self.edges[u]:
                if v < u:
                    continue
                yield emb_u, self.get_embedding(v)

    def max_pool_embeddings(self):
        """
        :return: element-wise max of the embeddings of all nodes in the graph
        """
        return np.max(self.embeddings, axis=1, initial=float('-inf'), where=self.mask_embeddings)



def load_graph_facebook(directory: str, name: str):
    with open(os.path.join(directory, name+'.edges'), 'r') as f_edges:
        edges = defaultdict(lambda: set())
        m = 0
        for line in f_edges:
            m += 1
            u, v = tuple(map(lambda x: int(x)-0, line.split())) # -1 as 1 indexed
            edges[u].add(v)
            edges[v].add(u)
    nodes = set()
    node2ind = dict()
    with open(os.path.join(directory, name+'.feat')) as f_features:
        features_list = []
        for i, line in enumerate(f_features):
            line_content = list(map(int, line.split()))
            node = line_content[0]
            nodes.add(node)
            node2ind[node] = i
            features_list.append(line_content[1:])
        features = np.array(features_list, dtype=float)

    return Graph(features.shape[0], m, edges, embeddings=features, nodes=nodes, nodes2index=node2ind)


def load_graph_edge_list(path: str, line_start: int) -> Graph:

    max_node = 0
    with open(path, 'r') as f_edges:
        edges = defaultdict(lambda: set())
        m = 0
        for line in f_edges:
            m += 1
            if m <= line_start:
                continue
            u, v = tuple(map(lambda x: int(x)-0, line.split())) # -1 as 1 indexed
            max_node = max(max_node, u, v)
            edges[u].add(v)
            edges[v].add(u)

    return Graph(max_node+1, m, edges)

