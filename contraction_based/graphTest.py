import unittest
import numpy as np
import random
from graph import load_graph_facebook, load_graph_edge_list, Graph
import generate_graphs
from tqdm import tqdm

def get_example_graph() -> Graph:
    a = [[2, 3, 4], [2, 3, 4], [0, 1], [0, 1, 4], [0, 1, 3]]
    weights = dict()
    for u, vs in enumerate(a):
        for v in vs:
            u_curr, v_curr = min(u, v), max(u, v)
            weights[(u_curr, v_curr)] = 1
    embeddings = [np.array([1, i]) for i in range(5)]
    return Graph(5, 7, adj_list=a, weights=weights, embeddings=embeddings)


class TestGraph(unittest.TestCase):

    def testSimple(self):
        graph = get_example_graph()
        graph.contract_edge(0, 3)

        edges_contr = {0: {1, 2, 4}, 1: {2, 0, 4}, 2: {0, 1}, 4: {0, 1}}
        weights_contr = {(0, 1): 1, (0, 2): 1, (0, 4): 2, (1, 2): 1, (1, 4): 1}

        self.assertEqual(4, graph.n)
        self.assertEqual(5, graph.m)
        self.assertEqual(edges_contr, graph.edges)
        self.assertEqual(weights_contr, graph.weights)

    def testKargerStein(self):
        return
        graph = get_example_graph()
        graph = karger_stein_simple(graph, 2)
        self.assertEqual(2, graph.n)
        self.assertEqual(1, graph.m)
        print('cut found:', graph.weights)

    def testKargerSteinK(self):
        return
        graph = get_example_graph()
        graph = karger_stein_simple(graph, 3)
        self.assertEqual(3, graph.n)
        print('cut found:', graph.weights)

    def testKargerSteinBig(self):
        return
        graph = load_graph_facebook('../data/facebook', '1912')
        graph = karger_stein_simple(graph, 3)
        print('cut found:', graph.weights)

    def testKargerSteinLiveJournal(self):
        return
        graph = load_graph_edge_list('../data/soc-LiveJournal/soc-LiveJournal1.txt', 4)
        graph = karger_stein_simple(graph, 3)
        print('cut found:', graph.weights)

    def testKargerSteinPowerlaw(self):
        graph = generate_graphs.powerlaw_graph(1000, 100, 0.3)
        print('got graph')
        graph = karger_stein_simple(graph, 3)
        print('cut found:', graph.weights)



def karger_stein_simple(graph, k):
    for i in tqdm(range(graph.n - k)):
        if len(graph.weights) == 1:
            break
        (u, v) = random.choice(list(graph.weights.keys()))
        graph.contract_edge(u, v)
    return graph


if __name__ == '__main__':
    unittest.main()