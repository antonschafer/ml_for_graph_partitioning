import numpy as np
import networkx as nx
from graph import Graph

def powerlaw_graph(n: int, m: int, p: int):
    g = nx.powerlaw_cluster_graph(n, m, p)
    print('netx done')
    nodes = g.nodes()
    node2ind = {v: i for i, v in enumerate(nodes)}
    adj_dict = {v: set(ud.keys()) for v, ud in g.adjacency()}
    return Graph(n=len(nodes), m=g.number_of_edges(), adj_list=adj_dict, nodes2index=node2ind)




