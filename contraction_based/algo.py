
class ContractAlgo:
    def __init__(self, graph, calc_q, k):
        self.graph = graph
        self.calc_q = calc_q
        self.k = k

    def compute_cut(self):
        for _ in self.step():
            pass
        return self.graph

    def step(self):
        max_pool = self.graph.max_pool_embeddings()
        best_edge, _ = self.choose_edge(max_pool)
        edge_weight = self.graph.total_edge_weight
        while self.graph.n > self.k:

            self.graph.contract_edge(best_edge)

            new_edge_weight = self.graph.total_edge_weight
            new_max_pool = self.graph.max_pool_embeddings()
            new_best_edge, new_best_reward = self.choose_edge(new_max_pool)
            reward = edge_weight - new_edge_weight

            yield max_pool, best_edge, reward, new_best_reward

            max_pool = new_max_pool
            edge_weight = new_edge_weight
            best_edge = new_best_edge

    def choose_edge(self, state):
        best_edge, best_reward = None,0
        for e in self.graph.edge_embedding_iter():
            # note that e are in order of the node ids. deal with this somehow?
            # parallelize this !?
            reward = self.compute_q(state, e, self.k)
            if reward >= best_reward:
                best_edge = e
                best_reward = reward
        return best_edge, best_reward

