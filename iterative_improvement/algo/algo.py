import random
from utils import arg_n_largest, argmax_1d_like_2d_structure, random_partition, cut_weight_mapping
from abc import abstractmethod, ABC
import copy


class Algo(ABC):
    def __init__(
        self, *, G, k, run_all_steps, max_steps_per_n, initial_partitioning=None
    ):
        """
        Initialize Algorithm
        :param G: networkx graph
        :param k: number of partitions
        """

        self.G = G
        self.k = k

        if initial_partitioning is None:
            self.node_2_partition = random_partition(G, k)
        else:
            self.node_2_partition = copy.deepcopy(initial_partitioning)  # safety first

        # compute other representation + cut weight
        self.partitions = [set() for _ in range(k)]
        for n, p in enumerate(self.node_2_partition):
            self.partitions[p].add(n)
        self.cut_weight = cut_weight_mapping(G, self.node_2_partition)

        # initialize variables
        self.best_cut_weight = self.cut_weight
        self.steps = 0
        self.last_swap = None
        self.max_steps = max_steps_per_n * len(G)

        self.next_swap = self.compute_next_swap()

        self.run_all_steps = run_all_steps


    def swap_nodes(self, n1, n2):
        """
        Swap two nodes' partitions
        :param n1: node 1
        :param n2: node 2
        """
        p1, p2 = self.node_2_partition[n1], self.node_2_partition[n2]

        if p1 == p2:
            return 0

        # update cut weight
        diff = self.weight_diff_of_swap(n1, n2)
        self.cut_weight += diff

        self.best_cut_weight = min(self.best_cut_weight, self.cut_weight)

        # update partitions
        self.partitions[p1].remove(n1)
        self.partitions[p2].remove(n2)
        self.partitions[p2].add(n1)
        self.partitions[p1].add(n2)
        self.node_2_partition[n1] = p2
        self.node_2_partition[n2] = p1

    def weight_diff_of_swap(self, n1, n2):
        p1, p2 = self.node_2_partition[n1], self.node_2_partition[n2]

        if p1 == p2:
            return 0

        # calculate difference in cut weight
        diff = 0
        diff += sum(1 for u in self.G.neighbors(n1) if self.node_2_partition[u] == p1)
        diff += sum(1 for u in self.G.neighbors(n2) if self.node_2_partition[u] == p2)
        diff -= sum(1 for u in self.G.neighbors(n1) if self.node_2_partition[u] == p2 and u != n2)
        diff -= sum(1 for u in self.G.neighbors(n2) if self.node_2_partition[u] == p1 and u != n1)

        return diff

    @abstractmethod
    def compute_next_swap(self):
        """
        Identify the nodes that yield the highest Q value, or random nodes with prob p_explore
        has to set self.keep_going
        """
        ...

    def step(self):
        """
        make one algorithm step
        """
        if self.steps >= self.max_steps:
            raise Exception("Performed too many algo steps")

        self.swap_nodes(*self.next_swap["nodes"])

        self.steps += 1
        self.last_swap = self.next_swap
        self.next_swap = self.compute_next_swap()

    def solve(self):
        """
        Partition graph
        :param max_iter: maximum number of iterations
        :return: best cut weight and number of steps
        """
        if self.steps > 0:
            raise Exception("Trying to solve an algorithm that steps have been executed on already.")
        while not self.done():
            self.step()
        return self.best_cut_weight, self.steps

    def done(self):
        return self.steps == self.max_steps or not (self.keep_going or self.run_all_steps)
