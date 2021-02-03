import random
from utils import arg_n_largest, argmax_1d_like_2d_structure, dict_to_device
from iterative_improvement.algo.algo import Algo
import copy
import torch


class AlgoQ12(Algo):
    def __init__(
        self,
        *,
        G,
        k,
        model,
        adj,
        node_features,
        use_progress,
        initial_partitioning,
        p_exploration,
        n_candidates_first,
        max_steps_per_n,
        device="cpu",
        run_all_steps,

    ):
        """
        Initialize Algorithm
        :param G: networkx graph
        :param k: number of partitions
        :param model: the model
        :param initial_partitioning: Optional,  mapping of nodes to partitions
        :param n_candidates_first: number of nodes to consider for candidate 1 for swapping
        """

        self.model = model
        self.p_exploration = p_exploration
        self.n_candidates_first = n_candidates_first
        self.use_progress = use_progress
        self.device = device
        self.static_params = {"adj": adj, "node_features": node_features, "k": k}
        self.static_in = model.collate_fn([{"params_static": self.static_params}])["static_in"]
        self.static_rep_timestamp = -1
        self.static_rep_dirty = True

        super(AlgoQ12, self).__init__(
            G=G, k=k, max_steps_per_n=max_steps_per_n, initial_partitioning=initial_partitioning, run_all_steps=run_all_steps
        )

        # assert (
        #     n_candidates_first == 1 or p_exploration == 0
        # ), "Exploration not clear when sampling more than one candidate"

    def compute_next_swap(self):
        """
        Identify the nodes that yield the highest Q value, or random nodes with prob p_explore
        """
        # determine whether to explore
        explore_first = random.random() < self.p_exploration
        explore_second = random.random() < self.p_exploration

        # Recompute static representation if needed
        if self.static_rep_dirty or self.static_rep_timestamp == -1:
            self.compute_static()
            self.static_rep_timestamp = self.steps
            self.static_rep_dirty = False

        # Compute intermediate representation
        additional_in = []
        if self.use_progress:
            additional_in.append(self.steps / self.max_steps)
        self.compute_intermediate(additional_in=additional_in)

        # Compute Q1 estimates
        all_nodes = list(self.G)
        q1_vals = self.q1(all_nodes)

        # Compute Q2 estimates
        candidate_nodes_1 = arg_n_largest(self.n_candidates_first, q1_vals)
        if explore_first:
            node_1 = random.choice(list(self.G))

            # we still need to compute the optimal prediction so we need to keep the other candidates
            candidate_nodes_1 = list(candidate_nodes_1)
            candidate_nodes_1.append(node_1)

        candidate_nodes_2 = [
            [n2 for n2 in self.G if self.node_2_partition[n2] != self.node_2_partition[n1]] for n1 in candidate_nodes_1
        ]

        q2_vals = self.q2(candidate_nodes_1, candidate_nodes_2)

        # get nodes to swap
        (idx_best_node_1, idx_best_node_2), best_pred_q2 = argmax_1d_like_2d_structure(
            q2_vals, structure=candidate_nodes_2, return_val=True
        )
        best_node_2 = candidate_nodes_2[idx_best_node_1][idx_best_node_2]
        best_node_1 = candidate_nodes_1[idx_best_node_1]
        best_pred_q1 = float(q1_vals[best_node_1])
        best_pred_q2 = float(best_pred_q2)

        if explore_first:
            # node_1 already set, get best corresponding node_2
            if explore_second:
                idx_node_2 = random.randrange(len(candidate_nodes_2[-1]))  # node_1 was last in list candidate_node_1
                node_2 = candidate_nodes_2[-1][idx_node_2]
                pred_q2 = q2_vals[len(q2_vals) - len(candidate_nodes_2[-1]) + idx_node_2]  # flat index
            else:
                (idx_node_1, idx_node_2), pred_q2 = argmax_1d_like_2d_structure(
                    q2_vals, structure=candidate_nodes_2, return_val=True, fix_i=len(candidate_nodes_1) - 1
                )
                assert idx_node_1 == len(candidate_nodes_1) - 1
                node_2 = candidate_nodes_2[idx_node_1][idx_node_2]

        else:
            node_1 = best_node_1
            if explore_second:
                idx_node_2 = random.randrange(len(candidate_nodes_2[idx_best_node_1]))
                node_2 = candidate_nodes_2[idx_best_node_1][idx_node_2]
                flat_idx_pred = sum(len(candidate_nodes_2[i]) for i in range(idx_best_node_1)) + idx_node_2
                pred_q2 = q2_vals[flat_idx_pred]
            else:
                node_2 = best_node_2
                pred_q2 = best_pred_q2

        pred_q1 = float(q1_vals[node_1])

        swap_info = dict(
            nodes=(node_1, node_2),
            node_2_partition_preswap=copy.deepcopy(self.node_2_partition),
            cut_weight_pre_swap=self.cut_weight,
            pred=(pred_q1, pred_q2),
            explore=(explore_first, explore_second),
            best_q2=best_pred_q2,
            additional_in=additional_in,
        )
        self.keep_going = best_pred_q2 >= 0

        return swap_info

    def compute_static(self):
        self.model.eval()
        with torch.no_grad():
            self.static_rep = self.model.static_representation(
                **dict_to_device(self.static_in, self.device)  # TODO keep on GPU?
            )

    def compute_intermediate(self, additional_in):
        self.model.eval()
        with torch.no_grad():
            self.intermediate = self.model.dynamic_representation(
                node_2_partition=[self.node_2_partition],
                additional_in=[additional_in],
                static_rep=dict_to_device(self.static_rep, self.device),
            )

    def q1(self, nodes):
        """
        Q function for choice of first node.
        :param nodes: candidate nodes
        :return: list of estimated lifetime rewards when choosing respective nodes for swap
        """
        self.model.eval()
        with torch.no_grad():
            return (
                self.model.q1(nodes=[nodes], dynamic_rep=dict_to_device(self.intermediate, self.device)).cpu().numpy()
            )

    def q2(self, nodes_1, nodes_2):
        """
        Q function for choice of second node.
        :param nodes_1: candidate nodes 1
        :param nodes_2: candidate nodes 2
        :return: list of lists of long term benefits of choosing candidate nodes 1 and 2 respectively
        """
        self.model.eval()
        with torch.no_grad():
            return (
                self.model.q2(
                    nodes_1=[nodes_1], nodes_2=[nodes_2], dynamic_rep=dict_to_device(self.intermediate, self.device)
                )
                .cpu()
                .numpy()
            )

    def set_static_rep_dirty(self):
        self.static_rep_dirty = True

    def get_x_r_cdn_nextq(self):
        """

        :return: prediction, reward, cut decrease normalized, next q pred to use for target computation
        """
        cut_decrease = self.last_swap["cut_weight_pre_swap"] - self.cut_weight
        cdn = cut_decrease / (self.G.number_of_edges() / len(self.G))  # divide by (m/n) to normalize
        if self.use_progress and self.done():
            next_q = 0
        else:
            next_q = self.next_swap["best_q2"]
        return self.last_swap["pred"], cdn, cdn, next_q

    def get_state_action_reward(self):
        cut_decrease = self.last_swap["cut_weight_pre_swap"] - self.cut_weight
        reward = cut_decrease / (self.G.number_of_edges() / len(self.G))  # divide by (m/n) to normalize
        return {
            "params_static": self.static_params,
            "node_2_partition": self.last_swap["node_2_partition_preswap"],
            "next_n2p": copy.deepcopy(self.node_2_partition),
            "swap": self.last_swap["nodes"],
            "reward": reward,
            "explore": self.last_swap["explore"],
            "additional_in": copy.deepcopy(self.last_swap["additional_in"]),
            "next_add_in": copy.deepcopy(self.next_swap["additional_in"]),
            "next_q": self.next_swap["best_q2"] if not (self.use_progress and self.done()) else 0,
            "G": self.G,
            "last_step": self.done(),
        }
