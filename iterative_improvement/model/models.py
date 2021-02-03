import math
import torch
import torch.nn as nn
from typing import List
import numpy as np

from layers import SimpleLinear
from models import ComposedModel
from utils import flatten, pad_to_max_2d, dict_to_device, lofd_to_dofl


# Basically an abstract class
class IterativeImprovementModel(ComposedModel):
    def __init__(self, **kwargs):
        super(IterativeImprovementModel, self).__init__(**kwargs)

        # training information
        self.static_trained = None
        self.dynamic_trained = None

    def static_in(self, batch, device):
        pass

    def collate_fn(self, batch):
        """
        prepare batch depending on which submodels are trained
        """
        pass

    def separate_samples(self, rep):
        """
        separate batch for independent saving of samples
        :param rep: representation (static or dynamic) of batch
        :return: list of representation for every sample
        """
        pass

    def static_representation(self, adj, node_features, node_mask, k, ns):
        """
        computed once on every graph
        """
        pass

    def dynamic_representation(self, node_2_partition, additional_in, static_rep):
        """
        recomputed after each swap / step
        """
        pass

    def q1(self, dynamic_rep, nodes, separate_samples):
        """
        Compute Q1 values
        """
        pass

    def q2(self, dynamic_rep, node_1, node_2, separate_samples):
        """
        Compute Q2 values
        """
        pass

    def forward(self, batch: dict, device, q2_only=False):
        """
        Compute Q1 and Q2
        :param batch: batched input
        :param device: device to use
        :param q2_only: do not compute q1 vals
        :return: Q1, Q2
        """
        if self.static_trained:
            static_in = dict_to_device(batch["static_in"], device)
            batch["dynamic_in"]["static_rep"] = self.static_representation(**static_in)
        if self.dynamic_trained:
            dynamic_in = dict_to_device(batch["dynamic_in"], device)
            dynamic_representation = self.dynamic_representation(**dynamic_in)
            batch["q2_in"]["dynamic_rep"] = dynamic_representation
            if not q2_only:
                batch["q1_in"]["dynamic_rep"] = dynamic_representation

        q2 = self.q2(**dict_to_device(batch["q2_in"], device))
        if not q2_only:
            q1 = self.q1(**dict_to_device(batch["q1_in"], device))
            return q1, q2
        else:
            return q2


class DoubleGCNAttend(IterativeImprovementModel):
    def __init__(
        self, models, train_models, save_fnames, load_fnames, partition_encoding, parallel,
    ):
        super(DoubleGCNAttend, self).__init__(
            models=models,
            train_models=train_models,
            save_fnames=save_fnames,
            load_fnames=load_fnames,
            parallel=parallel,
        )

        (self.node_emb_static, self.node_emb_dynamic, self.node_attention, self.q1_model, self.q2_model,) = self.models

        assert len(train_models) == 5 and len(load_fnames) == 5 and len(save_fnames) == 5
        assert partition_encoding in ["one_hot"]

        # training information
        self.static_trained = train_models[0]
        self.dynamic_trained = any(train_models[:3])

        self.partition_encoding = partition_encoding

    def collate_fn(self, batch: List[dict]):
        """
        prepare batch
        :param batch: inputs for samples
        :return: batch
        """
        batch = lofd_to_dofl(batch)
        if self.static_trained:
            adj_list = [sample["adj"] for sample in batch["params_static"]]
            node_feature_list = [sample["node_features"] for sample in batch["params_static"]]
            node_features = pad_to_max_2d(node_feature_list)
            adj = pad_to_max_2d(adj_list)
            node_mask = torch.stack([torch.ones((t.shape[0], 1)) for t in node_feature_list])
            k = batch["params_static"][0]["k"]
            assert all(sample["k"] == k for sample in batch["params_static"]), "different k values not supported"
            ns = np.array([sample["node_features"].shape[0] for sample in batch["params_static"]])
            del batch["params_static"]
            batch["static_in"] = {"adj": adj, "node_features": node_features, "node_mask": node_mask, "k": k, "ns": ns}

        elif self.dynamic_trained:
            batch["dynamic_in"] = lofd_to_dofl(batch["params_dynamic"])
            del batch["params_dynamic"]
            static_reps = batch["dynamic_in"]["static_rep"]
            static_reps = lofd_to_dofl(static_reps)
            node_embeddings_static = pad_to_max_2d(static_reps["node_embeddings_static"])
            node_mask = pad_to_max_2d(static_reps["node_mask"])  # make faster?
            adj = pad_to_max_2d(static_reps["adj"])  # make faster?
            ns = np.array(static_reps["ns"])
            k = static_reps["k"][0]
            assert all(k == k_other for k_other in static_reps["k"]), "different k values not supported"
            batch["dynamic_in"]["static_rep"] = {
                "node_embeddings_static": node_embeddings_static,
                "node_mask": node_mask,
                "adj": adj,
                "ns": ns,
                "k": k,
            }

        else:
            # TODO implement training only q
            raise Exception("Not implemented")

        return batch

    def separate_samples(self, rep):
        """
        separate batch for independent saving of samples
        :param rep: representation (static or dynamic) of batch
        :return: list of representation for every sample
        """
        # TODO implement
        # separating rep into different samples. Needed when training only some parts of model and caching results
        raise Exception("Not implemented")

    def static_representation(self, adj, node_features, node_mask, k, ns):
        assert len(adj.shape) == 3
        assert len(node_features.shape) == 3
        assert len(node_mask.shape) == 3 and node_mask.shape[2] == 1

        node_embeddings_static = self.node_emb_static(features=node_features, adj=adj, node_mask=node_mask)

        return {
            "node_embeddings_static": node_embeddings_static,
            "node_mask": node_mask,
            "adj": adj,
            "ns": ns,
            "k": k,
        }

    @staticmethod
    def _compute_diff_vals(partitions, adj, node_2_partition, static_rep):
        neighbors_in_partition = torch.matmul(adj, partitions)

        max_len = max(len(x) for x in node_2_partition)
        neighbors_in_own = torch.zeros((len(node_2_partition), max_len, 1))
        for sample_idx, n2p_sample in enumerate(node_2_partition):
            for n, p in enumerate(n2p_sample):
                neighbors_in_own[sample_idx, n] = neighbors_in_partition[sample_idx, n, p]

        unnormalized = neighbors_in_partition-neighbors_in_own
        m = torch.sum(torch.sum(adj, dim=2), dim=1)
        return unnormalized / (m/torch.tensor(static_rep["ns"])).view(-1, 1, 1)

    def dynamic_representation(self, node_2_partition, additional_in, static_rep):
        device = static_rep["node_embeddings_static"].device
        max_len = max(len(x) for x in node_2_partition)
        partition_embeddings = torch.zeros((len(node_2_partition), max_len, static_rep["k"]))
        for sample_idx, n2p_sample in enumerate(node_2_partition):
            for n, p in enumerate(n2p_sample):
                partition_embeddings[sample_idx, n, p] = 1

        partition_embeddings = partition_embeddings.to(device)
        additional_in = torch.tensor(
            [[params for _ in range(n)] for n, params in zip(static_rep["ns"], additional_in)],
            dtype=torch.float32,
            device=device,
        )
        diff_vals = self._compute_diff_vals(partition_embeddings, static_rep["adj"], node_2_partition, static_rep)
        node_partition_features = torch.cat(
            [static_rep["node_embeddings_static"], partition_embeddings, additional_in, diff_vals], dim=2
        )
        node_embeddings_dynamic = self.node_emb_dynamic(
            node_partition_features, static_rep["adj"], node_mask=static_rep["node_mask"]
        )
        assert len(static_rep["node_mask"].shape) == 3
        node_embeddings_dynamic = self.node_attention(node_embeddings_dynamic, mask=static_rep["node_mask"])

        return {"node_embeddings_dynamic": node_embeddings_dynamic}

    def q1(self, dynamic_rep, nodes, separate_samples=False):

        node_embeddings_dynamic = dynamic_rep["node_embeddings_dynamic"]
        assert len(nodes) == len(node_embeddings_dynamic), "batches mismatched in length"

        # construct indices in embedding tensor
        node_indices = []
        max_n = node_embeddings_dynamic.shape[1]
        for sample_idx, nodes_sample in enumerate(nodes):
            for n in nodes_sample:
                node_indices.append(n + sample_idx * max_n)

        # flatten
        embeddings_flat = node_embeddings_dynamic.view(-1, node_embeddings_dynamic.shape[2])

        # select
        embeddings_in = embeddings_flat[node_indices]

        q1_vals = self.q1_model(embeddings_in).view(-1)
        if not separate_samples:
            return q1_vals
        else:
            result = []
            running_count = 0
            for nodes_sample in nodes:
                n_take = len(nodes_sample)
                result.append(q1_vals[running_count : running_count + n_take])
                running_count += n_take
            return result

    def q2(self, dynamic_rep, nodes_1, nodes_2, separate_samples=False):
        node_embeddings_dynamic = dynamic_rep["node_embeddings_dynamic"]
        assert len(nodes_1) == len(node_embeddings_dynamic), "batches mismatched in length"
        assert len(nodes_2) == len(node_embeddings_dynamic), "batches mismatched in length"

        # construct indices in embedding tensor
        node_indices_1 = []
        node_indices_2 = []
        max_n = node_embeddings_dynamic.shape[1]
        for sample_idx, (nodes_1_sample, nodes_2_sample) in enumerate(zip(nodes_1, nodes_2)):
            for n1, nodes_with_n1 in zip(nodes_1_sample, nodes_2_sample):
                for n2 in nodes_with_n1:
                    node_indices_1.append(n1 + sample_idx * max_n)
                    node_indices_2.append(n2 + sample_idx * max_n)

        # flatten
        embeddings_flat = node_embeddings_dynamic.view(-1, node_embeddings_dynamic.shape[2])

        # select
        embeddings_in = torch.cat([embeddings_flat[node_indices_1], embeddings_flat[node_indices_2]], dim=1)

        q2_vals = self.q2_model(embeddings_in).view(-1)
        if not separate_samples:
            return q2_vals
        else:
            result = []
            running_count = 0
            for nodes_2_sample in nodes_2:
                q2_by_n1 = []
                for nodes_with_n1 in nodes_2_sample:
                    n_take = len(nodes_with_n1)
                    q2_by_n1.append(q2_vals[running_count : running_count + n_take])
                    running_count += n_take
                result.append(q2_by_n1)
            return result
