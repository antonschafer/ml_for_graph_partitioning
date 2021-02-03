import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ComposedModel

# TODO attention mask


class GCNTransformer(ComposedModel):
    def __init__(self, **kwargs):
        super(GCNTransformer, self).__init__(**kwargs)

    def forward(self, features, adj_gcn, node_mask=None):
        gcn, transformer, partitioner = self.models
        emb = gcn(features=features, adj=adj_gcn, node_mask=node_mask)

        emb_std = torch.mean(torch.std(emb, dim=1), dim=0)
        emb_stats = {
            "mean emb std": torch.mean(emb_std),
            "max emb std": torch.max(emb_std)
        }

        emb = transformer(emb, node_mask)
        return partitioner(emb), emb_stats


class GCNAttention(ComposedModel):
    def __init__(self, **kwargs):
        super(GCNAttention, self).__init__(**kwargs)

    def forward(self, *, features, adj_gcn, graph_info, node_mask=None):
        if node_mask is not None:
            raise NotImplementedError()

        gcn, global_attention, partitioner = self.models
        emb = gcn(features=features, adj=adj_gcn, node_mask=node_mask)

        emb_std = torch.mean(torch.std(emb, dim=1), dim=0)
        emb_stats = {
            "mean emb std": torch.mean(emb_std),
            "max emb std": torch.max(emb_std)
        }

        graph_embedding = global_attention(emb, graph_info)
        emb = torch.cat([emb, graph_embedding.view(emb.shape[0], 1, -1).repeat(1, emb.shape[1], 1)], dim=2)
        return partitioner(emb), emb_stats
