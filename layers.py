import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch_scatter


class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0, norm="batch"):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(p=dropout)

        if norm is None:
            self.norm = nn.Identity()
        else:
            if norm == "batch":
                self.norm = nn.BatchNorm1d(num_features=out_features, track_running_stats=False)
            elif norm == "layer":
                self.norm = nn.LayerNorm(out_features)
            else:
                raise ValueError('norm must be either None, "batch", or "layer"')

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.norm(x)
        return torch.relu(x)


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, hidden_size, n_layers, dropout, norm="batch"):
        super(FullyConnected, self).__init__()
        if n_layers < 2:
            raise ValueError("Need at least two layers")
        if norm is not None and norm not in ["batch", "layer"]:
            raise ValueError('norm must be either None, "batch", or "layer"')

        self.model = nn.Sequential(
            SimpleLinear(in_features=in_features, out_features=hidden_size, dropout=dropout, norm=norm),
            *[
                SimpleLinear(in_features=hidden_size, out_features=hidden_size, dropout=dropout, norm=norm)
                for _ in range(n_layers - 2)
            ],
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=out_features)
        )

    def forward(self, x):
        return self.model(x)


class ElementNorm(nn.Module):
    def __init__(self, *, elt_dim=2, eps=1e-5):
        super(ElementNorm, self).__init__()
        self.elt_dim = elt_dim
        self.eps = eps

    def forward(self, x, mask=None):
        return x / (self.eps + torch.linalg.norm(x, dim=self.elt_dim).view(*x.shape[:self.elt_dim], 1, *x.shape[self.elt_dim+1:]))


class SequenceNorm(nn.Module):
    def __init__(self, *, seq_dim=1, eps=1e-5):
        super(SequenceNorm, self).__init__()
        self.seq_dim = seq_dim
        self.eps = eps

    def forward(self, x, mask=None):
        if mask is None:
            return (x - torch.mean(x, dim=self.seq_dim, keepdim=True)) / (
                torch.std(x, dim=self.seq_dim, keepdim=True) + self.eps
            )
        else:
            mean = torch.sum(x * mask, dim=self.seq_dim, keepdim=True) / torch.sum(mask, dim=self.seq_dim, keepdim=True)
            de_meaned = x - mean
            std = torch.sqrt(
                torch.sum(torch.pow(de_meaned, 2) * mask, dim=self.seq_dim, keepdim=True)
                * 1.0
                / (torch.sum(mask, dim=self.seq_dim, keepdim=True) - 1)
            )
            return de_meaned / (std + self.eps)


class GraphConvolutionLayer(nn.Module):
    # adapted from https://github.com/tkipf/pygcn
    # Does not handle batched data!
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class KWLayer(GraphConvolutionLayer):
    def __init__(self, in_features, out_features, bias=True, sparse=False):
        super(KWLayer, self).__init__(in_features, out_features, bias)
        self.sparse = sparse
        if sparse:
            raise NotImplementedError()

    def forward(self, features, adj):
        support = torch.matmul(features, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class CatGCNLayer(GraphConvolutionLayer):
    def __init__(self, in_features, out_features, bias=True, sparse=False):
        super(CatGCNLayer, self).__init__(in_features * 2, out_features, bias)
        self.sparse = sparse

    def forward(self, features, adj):
        if len(features.shape) != 3:
            raise ValueError("Invalid number of dimensions, expected batch")
        if self.sparse:
            raise Exception("Sparse not implemented")
        else:
            aggregated = torch.matmul(adj, features)
            x = torch.cat([aggregated, features], dim=2)
            x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x


class PoolGCNLayer(GraphConvolutionLayer):
    def __init__(self, in_features, out_features, bias=True, sparse=False):
        super(PoolGCNLayer, self).__init__(in_features * 2, out_features, bias)
        self.sparse = sparse

    def forward(self, features, adj):
        if len(features.shape) != 3:
            raise ValueError("Invalid number of dimensions, expected batch")
        if self.sparse:
            raise Exception("Sparse not implemented")
        else:
            max_n = adj.shape[1]
            # repeat all node features n times. SLOW but easiest at this point
            expanded_features = features.view(features.shape[0], 1, *features.shape[1:]).expand(features.shape[0], max_n, *features.shape[1:])
            pooled = torch_scatter.scatter(src=expanded_features, index=adj.long(), reduce="max", dim=2)
            pooled = pooled[:, :, 1]  # only get max where adj = 1
            x = torch.cat([pooled, features], dim=2)
            x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x


class MixGCNLayer(GraphConvolutionLayer):
    def __init__(self, in_features, out_features, bias=True, sparse=False):
        super(MixGCNLayer, self).__init__(in_features * 3, out_features, bias)
        self.sparse = sparse

    def forward(self, features, adj):
        if len(features.shape) != 3:
            raise ValueError("Invalid number of dimensions, expected batch")
        if self.sparse:
            raise Exception("Sparse not implemented")
        else:
            max_n = adj.shape[1]
            # repeat all node features n times. SLOW but easiest at this point
            expanded_features = features.view(features.shape[0], 1, *features.shape[1:]).expand(features.shape[0], max_n, *features.shape[1:])
            pooled = torch_scatter.scatter(src=expanded_features, index=adj.long(), reduce="max", dim=2)
            pooled = pooled[:, :, 1]  # only get max where adj = 1

            summed = torch.matmul(adj, features)  # TODO mean here?

            x = torch.cat([pooled, summed, features], dim=2)
            x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x


class GCNEmbedder(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, layer, norm_pre_act, dropout, norm_post_act=None):
        super(GCNEmbedder, self).__init__()

        all_features = [in_features, *hidden_features, out_features]
        self.gcns = nn.ModuleList(
            [
                layer(in_features=all_features[i], out_features=all_features[i + 1])
                for i in range(len(all_features) - 1)
            ]
        )
        self.norms_pre = nn.ModuleList(
            [norm_pre_act() if norm_pre_act is not None else None
             for _ in range(len(all_features) - 1)]
        )
        self.norms_post = nn.ModuleList(
            [norm_post_act() if norm_post_act is not None else None
             for _ in range(len(all_features) - 1)]
        )
        self.dropout = dropout

    def forward(self, features, adj, node_mask=None):
        for gcn, norm_pre, norm_post in zip(self.gcns, self.norms_pre, self.norms_post):
            features = gcn(features, adj)
            if norm_pre is not None:
                features = norm_pre(features, node_mask)
            features = F.elu(features)
            if norm_post is not None:
                features = norm_post(features, node_mask)
        return features


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoderWrapper, self).__init__()
        self.model = nn.TransformerEncoder(layer, num_layers=num_layers,)

    def forward(self, x, mask):
        src_key_padding_mask = mask[:, :, 0] != 1 if mask is not None else None
        seq_in = x.permute(1, 0, 2)
        result = self.model(seq_in, src_key_padding_mask=src_key_padding_mask)
        result_perm = result.permute(
            1, 0, 2
        )  # permute as for transformer, the batch dimension is dimension 1 for whatever reason
        return result_perm


class PreLNTransformerLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # see paper "On Layer Normalization in the Transformer Architecture"
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)

        src = src + src2

        src2 = self.norm2(src)
        src2 = self.dropout(self.activation(self.linear1(src2)))
        src2 = self.dropout2(self.linear2(src2))

        return src + src2


class GlobalGraphAttention(nn.Module):
    def __init__(self, embedding_size, n_queries, graph_info_size, dropout, heads=4, norm="layer"):
        super(GlobalGraphAttention, self).__init__()

        self.n_queries = n_queries

        self.query_gen = nn.Sequential(
            SimpleLinear(embedding_size + graph_info_size, n_queries * embedding_size, dropout, norm=norm),
            SimpleLinear(n_queries * embedding_size, n_queries * embedding_size, dropout, norm=norm),
            nn.Dropout(p=dropout),
            nn.Linear(n_queries * embedding_size, n_queries * embedding_size),
            nn.ELU(),
        )

        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=heads)

        self.combinator = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_queries * embedding_size + graph_info_size, 2 * embedding_size),
            nn.LayerNorm(2 * embedding_size),
            nn.ELU(),
        )

    def forward(self, embeddings, graph_info):
        """
        TODO enable masking for uneven batching
        :param embeddings: batch x nodes x vector
        :param graph_info: batch x info-vec
        :return: batch x vector
        """

        assert len(embeddings.shape) == 3

        query_in = torch.cat([torch.mean(embeddings, dim=1), graph_info], dim=1)
        query = self.query_gen(query_in)  # batch x n_queries * embedding_size

        query = query.view(embeddings.shape[0], self.n_queries, embeddings.shape[2])  # batch x n_queries x embedding_size
        query_tp = query.permute(1, 0, 2)  # n_queries x batch x embedding_size

        embeddings_tp = embeddings.permute(1, 0, 2)
        att_out = self.attention(query_tp, embeddings_tp, embeddings_tp)[0]  # n_queries x batch x embedding_size
        att_out = att_out.permute(1, 0, 2).contiguous().view(embeddings.shape[0], -1)  # batch x n_queries * embedding_size
        comb_in = torch.cat([att_out, graph_info], dim=1)
        return self.combinator(comb_in)
