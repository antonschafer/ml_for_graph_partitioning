import torch.nn as nn
from layers import SimpleLinear\


class AttentionAggregator(nn.Module):
    def __init__(
        self, in_features, query_input_size, out_features, queries=4, heads=4, layers=4, dropout=0, norm="layer",
    ):
        super(AttentionAggregator, self).__init__()
        assert in_features % heads == 0

        self.query_gen = nn.Sequential(
            SimpleLinear(query_input_size, queries * in_features, dropout, norm=norm),
            SimpleLinear(queries * in_features, queries * in_features, dropout, norm=norm),
            nn.Dropout(p=dropout),
            nn.Linear(queries * in_features, queries * in_features),
            # No norm here to not restrict to normalized queries
            nn.ELU(),
        )

        self.attentions = [nn.MultiheadAttention(embed_dim=in_features, num_heads=heads) for _ in range(layers)]

        combinator_in_features = query_input_size + layers * in_features
        self.combinator = nn.Sequential(
            SimpleLinear(combinator_in_features, combinator_in_features, dropout, norm=norm),
            SimpleLinear(combinator_in_features, out_features, dropout, norm=norm),
            SimpleLinear(out_features, out_features, dropout, norm=norm),
        )

        self.queries = queries
        self.layers = layers

    def forward(self, features_in, query_in):
        # TODO make batch compatible
        queries_0 = self.query_gen(query_in).view(self.queries, 1, -1)

        # query_in = torch.cat([torch.mean(embeddings, dim=0), graph_info])
        # query = self.query_gen(query_in)
        # query = query.view(self.queries, 1, -1)
        # embeddings = embeddings.view(embeddings.shape[0], 1, -1)
        # att_out = self.attention(query, embeddings, embeddings)[0]
        # att_out = torch.flatten(att_out)
        # comb_in = torch.cat([att_out, graph_info])
        # return self.combinator(comb_in)



