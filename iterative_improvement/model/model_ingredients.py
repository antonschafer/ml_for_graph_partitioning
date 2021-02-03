import os
import torch.nn as nn
from iterative_improvement.model.models import DoubleGCNAttend
from layers import GCNEmbedder, TransformerEncoderWrapper, PreLNTransformerLayer, FullyConnected, CatGCNLayer, PoolGCNLayer, SequenceNorm, ElementNorm
from sacred import Ingredient
from mongo.utils import get_model_files

doublegcnattend_ingredient = Ingredient("DoubleGCNAttend Model")


@doublegcnattend_ingredient.config
def doublegcnattend_config():
    embedding_size = 32
    partition_encoding = "one_hot"
    q_dropout = 0.2
    gcn_dropout = 0.2
    transformer_dropout = 0.1

    train_models = (True, True, True, True, True)
    save_fnames = [
        "static_embedder.pt",
        "dynamic_embedder.pt",
        "node_attention.pt",
        "q1_model.pt",
        "q2_model.pt",
    ]
    load_from_run = 746
    load_fnames = (None, None, None, None, None)


@doublegcnattend_ingredient.capture
def get_double_gcn_attend_submodels(k, additional_in, embedding_size, q_dropout, gcn_dropout, transformer_dropout):
    node_emb_static = GCNEmbedder(
        in_features=4,
        out_features=embedding_size,
        hidden_features=[16, 32],
        norm_pre_act=SequenceNorm,
        norm_post_act=None,
        dropout=gcn_dropout,
        layer=CatGCNLayer
    )
    node_emb_dynamic = GCNEmbedder(
        in_features=embedding_size + k * 2 + additional_in,
        out_features=embedding_size,
        hidden_features=[],
        layer=CatGCNLayer,
        norm_pre_act=SequenceNorm,
        norm_post_act=None,
        dropout=gcn_dropout,
    )
    # node_attention = nn.Identity()
    # node_attention.forward = lambda x, **kwargs: x
    node_attention = TransformerEncoderWrapper(
        layer=PreLNTransformerLayer(d_model=embedding_size, nhead=4, dim_feedforward=64, dropout=transformer_dropout),
        num_layers=4,
    )

    # q1_model = nn.Sequential(nn.Dropout(p=q_dropout), nn.Linear(in_features=embedding_size, out_features=1))
    # q2_model = nn.Sequential(
    #     nn.Dropout(p=q_dropout), nn.Linear(in_features=2 * embedding_size, out_features=1)
    # )
    q1_model = nn.Sequential(nn.Tanh(), nn.Dropout(p=q_dropout), nn.Linear(in_features=embedding_size, out_features=128), nn.ReLU(),
        nn.Dropout(p=q_dropout), nn.Linear(in_features=128, out_features=1)
                             )

    q2_model = nn.Sequential(nn.Tanh(), nn.Dropout(p=q_dropout), nn.Linear(in_features=2*embedding_size, out_features=128), nn.ReLU(),
                             nn.Dropout(p=q_dropout), nn.Linear(in_features=128, out_features=1))

    return [node_emb_static, node_emb_dynamic, node_attention, q1_model, q2_model]


@doublegcnattend_ingredient.capture
def build_doublegcnattend_model(
    model_dir, k, additional_in, train_models, load_fnames, save_fnames, partition_encoding, parallel, load_from_run
):
    # build model
    save_fnames = [os.path.join(model_dir, name) for name in save_fnames]
    if load_from_run is not None:
        # load_fnames = get_model_files(run_id=load_from_run, save_dir=model_dir)
        load_fnames = [get_model_files(run_id=load_from_run, save_dir=model_dir)[0]] + load_fnames[1:]

        assert len(load_fnames) == len(save_fnames)

    submodels = get_double_gcn_attend_submodels(k=k, additional_in=additional_in)

    model = DoubleGCNAttend(
        models=submodels,
        train_models=train_models,
        save_fnames=save_fnames,
        load_fnames=load_fnames,
        partition_encoding=partition_encoding,
        parallel=parallel,
    )
    resources = [fname for fname in load_fnames if fname is not None]
    return model, resources
