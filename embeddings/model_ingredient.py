import os
import torch.nn as nn
from layers import (
    GCNEmbedder,
    TransformerEncoderWrapper,
    PreLNTransformerLayer,
    FullyConnected,
    PoolGCNLayer,
    CatGCNLayer,
    MixGCNLayer,
    SequenceNorm,
    ElementNorm,
    GlobalGraphAttention
)
from sacred import Ingredient
from embeddings.models import GCNTransformer, GCNAttention
from mongo.utils import get_model_files

gcn_transformer_ingredient = Ingredient("gcn_transformer_model")


@gcn_transformer_ingredient.config
def gcn_transformer_config():
    model_dims = dict(
        gcn_layers=[4, 16, 32, 32],
        n_layers_transformer=0,
        transformer_feedforward_dim=256,
        n_layers_partition=2,
        partition_dim=64,
        partition_norm=None,
    )

    gcn_params = dict(
        layer=CatGCNLayer, #MixGCNLayer #PoolGCNLayer,
        norm_pre_act=SequenceNorm,
        norm_post_act=None, #None, #ElementNorm,
    )

    dropout = dict(partition=0.2, gcn=0.2, transformer=0.1,)

    train_models = (True, True, True)

    save_fnames = [
        "embedder.pt",
        "node_transfomer.pt",
        "partition_model.pt",
    ]

    load_from_run = None
    load_fnames = (None, None, None)


@gcn_transformer_ingredient.capture
def build_model(
    model_dir, k, n_preds, parallel, model_dims, dropout, train_models, load_fnames, save_fnames, gcn_params, load_from_run
):

    if load_from_run is not None:
        load_fnames = get_model_files(run_id=load_from_run, save_dir=model_dir)
        assert len(load_fnames) == len(save_fnames)

    save_fnames = [os.path.join(model_dir, name) for name in save_fnames]

    gcn_layers = model_dims["gcn_layers"]
    assert len(gcn_layers) >= 2
    gcn = GCNEmbedder(
        in_features=gcn_layers[0],
        out_features=gcn_layers[-1],
        hidden_features=gcn_layers[1:-1],
        dropout=dropout["gcn"],
        **gcn_params
    )

    if model_dims["n_layers_transformer"] > 0:
        transformer = TransformerEncoderWrapper(
            layer=PreLNTransformerLayer(
                d_model=gcn_layers[-1],
                nhead=4,
                dim_feedforward=model_dims["transformer_feedforward_dim"],
                dropout=dropout["transformer"],
            ),
            num_layers=model_dims["n_layers_transformer"],
        )
    else:
        # No transformer
        transformer = nn.Identity()
        transformer.forward = lambda emb, feature_mask: emb

    if model_dims["n_layers_partition"] > 1:
        partition_model = nn.Sequential(
            nn.Tanh(),
            FullyConnected(
                in_features=gcn_layers[-1],
                out_features=k * n_preds,
                n_layers=model_dims["n_layers_partition"],
                dropout=dropout["partition"],
                hidden_size=model_dims["partition_dim"],
                norm=model_dims["partition_norm"],
            ),
        )
    elif model_dims["n_layers_partition"] == 1:
        partition_model = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(p=dropout["partition"]),
            nn.Linear(in_features=gcn_layers[-1], out_features=k * n_preds),
        )
    else:
        assert False

    model = GCNTransformer(
        models=[gcn, transformer, partition_model],
        save_fnames=save_fnames,
        train_models=train_models,
        load_fnames=load_fnames,
        parallel=parallel,
    )

    resources = [fname for fname in load_fnames if fname is not None]

    return model, resources
