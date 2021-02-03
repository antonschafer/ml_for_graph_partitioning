import torch
import numpy as np
import os
import random
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver
from feature_generators import degree_sorted_features
from dataset_ingredient import dataset_ingredient, build_dataset
from embeddings.model_ingredient import gcn_transformer_ingredient, build_model
from mongo.credentials import mongo_url, db_name
import tempfile


ex = Experiment("Train direct partitioner", ingredients=[dataset_ingredient, gcn_transformer_ingredient])

ex.captured_out_filter = apply_backspaces_and_linefeeds

# add db observer
ex.observers.append(
    MongoObserver(
        url=mongo_url, db_name=db_name
    )
)

# general paths
base_dir = ""
data_dir = os.path.join(base_dir, "data/synthetic")


@ex.config
def experiment_config():
    seed = 123
    n_gpu = 1#torch.cuda.device_count()

    k = 2
    n_epochs = 100
    batch_size = 8

    # learning rate schedule
    lr = 0.001
    lr_schedule_params = dict(step_size=15, gamma=0.5)

    n_preds = 1
    # loss_weight = dict(cut=4, balance=1)
    loss_weight = dict(cut=1, balance=1)

    weight_decay = 1e-5

    accumulation_steps = 1
    log_steps = 800/batch_size

    early_stopping = 40

    device = "cuda"

    node_features = degree_sorted_features
    sparse = False

    val_batch_size = 1024

    os.makedirs("temp", exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir="temp")



@ex.capture
def set_seed(_seed):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)


@ex.capture
def load_data(sparse, node_features):
    return build_dataset(data_dir=data_dir, compute_features=node_features, sparse=sparse)


@ex.capture
def load_model(sparse, k, n_preds, n_gpu, temp_dir):
    model, resources = build_model(model_dir=temp_dir, k=k, n_preds=n_preds, parallel=n_gpu > 1)

    for fname in resources:
        ex.add_resource(fname)
    return model
