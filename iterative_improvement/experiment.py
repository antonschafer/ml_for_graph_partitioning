import os
import torch
import random
import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver
from iterative_improvement.model.model_ingredients import (
    doublegcnattend_ingredient,
    build_doublegcnattend_model,
)
from dataset_ingredient import (
    dataset_ingredient,
    build_dataset,
)
from feature_generators import degree_sorted_features
from scheduler import CustomScheduler
import tempfile
from math import sqrt
from mongo.credentials import mongo_url, db_name

ex = Experiment("II Final Experiment v4", ingredients=[dataset_ingredient, doublegcnattend_ingredient])

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

    variation = "Input diff vals norm"

    time_lim = 23.8  # in hours

    seed = 123
    n_gpu = 1  # torch.cuda.device_count()

    exploration_schedule = dict(
        vals=[1 - sqrt(1 - x) for x in [0.8, 0.6, 0.4, 0.2, 0.1, 0.01]],
        steps=[x * (2 ** 11) for x in [0, 40, 55, 70, 85, 100]],
    )
    exploration_scheduler = CustomScheduler(**exploration_schedule)

    lr_schedule = dict(step_size=120000, gamma=0.5)

    steps_per_node = 0.5

    initial_partition = "random"
    train_config = {
        # algorithm config
        "p_exploration": lambda schedulers: schedulers[
            "exploration"
        ].val(),  # greedy policy with probability 1-p_exploration
        "steps_per_node": steps_per_node,
        "n_candidates_first": 1,
        "initial_partition": initial_partition,
        # train config
        "epochs": int(10 * (0.5/steps_per_node)),
        "lr": 0.001,
        "batch_size": 128,
        "weight_decay": 1e-5,
        "weight_loss_q1": 1,
        "weight_loss_q2": 1,
        "save_report": True,
        "n_workers_dl": 8,
    }

    set_target_steps = 2 ** 11

    dqn_config = {
        "r_steps": 3,
        "max_age": set_target_steps,  # Max age of the target values in global algo steps
        "train_interval": 2 ** 9,  # 32,  # introduce schedule
        "n_batches": 2 ** 12 // train_config["batch_size"],  # 32,  # introduce schedule
        "use_src_pred": False,
        "set_target_steps": set_target_steps,
        "min_replay_size": 2 ** 11,
        "replay_mem_size": 2 ** 17,
        "log_steps": 2 ** 11,
        "early_stopping": float("inf"),
        "freq_qplot": 2,
        "max_static_age": 0, # maximum no of retrains for which static rep of algo not updated
    }

    test_config = {
        "max_steps_per_node": steps_per_node,
        "n_swaps_parallel": 1,
        "n_candidates_first": 8,
        "initial_partition": initial_partition,
    }

    algo_config = {
        "k": 2,
        "discount_factor": 0.9,
    }

    q_plot_config = {
        "graph_wise_single": False,
        "graph_wise_mean": False,
        "all_single": True,
        "all_mean": True,

    }

    device_algo = "cpu"  # "cuda"
    device_train = "cpu"  # "cuda"
    device_target = "cpu"  # "cuda"
    device_test = "cpu"  # "cuda" TODO

    use_progress = True
    node_features = degree_sorted_features
    sparse = False

    # Schedulers
    schedulers = dict(exploration=exploration_scheduler)

    os.makedirs("temp", exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir="temp")


@ex.capture
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@ex.capture
def load_model(sparse, use_progress, n_gpu, temp_dir, algo_config):
    model, resources = build_doublegcnattend_model(
        model_dir=temp_dir, parallel=n_gpu > 1, k=algo_config["k"], additional_in=int(use_progress)
    )
    # model, resources = build_greedyalgo_model(model_dir=model_dir)

    for fname in resources:
        ex.add_resource(fname)
    return model


@ex.capture
def load_data(sparse, node_features):
    return build_dataset(data_dir=data_dir, compute_features=node_features, sparse=sparse)


@ex.capture
def log_metrics(metrics, prefix, step, _run):
    for metric, value in metrics.items():
        _run.log_scalar(prefix + metric, value, step)


