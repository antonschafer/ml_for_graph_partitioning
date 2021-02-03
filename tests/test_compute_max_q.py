from iterative_improvement.algo.algo_q12 import AlgoQ12
from iterative_improvement.replay_memory import ReplayMemory
from iterative_improvement.train import update_max_q
from torch.utils.data import DataLoader, RandomSampler
from transformers.optimization import AdamW
from iterative_improvement.experiment import load_model, load_data, ex
import numpy as np
import torch


def test_next_max():
    model = load_model()
    n_candidates = 1
    replay_memory = ReplayMemory(size=100)

    train_dataset, val_dataset = load_data()
    train_iter = DataLoader(dataset=train_dataset, batch_size=None, sampler=RandomSampler(train_dataset))
    i = 0
    for graph_info in train_iter:
        if i >= 50:
            break
        i += 1

        G = graph_info["G"]
        adj = graph_info["adj"]
        node_features = graph_info["node_features"]


        algo = AlgoQ12(
            G=G,
            k=2,
            model=model,
            adj=adj,
            node_features=node_features,
            use_progress=False,
            initial_partitioning=None,
            p_exploration=0,
            n_candidates_first=n_candidates,
            device="cpu",
        )


        algo.step()

        # store results of step in replay memory
        step_results = algo.get_state_action_reward()
        step_results["age"] = {"last_computed": 1, "src": True}  # dict to have reference later
        replay_memory.store_example(step_results)


    data_loader = DataLoader(
        replay_memory,
        batch_size=50,
        shuffle=False,
        collate_fn=model.collate_fn,
        num_workers=1,
    )

    optim = AdamW(
        params=model.training_params(),
        lr=0.001,
        weight_decay=1e-5,
    )

    train_config = {
        "steps_per_node": 1,
        "n_candidates_first": 1,  # 1
    }

    dqn_config = {
        "max_age": 0,
        "use_src_pred": False,
    }

    for sample_idx, batch in enumerate(data_loader):
        model.eval()
        optim.zero_grad()

        max_pred_q2 = update_max_q(model=model, batch=batch, global_algo_step=1, train_config=train_config, dqn_config=dqn_config, use_progress=False)

    print(torch.abs(max_pred_q2 - torch.tensor(batch["next_q"])))
    np.testing.assert_almost_equal(max_pred_q2, batch["next_q"])


@ex.automain
def test():
    test_next_max()
    return "Test"
