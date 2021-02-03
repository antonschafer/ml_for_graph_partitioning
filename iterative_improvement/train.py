import os
import time
import copy
import torch
import numpy as np
import logging
import shutil
from collections import defaultdict

from torch.optim.lr_scheduler import StepLR

from timer import Timer
from torch.utils.data import DataLoader, RandomSampler
from transformers.optimization import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from iterative_improvement.experiment import ex, set_seed, load_model, load_data, log_metrics
from iterative_improvement.replay_memory import ReplayMemory
from iterative_improvement.algo.algo_q12 import AlgoQ12
from iterative_improvement.algo.n_step_wrapper import NStepWrapper
from iterative_improvement.test import generate_q_plots, validation, compute_hindsight_truth
from baselines import get_baseline_weight, available_baselines
from utils import load_pickle, cut_weight_mapping, random_partition, mean_max_std, dict_to_device, arg_n_largest, flatten


@ex.capture
def scheduler_step(schedulers):
    for s in schedulers.values():
        s.step()


@ex.capture
def train(
    model,
    target_model,
    train_dataset,
    val_dataset,
    test_dataset,
    train_config,
    dqn_config,
    algo_config,
    use_progress,
    schedulers,
    _log: logging.Logger,
    _run,
    device_algo,
    device_train,
    device_target,
    temp_dir,
    time_lim,
    lr_schedule
):
    if time_lim is None:
        time_lim = float("inf")
    timestamp_stop = time.time() + time_lim * 60 * 60  # convert to sec
    timestamps_log = [time.time()]

    timer = Timer()
    replay_memory = ReplayMemory(size=dqn_config["replay_mem_size"])

    global_algo_step = 0
    cut_weights = dict()
    losses_q1, losses_q2, losses = [], [], []
    log_preds_q1, log_preds_q2, log_vals_q1, log_vals_q2 = [], [], [], []
    train_stats = defaultdict(lambda: defaultdict(list))

    best_val_result = float("inf")
    best_step = 0

    discount_factor = algo_config["discount_factor"]

    optim = AdamW(
        params=model.training_params(),
        lr=train_config["lr"],  # TODO lr schedule
        weight_decay=train_config["weight_decay"],
    )

    lr_scheduler = StepLR(optim, **lr_schedule)

    for epoch in range(train_config["epochs"]):
        print("Starting epoch {}".format(epoch))

        train_iter = DataLoader(dataset=train_dataset, batch_size=None, sampler=RandomSampler(train_dataset))
        for graph_info in tqdm(train_iter, total=len(train_iter)):

            if train_config["initial_partition"] == "random":
                initial_partition = None
            else:
                initial_partition = graph_info[train_config["initial_partition"]]
                assert initial_partition is not None

            algo_1_step = AlgoQ12(
                G=graph_info["G"],
                k=algo_config["k"],
                model=model,
                adj=graph_info["adj"],
                node_features=graph_info["node_features"],
                use_progress=use_progress,
                initial_partitioning=initial_partition,
                max_steps_per_n=train_config["steps_per_node"],
                p_exploration=train_config["p_exploration"](schedulers),
                n_candidates_first=train_config["n_candidates_first"],
                device=device_algo,
                run_all_steps=True,
            )
            assert use_progress or dqn_config["r_steps"] == 1, "n-step rewards w/o progress not implemented"
            assert algo_1_step.max_steps >= dqn_config["r_steps"], "cannot use n-step rewards when algo performs less than n steps"

            algo = NStepWrapper(algo=algo_1_step, r_steps=dqn_config["r_steps"])

            file_info = (graph_info["paths"]["dir_path"], graph_info["paths"]["idx"])

            preds, targets, cut_decreases = [], [], []

            while not algo.done():
                # continue collecting experience
                model = model.to(device_algo)
                timer.enter_op("algo")
                algo.step()
                timer.enter_op("other")

                # log results of step TODO
                x, r, cdn, next_q = algo.get_x_r_cdn_nextq()
                y = r + algo_config["discount_factor"] * next_q
                preds.append(x)
                targets.append(y)
                cut_decreases.append(cdn)

                step_res = algo.get_state_action_reward()
                step_res["age"] = {"last_computed": global_algo_step, "src": True}  # dict to have reference later
                # store
                replay_memory.store_example(step_res)
                global_algo_step += 1
                lr_scheduler.step()

                scheduler_step()

                if (
                    global_algo_step % dqn_config["train_interval"] == 0
                    and len(replay_memory) >= dqn_config["min_replay_size"]
                ):

                    data_loader = DataLoader(
                        replay_memory,
                        batch_size=train_config["batch_size"],
                        shuffle=True,
                        collate_fn=model.collate_fn,
                        num_workers=train_config["n_workers_dl"],
                    )

                    for sample_idx, batch in enumerate(data_loader):
                        if sample_idx >= dqn_config["n_batches"]:
                            break

                        model.train()
                        optim.zero_grad()

                        if device_train == device_target:
                            # save time by only copying to cuda once
                            batch["static_in"] = dict_to_device(batch["static_in"], device_train)

                        timer.enter_op("compute_preds")
                        pred_q1, pred_q2, mask_q1 = compute_preds(model=model, batch=batch)
                        timer.enter_op("other")

                        timer.enter_op("compute_targets")
                        max_pred_q2 = update_max_q(model=target_model, batch=batch, global_algo_step=global_algo_step)
                        timer.enter_op("other")

                        labels = torch.tensor(
                            batch["reward"], dtype=torch.float32, device=device_train
                        ) + discount_factor * max_pred_q2.to(device_train)

                        labels = labels[[int(i / 2) for i in range(2 * len(labels))]]  # repeat every entry

                        # TODO weigh q1 the same?
                        loss_q1 = F.mse_loss(pred_q1 * mask_q1, labels * mask_q1, reduction="sum") / len(labels)
                        loss_q2 = F.mse_loss(pred_q2, labels, reduction="mean")

                        loss = train_config["weight_loss_q1"] * loss_q1 + train_config["weight_loss_q2"] * loss_q2

                        timer.enter_op("backward_pass")
                        # optimizer step
                        loss.backward()
                        optim.step()
                        timer.enter_op("other")

                        # save values for logging
                        losses.append(float(loss))
                        losses_q2.append(float(loss_q2))
                        n_samples_q1 = float(torch.sum(mask_q1))
                        if n_samples_q1 > 0:
                            losses_q1.append(float(loss_q1) * len(labels) / n_samples_q1)
                        log_preds_q1 += [float(p) for p, m in zip(pred_q1, mask_q1) if m == 1]
                        log_preds_q2 += [float(p) for p in pred_q2]
                        float_labels = [float(l) for l in labels]
                        log_vals_q1 += [l for l, m in zip(float_labels, mask_q1) if m == 1]
                        log_vals_q2 += float_labels

                    # set static rep to dirty if necessary
                    if (
                        algo.steps - dqn_config["max_static_age"] * dqn_config["train_interval"]
                        > algo.static_rep_timestamp
                    ):
                        algo.set_static_rep_dirty()

                if global_algo_step % dqn_config["log_steps"] == 0:
                    plot_q = global_algo_step % (dqn_config["freq_qplot"] * dqn_config["log_steps"]) == 0
                    timer.enter_op("validation")
                    val_metrics = validation(
                            model=model,
                            dataset=val_dataset,
                            fname_text_report=os.path.join(temp_dir, "val_report_step_{}.txt".format(global_algo_step))
                            if train_config["save_report"]
                            else None,
                            fname_prefix_qplot=os.path.join(temp_dir, "val_qplot_step_{}--".format(global_algo_step))
                            if plot_q else None
                        )
                    log_metrics(
                        val_metrics,
                        prefix="validation: ",
                        step=global_algo_step,
                    )
                    timer.enter_op("other")
                    if plot_q:
                        generate_q_plots(stats=train_stats, fname_prefix=os.path.join(temp_dir, "train_qplot_step_{}--".format(global_algo_step)))
                    log_training(cut_weights, timer, global_algo_step)
                    timer.reset()
                    train_stats = defaultdict(lambda: defaultdict(list))
                    cut_weights = dict()

                    _run.log_scalar("lr", lr_scheduler.get_lr(), global_algo_step)
                    _run.log_scalar("p_exploration", train_config["p_exploration"](schedulers), global_algo_step)
                    _run.log_scalar("training: loss", np.mean(losses), global_algo_step)
                    _run.log_scalar("training: loss q1", np.mean(losses_q1), global_algo_step)
                    _run.log_scalar("training: loss q2", np.mean(losses_q2), global_algo_step)
                    _run.log_scalar("training: mean pred q1", np.mean(log_preds_q1), global_algo_step)
                    _run.log_scalar("training: mean pred q2", np.mean(log_preds_q2), global_algo_step)
                    _run.log_scalar("training: mean q1", np.mean(log_vals_q1), global_algo_step)
                    _run.log_scalar("training: mean q2", np.mean(log_vals_q2), global_algo_step)
                    losses, losses_q1, losses_q2 = [], [], []
                    log_preds_q1, log_preds_q2, log_vals_q1, log_vals_q2 = [], [], [], []

                    val_result = float(val_metrics["mean factor to hMETIS"])

                    if val_result < best_val_result:
                        best_val_result = val_result
                        best_step = global_algo_step

                        # save model after every improvement
                        model.save()

                    # respect timelimit
                    timestamps_log.append(time.time())
                    mean_duration = np.mean([timestamps_log[i] - timestamps_log[i-1] for i in range(1, len(timestamps_log))])
                    force_terminate = (timestamp_stop - time.time()) < 4 * mean_duration

                    # early stopping
                    if global_algo_step - best_step >= dqn_config["early_stopping"] * dqn_config["log_steps"] or force_terminate:
                            print("Early stopping")
                            return terminate(model=model, best_step=best_step, test_dataset=test_dataset)

                if global_algo_step % dqn_config["set_target_steps"] == 0:
                    target_model = copy.deepcopy(model).to(device_target)

            # save cut weight after algorithm termination
            cut_weights[file_info] = algo.best_cut_weight

            # save step information
            dir_name = graph_info["paths"]["dir_name"]
            train_stats[dir_name]["preds"].append(preds)
            train_stats[dir_name]["targets"].append(targets)
            train_stats[dir_name]["cut_decreases"].append(cut_decreases)
            train_stats[dir_name]["hs_truths"].append(compute_hindsight_truth(cut_decreases=cut_decreases, targets=targets))

    print("Done training")
    # final report
    return terminate(model=model, best_step=best_step, test_dataset=test_dataset)


@ex.capture
def terminate(model, best_step, test_dataset, temp_dir, _run):
    # save weights with sacred
    for fname in model.save_fnames:
        _run.add_artifact(fname)

    model.load(fnames=model.save_fnames)  # Restore best weights

    # run test at end
    test_metrics = validation(
        model=model,
        dataset=test_dataset,
        fname_text_report=os.path.join(temp_dir,
                                       "test_report_step_{}.txt".format(best_step)),
        fname_prefix_qplot=os.path.join(temp_dir,
                                        "test_qplot_step_{}--".format(best_step)),
    )
    log_metrics(
        test_metrics,
        prefix="test: ",
        step=best_step,
    )
    return float(test_metrics["mean factor to hMETIS"])


@ex.capture
def compute_preds(model, batch, device_train):
    model = model.to(device_train)
    batch["dynamic_in"] = {"node_2_partition": batch["node_2_partition"], "additional_in": batch["additional_in"]}
    batch["q1_in"] = {"nodes": [list(nodes) for nodes in batch["swap"]]}
    batch["q2_in"] = {
        "nodes_1": [list(nodes) for nodes in batch["swap"]],
        "nodes_2": [[[n2], [n1]] for n1, n2 in batch["swap"]],
    }

    pred_q1, pred_q2 = model.forward(batch=batch, device=device_train)

    mask_q1 = torch.ones(pred_q1.shape)
    for i, (explore_1, explore_2) in enumerate(batch["explore"]):
        if explore_1:
            mask_q1[2 * i + 1] = 0
            # TODO always set this to 0? as n2 here never satisfies assumption that it was chosen as best node for n1
        if explore_2:
            mask_q1[2 * i] = 0
            mask_q1[2 * i + 1] = 0

    return pred_q1, pred_q2, mask_q1


@ex.capture
def update_max_q(model, batch, global_algo_step, device_target, train_config, dqn_config, use_progress):
    """

    :param model:
    :param batch:
    :param global_algo_step:
    :param device_target:
    :param train_config:
    :param dqn_config:
    :return:

    effects: updates timestamps of recomputed q values in batch
    """
    indices_dirty = [
        i
        for i, (age, is_last) in enumerate(zip(batch["age"], batch["last_step"]))
        if (
                   (age["src"] and not dqn_config["use_src_pred"])
                   or global_algo_step - age["last_computed"] > dqn_config["max_age"]
           )
           and not (use_progress and is_last)
    ]

    recompute = [False for _ in batch["next_q"]]
    for i in indices_dirty:
        recompute[i] = True

    if len(indices_dirty) == 0:
        # return precomputed values if recent enough
        return torch.tensor(batch["next_q"], dtype=torch.float32)

    # build batch of only dirty values
    def select_dirty(x):
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            assert x.shape[0] == len(recompute)
            return x[indices_dirty]
        elif type(x) == list:
            assert len(x) == len(recompute)
            return [elt for elt, rc in zip(x, recompute) if rc]
        elif type(x) == dict:
            return {k: select_dirty(v) for k, v in x.items()}
        else:
            return x  # for single ints like k

    batch_dirty = select_dirty(batch)

    # construct result as combination of precomputed and newly computed values
    max_q2 = torch.tensor(batch["next_q"], dtype=torch.float32)
    max_q2_new = compute_max_q(model=model, batch=batch_dirty, device_target=device_target, train_config=train_config)
    for i, new_val in zip(indices_dirty, max_q2_new):
        max_q2[i] = new_val

    if use_progress:
        # overwrite last steps with future reward 0
        for i, is_last in enumerate(batch["last_step"]):
            if is_last:
                max_q2[i] = 0

    # update timestamp of recomputed values
    for i in indices_dirty:
        batch["age"][i]["last_computed"] = global_algo_step

    return max_q2


@ex.capture
def compute_max_q(model, batch, device_target, train_config):
    # compute next max q values
    with torch.no_grad():
        static_rep = model.static_representation(**dict_to_device(batch["static_in"], device_target))
        # TODO increment step in progress
        dynamic_rep = model.dynamic_representation(
            node_2_partition=batch["next_n2p"], additional_in=batch["next_add_in"], static_rep=static_rep
        )

        q1_vals_all = model.q1(dynamic_rep=dynamic_rep, nodes=[list(G) for G in batch["G"]], separate_samples=True)
        candidate_nodes_1_all = []
        for q1_vals in q1_vals_all:
            candidate_nodes_1_all.append(arg_n_largest(train_config["n_candidates_first"], q1_vals.cpu()))
        candidate_nodes_2_all = []
        cum_counter = 0
        num_candidates_cum = [0]
        for candidate_nodes_1, G, node_2_partition in zip(
            candidate_nodes_1_all, batch["G"], batch["next_n2p"]
        ):
            candidate_nodes_2 = [
                [n2 for n2 in G if node_2_partition[n2] != node_2_partition[n1]] for n1 in candidate_nodes_1
            ]
            candidate_nodes_2_all.append(candidate_nodes_2)
            cum_counter += sum(len(subl) for subl in candidate_nodes_2)
            num_candidates_cum.append(cum_counter)

        q2_vals_all = model.q2(
            dynamic_rep=dynamic_rep,
            nodes_1=candidate_nodes_1_all,
            nodes_2=candidate_nodes_2_all,
            separate_samples=False,
        )
        max_q2 = [
            max(q2_vals_all[num_candidates_cum[i] : num_candidates_cum[i + 1]])
            for i in range(len(candidate_nodes_2_all))
        ]

        return max_q2


@ex.capture
def log_training(cut_weights, timer, global_step, algo_config, _log: logging.Logger, _run):

    # factors
    factors_to_bl = defaultdict(list)
    for (dir_path, graph_idx), weight in cut_weights.items():
        for bl_name, col in available_baselines(algo_config["k"]).items():
            bl_weight = get_baseline_weight(dir_path, col, graph_idx=graph_idx)
            if bl_weight is not None:
                factors_to_bl[bl_name].append(weight / bl_weight)

    for bl, factors in factors_to_bl.items():
        if len(factors) == len(cut_weights) and len(cut_weights) > 0:  # only log if available for all graphs
            _run.log_scalar("training: mean factor to {}".format(bl), np.mean(factors), global_step)
            _run.log_scalar("training: min factor to {}".format(bl), min(factors), global_step)
            _run.log_scalar("training: max factor to {}".format(bl), max(factors), global_step)

    # times
    timer.done()
    for op, time in timer.times.items():
        _run.log_scalar("time {}".format(op), time, global_step)
    timer.reset()


@ex.automain
def run(device_algo, device_target, temp_dir):
    set_seed()
    torch.set_default_tensor_type(torch.FloatTensor)

    model, target_model = load_model(), load_model()
    model = model.to(device_algo)
    target_model = target_model.to(device_target)
    print("Number of trainable parameters:", model.num_params())

    train_dataset, val_dataset, test_dataset = load_data()
    res = train(model=model, target_model=target_model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

    # remove temp files (they are saved by sacred already)
    shutil.rmtree(temp_dir, ignore_errors=True)

    return res
