import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from utils import flatten, mean_max_std
from collections import defaultdict
from iterative_improvement.algo.algo_q12 import AlgoQ12
from iterative_improvement.algo.n_step_wrapper import NStepWrapper
from iterative_improvement.experiment import ex, set_seed, load_model, load_data, log_metrics
from baselines import get_baseline_weight, available_baselines
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

sns.set()


@ex.capture
def compute_hindsight_truth(*, cut_decreases, targets, dqn_config, algo_config, use_progress):
    r_steps = dqn_config["r_steps"]
    assert use_progress or r_steps == 1
    hindsight_truth = [0 for _ in cut_decreases]
    hindsight_truth[-r_steps:] = targets[-r_steps:]
    for step_idx in reversed(range(len(cut_decreases) - r_steps)):
        cum_reward = sum(cut_decreases[step_idx:step_idx + r_steps])
        hindsight_truth[step_idx] = cum_reward + algo_config["discount_factor"] * hindsight_truth[step_idx + r_steps]

    return hindsight_truth


@ex.capture
def compute_stats(model, dataset, test_config, algo_config, dqn_config, use_progress, device_test, _log: logging.Logger, _run):
    model.eval()

    stats = defaultdict(lambda: defaultdict(list))

    k = algo_config["k"]

    data_loader = DataLoader(dataset, batch_size=None)
    for graph_info in tqdm(data_loader, total=len(data_loader)):

        dir_name = graph_info["paths"]["dir_name"]
        G = graph_info["G"]

        if test_config["initial_partition"] == "random":
            initial_partition = None
        else:
            initial_partition = graph_info[test_config["initial_partition"]]
            assert initial_partition is not None

        algo_1_step = AlgoQ12(
            G=G,
            k=k,
            model=model,
            adj=graph_info["adj"],
            node_features=graph_info["node_features"],
            use_progress=use_progress,
            initial_partitioning=initial_partition,
            p_exploration=0,
            n_candidates_first=test_config["n_candidates_first"],
            max_steps_per_n=test_config["max_steps_per_node"],
            device=device_test,
            run_all_steps=False,
        )
        assert use_progress or dqn_config["r_steps"] == 1, "n-step rewards w/o progress not implemented"
        assert algo_1_step.max_steps >= dqn_config["r_steps"], "cannot use n-step rewards when algo performs less than n steps"

        algo = NStepWrapper(algo=algo_1_step, r_steps=dqn_config["r_steps"])

        preds, targets, cut_decreases, hs_truths = [], [], [], []

        while not algo.done():
            algo.step()
            x, r, cdn, next_q = algo.get_x_r_cdn_nextq()
            y = r + algo_config["discount_factor"] * next_q
            preds.append(x)
            targets.append(y)
            cut_decreases.append(cdn)

        stats[dir_name]["preds"].append(preds)
        stats[dir_name]["targets"].append(targets)
        stats[dir_name]["cut_decreases"].append(cut_decreases)
        stats[dir_name]["hs_truths"].append(compute_hindsight_truth(cut_decreases=cut_decreases, targets=targets))

        weight_algo = algo.best_cut_weight
        steps = algo.steps

        stats[dir_name]["weight_algo"].append(weight_algo)
        stats[dir_name]["steps_by_n"].append(1.0 * steps / len(G))

        dir_path = graph_info["paths"]["dir_path"]
        graph_idx = int(graph_info["paths"]["idx"])

        for bl_name, col in available_baselines(k).items():
            bl_weight = get_baseline_weight(dir_path, col, graph_idx=graph_idx)
            if bl_weight is not None:
                stats[dir_name][bl_name].append(bl_weight)

    return stats


def draw_q_plot(fname, x, hindsight_truth, cut_decrease, preds, targets):
    figure(num=None, figsize=(10, 5), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(x, hindsight_truth, label="Hindsight truth")
    plt.plot(x, cut_decrease, label="Cut decrease")
    plt.plot(x, preds[:, 0], label="Preds Q1")
    plt.plot(x, preds[:, 1], label="Preds Q2")
    plt.plot(x, targets, label="Target")
    plt.xlabel("Step")
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    ex.add_artifact(fname)


@ex.capture
def generate_q_plots(stats, fname_prefix, q_plot_config):
    # TODO make part of algo as might differ for different implementations

    if not any(q_plot_config.values()):
        return

    overall_max_len = 0
    n_runs_total = 0
    for graph_stats in stats.values():
        overall_max_len = max(max(len(t) for t in graph_stats["targets"]), overall_max_len)
        n_runs_total += len(graph_stats["targets"])

    cut_decreases = np.empty((n_runs_total, overall_max_len))
    preds = np.empty((n_runs_total, overall_max_len, 2))
    targets = np.empty((n_runs_total, overall_max_len))
    hindsight_truth = np.empty((n_runs_total, overall_max_len))
    cut_decreases[:] = np.NaN
    preds[:] = np.NaN
    targets[:] = np.NaN
    hindsight_truth[:] = np.NaN

    base_idx = 0
    for graph_type, graph_stats in stats.items():
        n_runs = len(graph_stats["targets"])
        max_len = max(len(t) for t in graph_stats["targets"])

        for i, (cd, p, t, hs) in enumerate(zip(graph_stats["cut_decreases"], graph_stats["preds"], graph_stats["targets"], graph_stats["hs_truths"])):
            run_idx = base_idx + i
            if len(t) == 0:
                # skip if no steps were taken
                continue
            assert len(cd) == len(p) == len(t) == len(hs)

            cut_decreases[run_idx, :len(cd)] = cd
            preds[run_idx, :len(p), :] = p
            targets[run_idx, :len(t)] = t
            hindsight_truth[run_idx, :len(hs)] = hs

        if q_plot_config["graph_wise_single"]:
            # plot single run
            fname_single = fname_prefix + graph_type + "--1.png"
            len_single = len(graph_stats["targets"][0])
            draw_q_plot(fname=fname_single,
                        x=list(range(len_single)),
                        hindsight_truth=hindsight_truth[base_idx][:len_single],
                        cut_decrease=cut_decreases[base_idx][:len_single],
                        preds=preds[base_idx][:len_single],
                        targets=targets[base_idx][:len_single])

        if q_plot_config["graph_wise_mean"]:
            last_idx = base_idx + n_runs
            # plot mean
            fname_mean_graph = fname_prefix + graph_type + "--mean.png"
            draw_q_plot(fname=fname_mean_graph,
                        x=list(range(max_len)),
                        hindsight_truth=np.nanmean(hindsight_truth[base_idx:last_idx][:max_len], axis=0),
                        cut_decrease=np.nanmean(cut_decreases[base_idx:last_idx][:max_len], axis=0),
                        preds=np.nanmean(preds[base_idx:last_idx][:max_len], axis=0),
                        targets=np.nanmean(targets[base_idx:last_idx][:max_len], axis=0))

        base_idx += n_runs

    if q_plot_config["all_single"]:
        # plot single run
        fname_single = fname_prefix + "--1.png"
        non_nan_indices = [x for x in range(overall_max_len) if not np.isnan(cut_decreases[0][x])]
        len_single = len(non_nan_indices)
        draw_q_plot(fname=fname_single,
                    x=non_nan_indices,
                    hindsight_truth=hindsight_truth[0][:len_single],
                    cut_decrease=cut_decreases[0][:len_single],
                    preds=preds[0][:len_single],
                    targets=targets[0][:len_single])

    if q_plot_config["all_mean"]:
        # plot mean
        fname_mean = fname_prefix + "--overall--mean.png"
        draw_q_plot(fname=fname_mean,
                    x=list(range(overall_max_len)),
                    hindsight_truth=np.nanmean(hindsight_truth, axis=0),
                    cut_decrease=np.nanmean(cut_decreases, axis=0),
                    preds=np.nanmean(preds, axis=0),
                    targets=np.nanmean(targets, axis=0))


def write_report(stats):
    all_factors_random = []
    all_factors_kl = []
    all_factors_greedy = []
    all_factors_hmetis = []
    all_weights = []
    all_steps = []

    report = ""
    for graph_type, graph_stats in stats.items():
        weight_algo = graph_stats["weight_algo"]
        weight_random = graph_stats["weight_random"]
        weight_kl = graph_stats["weight_kl"]
        weight_hmetis = graph_stats["weight_hmetis"]
        weight_greedy = graph_stats["weight_greedy"]
        steps = graph_stats["steps_by_n"]
        cut_decreases = flatten(graph_stats["cut_decreases"])
        preds = torch.tensor(flatten(graph_stats["preds"]))
        targets = torch.tensor(flatten(graph_stats["targets"]))

        EMPTY = torch.tensor([], dtype=torch.float32)
        preds_q1, preds_q2 = (preds[:, 0], preds[:, 1]) if len(preds) > 0 else (EMPTY, EMPTY)

        array_weight_algo = np.array(weight_algo)
        factor_random = list(array_weight_algo / np.array(weight_random))  # slow but does not matter
        if len(weight_kl) == len(weight_algo):
            factor_kl = list(array_weight_algo / np.array(weight_kl))
        else:
            factor_kl = []
        if len(weight_greedy) == len(weight_algo):
            factor_greedy = list(array_weight_algo / np.array(weight_greedy))
        else:
            factor_greedy = []
        if len(weight_hmetis) == len(weight_algo):
            factor_hmetis = list(array_weight_algo / np.array(weight_hmetis))
        else:
            factor_hmetis = []

        all_weights += weight_algo
        all_steps += steps
        all_factors_kl += factor_kl
        all_factors_greedy += factor_greedy
        all_factors_hmetis += factor_hmetis
        all_factors_random += factor_random

        report += "{}:\n".format(graph_type)
        report += "\tMean number of steps by n: {:.4f}\n".format(np.mean(steps))
        report += "\tMean cut weight: {:.4f}\n".format(np.mean(weight_algo))
        report += "\tMean factor to hmetis: {:.4f}\n".format(np.mean(factor_hmetis))
        report += "\tMean factor to K/L: {:.4f}\n".format(np.mean(factor_kl))
        report += "\tMean factor to greedy: {:.4f}\n".format(np.mean(factor_greedy))
        report += "\tMean factor to random: {:.4f}\n".format(np.mean(factor_random))
        report += "\n"
        report += "\tmean Q: {:.4f}\n".format(torch.mean(targets).item())
        report += "\tmean absolute Q: {:.4f}\n".format(torch.mean(torch.abs(targets)).item())
        report += "\tstd Q: {:.4f}\n".format(torch.std(targets).item())
        report += "\tstd pred Q1: {:.4f}\n".format(torch.std(preds_q1).item())
        report += "\tstd pred Q2: {:.4f}\n".format(torch.std(preds_q2).item())
        report += "\n"
        cut_decreases = np.array(cut_decreases)
        report += "\tMean total cut decrease: {:.4f}\n".format(np.sum(cut_decreases) / len(weight_algo))
        report += "\tMean # cut decreases: {:.4f}\n".format(np.sum(cut_decreases > 0) / len(weight_algo))
        report += "\tMean # cut increases: {:.4f}\n".format(np.sum(cut_decreases < 0) / len(weight_algo))
        report += "\n\n\n"

    report += "Overall\n"
    report += "Mean number of steps by n: {:.4f}\n".format(np.mean(all_steps))
    report += "Mean cut weight: {:.4f}\n".format(np.mean(all_weights))
    report += "Mean factor to hMETIS: {:.4f}\n".format(np.mean(all_factors_hmetis))
    report += "Mean factor to K/L: {:.4f}\n".format(np.mean(all_factors_kl))
    report += "Mean factor to greedy: {:.4f}\n".format(np.mean(all_factors_greedy))
    report += "Mean factor to random: {:.4f}".format(np.mean(all_factors_random))

    return report


@ex.capture
def validation(*, model, dataset, _run, fname_text_report=None, fname_prefix_qplot=None):
    stats = compute_stats(model=model, dataset=dataset)

    # combine stats from all graph types
    preds, targets, cut_decreases, hs_truths = [], [], [], []
    weights_algo, weights_random, weights_kl, weights_greedy, weights_hmetis, steps_by_n = [], [], [], [], [], []
    for k, vals in stats.items():
        weights_algo += vals["weight_algo"]
        weights_kl += vals["weight_kl"]
        weights_greedy += vals["weight_greedy"]
        weights_random += vals["weight_random"]
        weights_hmetis += vals["weight_hmetis"]
        steps_by_n += vals["steps_by_n"]

        preds += flatten(vals["preds"])
        hs_truths += flatten(vals["hs_truths"])
        targets += flatten(vals["targets"])
        cut_decreases += flatten(vals["cut_decreases"])

    array_weights_algo = np.array(weights_algo)
    factors_random = array_weights_algo / np.array(weights_random)

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    hs_truths = torch.tensor(hs_truths)
    EMPTY = torch.tensor([], dtype=torch.float32)
    preds_q1, preds_q2 = (preds[:, 0], preds[:, 1]) if len(preds) > 0 else (EMPTY, EMPTY)  # edge case that no steps taken

    metrics = {
        "steps per node": np.mean(steps_by_n),
        "mean factor to random": np.mean(factors_random),
        "loss Q1": F.mse_loss(preds_q1, targets).item() if len(preds) > 0 else 0,
        "loss Q2": F.mse_loss(preds_q2, targets).item() if len(preds) > 0 else 0,
        "mean hindsight truth": torch.mean(hs_truths).item(),
        "mean Q": torch.mean(targets).item(),
        "mean absolute Q": torch.mean(torch.abs(targets)).item(),
        "std Q": torch.std(targets).item(),
        "std pred Q1": torch.std(preds_q1).item(),
        "std pred Q2": torch.std(preds_q2).item(),
        "mean pred Q1": torch.mean(preds_q1).item(),
        "mean pred Q2": torch.mean(preds_q2).item(),
    }

    if len(weights_hmetis) == len(weights_algo):  # report none if not all available
        metrics = {**metrics, **mean_max_std(array_weights_algo / np.array(weights_hmetis), "factor to hMETIS")}

    if len(weights_kl) == len(weights_algo):  # report none if not all available
        metrics = {
            **metrics,
            "mean factor to K/L": np.mean(array_weights_algo / np.array(weights_kl)),
        }

    if len(weights_greedy) == len(weights_algo):  # report none if not all available
        metrics = {
            **metrics,
            "mean factor to greedy": np.mean(array_weights_algo / np.array(weights_greedy)),
        }

    if len(cut_decreases) > 0:
        metrics = {**metrics, **mean_max_std(cut_decreases, "cut decrease")}

    # save graph-wise report in text form
    if fname_text_report is not None:
        text_report = write_report(stats=stats)
        with open(fname_text_report, "w") as f:
            f.write(text_report)
        _run.add_artifact(fname_text_report)

    # save q plots
    if fname_prefix_qplot is not None:
        generate_q_plots(stats=stats, fname_prefix=fname_prefix_qplot)

    return metrics


@ex.automain
def run(device_algo, temp_dir):
    set_seed()
    torch.set_default_tensor_type(torch.FloatTensor)

    model = load_model()
    model = model.to(device_algo)
    print("Number of trainable parameters:", model.num_params())

    _, _, test_dataset = load_data()

    test_metrics = validation(
        model=model,
        dataset=test_dataset,
        fname_text_report=os.path.join(temp_dir, "test_report_step_{}.txt".format(0)),
        fname_prefix_qplot=os.path.join(temp_dir, "test_qplot_step_{}--".format(0))
    )
    log_metrics(
        test_metrics,
        prefix="test: ",
        step=0,
    )

    # remove temp files (they are saved by sacred already)
    shutil.rmtree(temp_dir, ignore_errors=True)

    test_result = float(test_metrics["mean factor to hMETIS"])
    return test_result
