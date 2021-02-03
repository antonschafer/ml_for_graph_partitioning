import os
import shutil
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from embeddings.solve import compute_cut
from embeddings.experiment import ex, load_data, load_model, set_seed
from embeddings.dataset import collate_function
from baselines import get_baseline_weight, available_baselines


@ex.capture
def compute_weight_imbalance(preds, adj, mask, n, k, max_prob=True):
    """

    :param preds: probabilities (padded), shape (batchsize, nodes, k)
    :param adj: adj matrices (padded), shape (batchsize, nodes, nodes)
    :param mask: shape (batchsize, nodes), 1 for non-padding, 0 for padding
    :param n: shape (batchsize), number of nodes per sample
    :param k: # of partitions
    :param max_prob: whether to use max probability or sample
    :return: cut weight, imbalance
    """

    preds = preds + torch.rand(preds.shape, device=preds.device) * 1e-4  # for random tie breaking
    if max_prob:
        one_hot = preds >= torch.max(preds, dim=2)[0].view(preds.shape[0], -1, 1).repeat(1, 1, k)
        one_hot = one_hot.float()
    else:
        # Not yet implemented. Implementation below just for non-batched data
        # one_hot = torch.zeros_like(preds)
        # one_hot = one_hot[range(preds.shape[0]), torch.multinomial(preds, 1).view(-1)] = 1
        raise Exception("Not yet implemented")
    one_hot *= mask

    # compute cut weights
    B = torch.matmul(adj, 1 - one_hot)
    cut_weight = torch.sum(torch.sum(one_hot * B, dim=1), dim=1) / 2

    # compute imbalances
    partition_weights = torch.sum(one_hot, dim=1)  # shape batchsize x k
    ideal_weight = n / k
    epsilons = (partition_weights / ideal_weight.view(-1, 1)) - 1
    imbalance = torch.max(epsilons, dim=1)[0]  # shape batchsize

    return cut_weight.cpu(), imbalance.cpu()


@ex.capture
def track_stats(preds, inputs, stats, k):
    with torch.no_grad():
        if inputs["node_mask"] is None:
            mask = torch.ones(*inputs["features"].shape[:-1], 1, device=inputs["features"].device)
        else:
            mask = inputs["node_mask"][:, :, [0]]
        cut_weights, imbalances = compute_weight_imbalance(
            preds=preds, adj=inputs["adj"], mask=mask, n=inputs["n"]
        )

    for sample_idx in range(len(inputs["paths"])):

        dir_path = inputs["paths"][sample_idx]["dir_path"]
        graph_idx = int(inputs["paths"][sample_idx]["idx"])
        dir_name = inputs["paths"][sample_idx]["dir_name"]

        for bl_name, col in available_baselines(k).items():
            bl_weight = get_baseline_weight(dir_path, col, graph_idx=graph_idx)
            if bl_weight is not None:
                stats[dir_name][bl_name].append(bl_weight)

        stats[dir_name]["weight_algo"].append(float(cut_weights[sample_idx]))
        stats[dir_name]["imbalance"].append(float(imbalances[sample_idx]))


@ex.capture
def compute_stats(model, dataset, val_batch_size, device):
    model.eval()
    data_loader = DataLoader(
        dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, collate_fn=collate_function
    )
    stats = defaultdict(lambda: defaultdict(list))

    overall_loss, overall_loss_cut, overall_loss_balance = 0, 0, 0

    all_emb_stats = defaultdict(list)
    with torch.no_grad():
        for inputs in tqdm(data_loader):
            cuda_required = ["features", "adj", "adj_gcn", "n", "m"]
            inputs = {k: v if k not in cuda_required else v.to(device) for k, v in inputs.items()}

            preds, (loss, loss_cut, loss_balance), emb_stats = compute_cut(model=model, inputs=inputs)
            track_stats(preds, inputs, stats)
            overall_loss += float(loss)
            overall_loss_cut += float(loss_cut)
            overall_loss_balance += float(loss_balance)
            for k, v in emb_stats.items():
                all_emb_stats[k].append(float(v))

    return (
        stats,
        {
            "loss": overall_loss / len(data_loader),
            "loss cut": overall_loss_cut / len(data_loader),
            "loss balance": overall_loss_balance / len(data_loader),
            **{k: np.mean(v) for k, v in all_emb_stats.items()}
        },
    )


def write_report(stats):
    baselines = ["random", "kl", "greedy", "hmetis"]
    all_factors = {b: [] for b in baselines}
    all_weights = []
    all_imbalances = []

    report = ""
    for graph_type, graph_stats in stats.items():
        weight_algo = graph_stats["weight_algo"]
        all_weights += weight_algo

        imbalance = np.mean(graph_stats["imbalance"])
        all_imbalances.append(imbalance)

        report += "{}:\n".format(graph_type)
        report += "\tMean imbalance: {}".format(imbalance)
        report += "\tMean cut weight: {:.4f}\n".format(np.mean(weight_algo))

        for baseline in baselines:
            weight_baseline = graph_stats["weight_{}".format(baseline)]
            if len(weight_baseline) == 0:
                continue
            elif len(weight_baseline) != len(weight_algo):
                raise Exception("Baseline not available for all inputs")
            factor_baseline = list(np.array(weight_algo) / np.array(weight_baseline))  # slow but does not matter
            all_factors[baseline] += factor_baseline
            report += "\tMean factor to {}: {:.4f}\n".format(baseline, np.mean(factor_baseline))

        report += "\n"

    overall_metrics = {
        "mean imbalance": np.mean(all_imbalances),
        "mean cut weight": np.mean(all_weights),
        **{"mean factor to {}".format(baseline): np.mean(factors) for baseline, factors in all_factors.items()}
    }

    report += "Overall\n"
    for name, val in overall_metrics.items():
        report += " {}: {:.4f}\n".format(name, val)

    return overall_metrics, report


@ex.automain
def main(device, temp_dir, _run):
    set_seed()

    _, _, test_data = load_data()

    model = load_model()
    model = model.to(device)

    # Log test
    test_stats, test_losses = compute_stats(model, test_data)
    metrics, report = write_report(stats=test_stats)
    for m, v in {**metrics, **test_losses}.items():
        _run.log_scalar("test " + m, v, 0)

    fname_report = os.path.join(temp_dir, "test_report")
    with open(fname_report, "w") as f:
        f.write(report)
    _run.add_artifact(fname_report)

    # remove temp files (they are saved by sacred already)
    shutil.rmtree(temp_dir, ignore_errors=True)

    return float(metrics["mean factor to hmetis"])
