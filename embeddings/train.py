import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from embeddings.test import track_stats, write_report, compute_stats
from embeddings.experiment import ex, set_seed, load_data, load_model
from embeddings.solve import compute_cut
from collections import defaultdict
from embeddings.dataset import collate_function
from torch.optim.lr_scheduler import StepLR
from gradient_tracker import GradientTracker
import os
import shutil


@ex.capture
def log_metrics(metrics, prefix, step, _run):
    for metric, value in metrics.items():
        _run.log_scalar(prefix + metric, value, step)


@ex.capture
def train(
    model,
    dataset,
    val_dataset,
    test_dataset,
    n_epochs,
    lr,
    batch_size,
    weight_decay,
    accumulation_steps,
    log_steps,
    device,
    lr_schedule_params,
    _run,
    temp_dir,
    early_stopping
):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_function, drop_last=True)

    optim = AdamW(params=model.training_params(), lr=lr, weight_decay=weight_decay,)
    optim.zero_grad()
    scheduler = StepLR(optim, **lr_schedule_params)

    global_step = 0
    stats = defaultdict(lambda: defaultdict(list))

    grad_tracker = GradientTracker(model.named_training_params())

    losses, losses_cut, losses_balance = [], [], []
    val_loss_history = []
    last_results = []

    for epoch in range(n_epochs):
        print("Starting epoch {}".format(epoch))

        for inputs in tqdm(data_loader):

            model.train()

            cuda_required = ["features", "adj", "adj_gcn", "adj_mask", "n", "m"]
            inputs = {k: v if k not in cuda_required else v.to(device) for k, v in inputs.items()}

            preds, (loss, loss_cut, loss_balance), _ = compute_cut(model=model, inputs=inputs)

            loss.backward()

            grad_tracker.track()

            losses.append(float(loss))
            losses_cut.append(float(loss_cut))
            losses_balance.append(float(loss_balance))

            track_stats(preds=preds, inputs=inputs, stats=stats)

            global_step += len(inputs["paths"])

            # Inputs no longer needed.
            # Free GPU memory as might be needed by validation computations where inputs still in scope
            del inputs
            torch.cuda.empty_cache()

            if global_step % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()

            if global_step % log_steps == 0:

                _run.log_scalar("lr", scheduler.get_lr(), global_step)

                def log_results(split_stats, split_losses, split):
                    metrics, report = write_report(stats=split_stats)
                    log_metrics({**metrics, **split_losses}, "{} ".format(split), global_step)
                    fname_report = os.path.join(temp_dir, "{}_report_{}".format(split, global_step))
                    with open(fname_report, "w") as f:
                        f.write(report)
                    _run.add_artifact(fname_report)
                    return metrics["mean factor to hmetis"]

                # Log training
                _run.log_scalar("train loss", np.mean(losses), global_step)
                _run.log_scalar("train loss cut", np.mean(losses_cut), global_step)
                _run.log_scalar("train loss balance", np.mean(losses_balance), global_step)
                log_results(stats, {}, "train")

                # Reset metrics
                losses, losses_cut, losses_balance = [], [], []
                stats = defaultdict(lambda: defaultdict(list))

                # Log validation
                val_stats, val_losses = compute_stats(model, val_dataset)
                val_loss_history.append(val_losses["loss"])
                log_results(val_stats, val_losses, "val")

                # Log test
                test_stats, test_losses = compute_stats(model, test_dataset)
                current_res = log_results(test_stats, test_losses, "test")
                last_results.append(current_res)

                # Log gradients
                grad_fname = os.path.join(temp_dir, "grad_plot_{}.png".format(global_step))
                grad_tracker.plot_and_reset(grad_fname)
                _run.add_artifact(grad_fname)

                if len(val_loss_history) > early_stopping:
                    if val_loss_history[0] == min(val_loss_history):
                        model.save()
                        for fname in model.save_fnames:
                            _run.add_artifact(fname)
                        print("Early stopping")
                        return float(last_results[0])
                    else:
                        val_loss_history.pop(0)
                        last_results.pop(0)

        scheduler.step()

    model.save()
    for fname in model.save_fnames:
        _run.add_artifact(fname)
    print("Done training")
    return float(current_res)



@ex.automain
def main(device, temp_dir):
    set_seed()

    train_data, val_data, test_data = load_data()

    model = load_model()
    model = model.to(device)
    print("Number of trainable parameters:", model.num_params())

    torch.autograd.set_detect_anomaly(True)

    res = train(model=model, dataset=train_data, val_dataset=val_data, test_dataset=test_data)

    # remove temp files (they are saved by sacred already)
    shutil.rmtree(temp_dir, ignore_errors=True)

    return res
