import torch
from embeddings.experiment import ex
from embeddings.losses import balance_loss, cut_weight_loss, gap_cw_loss


@ex.capture
def compute_cut(model, inputs, n_preds, k, loss_weight):

    model_required_inputs = ["features", "feature_mask", "adj_gcn"]
    model_inputs = {k: v for k, v in inputs.items() if k in model_required_inputs}

    logits, emb_stats = model(**model_inputs)

    min_loss = None
    best_preds = None
    for i in range(n_preds):
        preds = logits[:, :, k * i : k * (i + 1)]

        preds = torch.softmax(preds, dim=2)

        # take balance loss before balancing correction
        loss_balance = balance_loss(preds, n=inputs["n"])

        # # make perfectly balanced
        # partition_weights = torch.sum(preds * inputs["feature_mask"][:, :, [0]], dim=1, keepdim=True)
        # diff_ideal = 1/k - partition_weights / inputs["n"].view(-1, 1, 1)
        # preds = preds + diff_ideal

        loss_cut = cut_weight_loss(preds, adj=inputs["adj"], m=inputs["m"])
        # loss_cut = gap_cw_loss(preds, adj=inputs["adj"])
        loss = loss_weight["cut"] * loss_cut + loss_weight["balance"] * loss_balance

        if min_loss is None or min_loss[0] > loss:
            min_loss = loss, loss_cut, loss_balance
            best_preds = preds

    return best_preds, min_loss, emb_stats
