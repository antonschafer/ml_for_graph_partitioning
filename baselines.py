import os
import pandas as pd


def get_baseline_weight(graph_dir_path, baseline, graph_idx):
    try:
        df = pd.read_csv(os.path.join(graph_dir_path, "baselines.csv"))
        return df.iloc[graph_idx][baseline]
    except:
        return None


def available_baselines(k):
    return {
        "weight_random": "random_{}".format(k),
        "weight_hmetis": "hmetis_{}_1".format(k),
        **({
               "weight_kl": "kl",
               "weight_greedy": "greedy",
           } if k == 2 else dict())
    }

