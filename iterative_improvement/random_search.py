import random

search_space = {
    "dqn_config.r_steps": lambda: random.choice([3, 5, 10]),
    "algo_config.discount_factor": lambda: random.choice([0.7, 0.9, 0.95, 0.99]),
    "steps_per_node": lambda: random.choice([0.3, 0.5]),
    "train_config.lr": lambda: random.choice([0.001, 0.0005]),
    "set_target_steps": lambda: random.choice([2**9, 2**10, 2 ** 11]),
}

cmd_template = "python -m iterative_improvement.train with "

run_ault = "srun -w ault0{} --partition=long --time 12:00:00 {} &"
ault_nodes = [2, 3]
max_per_node = 16


n_runs = 32

if __name__ == "__main__":
    runs = []

    while len(runs) < n_runs:
        options = []
        for param, get_val in search_space.items():
            options.append("'{}={}'".format(param, get_val()))
        cmd = cmd_template + " ".join(options)
        if cmd not in runs:
            runs.append(cmd)

    n_assigned = 0
    node_idx = 0
    for run in runs:
        if n_assigned == max_per_node:
            node_idx += 1
        print(run_ault.format(ault_nodes[node_idx], run))
        n_assigned += 1




