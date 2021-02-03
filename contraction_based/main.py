from optimizer import Optimizer
from model import DQN
import numpy as np

EMBEDDING_SIZE = 32
DISCOUNT = 1


def build_input(state, edge, k):
    k = np.array([k])
    return np.concatenate((state, edge[0], edge[1], k))


def build_label(reward, best_next_reward):
    return reward + DISCOUNT*best_next_reward


dqn = DQN(EMBEDDING_SIZE * 3 + 1)
optimizer = Optimizer(dqn, build_input, build_label, learning_rate=0.01)
# consider huber loss

optimizer.train(graph_data)
