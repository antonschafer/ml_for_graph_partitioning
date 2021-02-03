import random
from collections import namedtuple
import torch
from torch import data


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, input, label):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (input, label)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_data(self, batch_size):
        # creating a dataloader every time might be big overhead
        tensor_x = torch.tensor(list(map(lambda x: x[0], self.memory)))
        tensor_y = torch.tensor(list(map(lambda x: x[1], self.memory)))
        dataset = data.TensorDataset(tensor_x, tensor_y)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def __len__(self):
        return len(self.memory)