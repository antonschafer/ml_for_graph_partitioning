from torch.utils.data import Dataset
import random


class ReplayMemory(Dataset):
    def __init__(self, size):
        super(ReplayMemory, self).__init__()

        self.size = size
        self.data = []

        self.static_trained = True
        self.dynamic_trained = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def store_example(self, x):
        self.data.append(x)
        if len(self.data) > self.size:
            self.data.pop(0)

    def sample(self, k):
        return random.sample(self.data, k)
