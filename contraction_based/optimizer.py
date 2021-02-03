from memory import ReplayMemory
import torch
from algo import ContractAlgo
from tqdm import tqdm
import torch.nn.functional as F

class Optimizer:
    def __init__(self, model, build_input, build_label, criterion=F.mse_loss, memory_size=8192, model_save_path='model'):
        self.model = model
        self.memory = ReplayMemory(memory_size)
        self.criterion = criterion
        self.build_input = build_input
        self.build_label = build_label
        self.model_save_path = model_save_path

    def train(self, graph_data, learning_rate=0.01, epochs=3, steps_until_update=64, batch_size=256):

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model = self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) # use RMSprop here (used in RL example)
        # writer = SummaryWriter ...

        for epoch in range(epochs):
            i = 0
            for graph, k in graph_data: # how to incorporate k
                algo = ContractAlgo(graph, k=k, calc_q = lambda s,e,k: self.model(self.build_input(s,e,k)))

                self.model.eval()
                steps = 0
                for state, action, reward, best_next_reward in tqdm(algo.step()):
                    self.memory.push(self.build_input(state, action, k), self.build_label(reward, best_next_reward))
                    steps += 1
                    if steps % steps_until_update == 0:
                        loss = self.train_step(optimizer, batch_size)
                        self.model.eval()

                print('Epoch {}, graph {}, Loss: {}'.format(epoch, i, loss))
                i += 1
                torch.save(self.model.state_dict(), self.model_save_path)

    def train_step(self, optimizer, batch_size):
        if len(self.memory) < batch_size:
            return
        self.model.train()
        data = self.memory.get_data()
        for inputs, labels in data:
            optimizer.zero_grad()
            out = self.model(inputs)
            loss = self.criterion(out, labels)
            loss.backward()
            optimizer.step()
        return loss










