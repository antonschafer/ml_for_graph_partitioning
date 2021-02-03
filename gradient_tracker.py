import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.pyplot import figure


# Partially Adapted from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8

class GradientTracker:
    def __init__(self, named_parameters):
        self.named_parameters = named_parameters
        self.layers = []
        self.ave_grads = []
        self.max_grads = []
        for n, p in self.named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                self.layers.append(n)
                self.ave_grads.append(0)
                self.max_grads.append(float("-inf"))
        self.n_tracked = 0

    def _reset(self):
        self.ave_grads = [0 for _ in self.layers]
        self.max_grads = [float("-inf") for _ in self.layers]
        self.n_tracked = 0

    def track(self):
        i = 0
        for n, p in self.named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                assert self.layers[i] == n
                self.ave_grads[i] += p.grad.abs().mean().item()
                self.max_grads[i] = max(self.max_grads[i], p.grad.abs().max().item())
                i += 1
        self.n_tracked += 1

    def _plot(self, fname):
        ave_grads = [x/self.n_tracked for x in self.ave_grads]
        figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.bar(np.arange(len(self.max_grads)), self.max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(self.max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), self.layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

    def plot_and_reset(self, fname):
        self._plot(fname)
        self._reset()

