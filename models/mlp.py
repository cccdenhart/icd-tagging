"""Contains architecture for a multilayer perceptron."""
from functools import reduce
from typing import Callable, List, Sequence

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

# define activation type
ActFn = Callable[[nn.Linear], Tensor]


class Mlp(nn.Module):
    """Initialize and train a multilayer perceptron."""

    def __init__(self, layer_nodes: Sequence[int], act_fns: List[ActFn], n_epochs: int) -> None:
        """Initialize variables."""
        super(Mlp, self).__init__()
        self.act_fns = act_fns
        self.n_epochs = n_epochs
        self.layer_shapes = ((layer_nodes[i - 1], layer_nodes[i])
                             for i in range(1, len(layer_nodes)))
        self.linears = nn.ModuleList([nn.Linear(i, o)
                                      for i, o in self.layer_shapes])

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagate through network."""
        output = x
        for fn, l in zip(self.act_fns, self.linears):
            try:
                output = fn(l(output))
            except TypeError:
                breakpoint()
        return output

    def run_train(self, trainloader: DataLoader) -> None:
        """Train network on training set."""
        self.train()  # set model to train mode
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # loop over the dataset for each epoch
        print("Training mlp .....")
        for i in range(self.n_epochs):
            print(f"Epoch: {i}", end=" ")
            for data in trainloader:
                # extracts features/labels from loader
                feats, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, backward, then optimize
                outputs = self.forward(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print("done.")
        print("done.")
