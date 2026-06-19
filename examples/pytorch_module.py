#!/usr/bin/env python3
"""
Example pytorch Module

Implement a simple feedforward neural network

Write a PyTorch Module for a two-layer feedforward network (MLP) that:
- Takes an input of dimension input_dim
- Has one hidden layer of dimension hidden_dim with ReLU activation
- Produces an output of dimension output_dim
- Has a forward method

Bonus: add dropout to the hidden layer.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, p_drop=0.1):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(p=p_drop)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.layer2(x)
        return x


def main():
    batch_size = 8
    input_dim = 16
    hidden_dim = 32
    output_dim = 4
    p_drop = 0.1

    model = ExampleModel(input_dim, hidden_dim, output_dim, p_drop)

    x = torch.randn(batch_size, input_dim)

    model.eval()
    with torch.no_grad():
        y = model(x)

    print("x = ", x)
    print("y = ", y)


if __name__ == "__main__":
    main()
