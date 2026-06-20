#!/usr/bin/env python3
"""
Example pytorch training loop

-   Creates structured toy data: Gaussian blobs with sparse informative features
    (only n_informative of the input_dim features carry signal; the rest are noise)
-   Uses CrossEntropyLoss
-   Uses AdamW optimizer
-   Runs for some number of epochs
-   Prints the loss each epoch

Bonus: wrap the training in a function, and call model.train() and model.eval() at the right times.

See also:

-   https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
-   https://sebastianraschka.com/faq/docs/training-loop-in-pytorch.html
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ExampleModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, p_drop=0.1):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(p=p_drop)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(p=p_drop)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.layer3(x)
        return x


def train_one_epoch(device, model, dataloader, loss_fn, optimizer):

    # Set model to training mode (activates dropout, batchnorm, etc.)
    model.train() 

    running_loss = 0.0
    total_samples = 0
    
    for i_batch, (features, targets) in enumerate(dataloader):
        # Move data to the active device (GPU or CPU)
        features, targets = features.to(device), targets.to(device)
        
        # Forward pass: Compute predictions by passing data to the model
        outputs = model(features)
        
        # Calculate loss
        loss = loss_fn(outputs, targets)
        
        # Backward pass: Clear old gradients, compute new ones, and update weights
        optimizer.zero_grad()  # Reset gradients from the previous step
        loss.backward()        # Compute gradients via backpropagation
        optimizer.step()       # Update model parameters
        
        # Track statistics
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
    epoch_loss = running_loss / total_samples
    return epoch_loss


def make_dataloader(n_samples, input_dim, output_dim, batch_size, n_informative=4):
    """
    Structured toy dataset: Gaussian blobs with sparse informative features.

    Each class is a cluster centered at a random point in an n_informative-dimensional
    subspace. The remaining (input_dim - n_informative) features are pure noise.
    """
    assert n_informative <= input_dim

    # Random cluster centers only in the informative subspace
    centers = torch.randn(output_dim, n_informative) * 3.0

    # Sample labels uniformly, then generate features
    labels = torch.randint(0, output_dim, (n_samples,))
    features = torch.randn(n_samples, input_dim)  # all noise to start

    # Overwrite the informative features with cluster signal + noise
    features[:, :n_informative] = centers[labels] + torch.randn(n_samples, n_informative)

    # Randomly permute feature columns so informative features aren't always first
    perm = torch.randperm(input_dim)
    features = features[:, perm]

    ds = TensorDataset(features, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def main():
    batch_size = 8
    input_dim = 16
    hidden_dim = 32
    output_dim = 4
    p_drop = 0.1
    learning_rate = 0.005
    weight_decay = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    model = ExampleModel(input_dim, hidden_dim, output_dim, p_drop)
    model.to(device)

    n_samples = 256
    dataloader = make_dataloader(n_samples=n_samples, input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n_epochs = 20
    for i_epoch in range(n_epochs):
        epoch_loss = train_one_epoch(device, model, dataloader, loss_fn, optimizer)
        print(f"Epoch: {i_epoch+1:2d}/{n_epochs}, Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    main()
