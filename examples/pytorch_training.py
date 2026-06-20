#!/usr/bin/env python3
"""
Example pytorch training loop

-   Creates structured toy data: Gaussian blobs with sparse informative features
    (only n_informative of the input_dim features carry signal; the rest are noise)
-   Uses CrossEntropyLoss
-   Uses AdamW optimizer
-   Runs for some number of epochs
-   Prints train and val loss each epoch, using model.train() and model.eval() appropriately

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


def make_dataset_config(output_dim, input_dim, n_informative, cluster_sep=3.0):
    """
    Generate shared dataset config (centers and feature permutation).

    Pass this to make_dataloader for both train and val so they share
    the same underlying structure and feature ordering.

    Centers are placed on a random orthonormal basis scaled by cluster_sep,
    guaranteeing all classes are well-separated regardless of random seed.
    """
    # QR decomposition of a random matrix gives a random orthonormal basis;
    # scaling by cluster_sep guarantees minimum separation between centers.
    basis = torch.linalg.qr(torch.randn(n_informative, output_dim)).Q  # (n_informative, output_dim)
    centers = (basis * cluster_sep).T  # (output_dim, n_informative)
    perm = torch.randperm(input_dim)
    return centers, perm


def make_dataloader(n_samples, input_dim, output_dim, batch_size, centers, perm, n_informative=4):
    """
    Structured toy dataset: Gaussian blobs with sparse informative features.

    Each class is a cluster centered at a point defined by `centers` in an
    n_informative-dimensional subspace. The remaining (input_dim - n_informative)
    features are pure noise. Pass the same centers and perm to train and val
    splits so they share the same underlying structure and feature ordering.
    """
    assert n_informative <= input_dim

    # Sample labels uniformly, then generate features
    labels = torch.randint(0, output_dim, (n_samples,))
    features = torch.randn(n_samples, input_dim)  # all noise to start

    # Overwrite the informative features with cluster signal + noise
    features[:, :n_informative] = centers[labels] + torch.randn(n_samples, n_informative)

    # Permute feature columns so informative features aren't always first
    features = features[:, perm]

    ds = TensorDataset(features, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_one_epoch(device, model, dataloader, loss_fn, optimizer):

    # Set model to training mode (activates dropout, batchnorm, etc.)
    model.train()

    running_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
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


def eval_one_epoch(device, model, dataloader, loss_fn):

    # Set model to eval mode (disables dropout, batchnorm uses running stats, etc.)
    model.eval()

    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, targets)

            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    return epoch_loss


def main():
    torch.manual_seed(42)

    batch_size = 8
    n_informative = 4
    input_dim = 16
    hidden_dim = 24
    output_dim = 4
    p_drop = 0.1
    learning_rate = 0.001
    weight_decay = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    model = ExampleModel(input_dim, hidden_dim, output_dim, p_drop)
    model.to(device)

    centers, perm = make_dataset_config(output_dim, input_dim, n_informative, cluster_sep=3.0)
    train_loader = make_dataloader(n_samples=1024, input_dim=input_dim, output_dim=output_dim, batch_size=batch_size, centers=centers, perm=perm, n_informative=n_informative)
    val_loader   = make_dataloader(n_samples=256,  input_dim=input_dim, output_dim=output_dim, batch_size=batch_size, centers=centers, perm=perm, n_informative=n_informative)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n_epochs = 20
    for i_epoch in range(n_epochs):
        train_loss = train_one_epoch(device, model, train_loader, loss_fn, optimizer)
        val_loss   = eval_one_epoch(device, model, val_loader, loss_fn)
        print(f"Epoch: {i_epoch+1:2d}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
