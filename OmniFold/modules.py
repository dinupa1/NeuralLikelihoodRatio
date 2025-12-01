import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os


class MLPClassifier(nn.Module):
    def __init__(self, x_dim:int = 1, hidden_dim:int = 64, n_classes:int = 1):
        super(MLPClassifier, self).__init__()

        self.log_likelihood_ratio = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        log_r = self.log_likelihood_ratio(x)
        logit = self.sigmoid(log_r)
        return logit


class MLPLoss(nn.Module):
    def __init__(self):
        super(MLPLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, logit, target, weight):
        loss = self.bce_loss(logit, target)
        return torch.sum(weight * loss) / torch.sum(weight)


class MLPDataset(Dataset):
    def __init__(self, X, Y, W):
        super(MLPDataset, self).__init__()

        self.X = X.reshape(-1, 1)
        self.Y = Y.reshape(-1, 1)
        self.W = W.reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]


def train_mlp(mlp, train_loader, val_loader, learning_rate, n_epochs, device, best_model="./outputs/best_model.pth"):
    os.makedirs(os.path.dirname(best_model), exist_ok=True)

    mlp.float().to(device)

    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)
    criterion = MLPLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    def train_one_epoch(mlp, data_loader, criterion, optimizer, device):
        mlp.train()
        train_loss = 0.
        for X_batch, Y_batch, W_batch in data_loader:
            X_batch = X_batch.float().to(device)
            Y_batch = Y_batch.float().to(device)
            W_batch = W_batch.float().to(device)

            logit = mlp(X_batch)
            loss = criterion(logit, Y_batch, W_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        return train_loss / len(data_loader)

    @torch.no_grad()
    def val_one_epoch(mlp, data_loader, criterion, device):
        mlp.eval()
        val_loss = 0.
        for X_batch, Y_batch, W_batch in data_loader:
            X_batch = X_batch.float().to(device)
            Y_batch = Y_batch.float().to(device)
            W_batch = W_batch.float().to(device)

            logit = mlp(X_batch)
            loss = criterion(logit, Y_batch, W_batch)

            val_loss += loss.item()

        return val_loss / len(data_loader)

    best_val_loss = float("inf")
    best_epoch = 0.
    train_history, val_history = [], []

    for epoch in range(n_epochs):
        avg_train_loss = train_one_epoch(mlp, train_loader, criterion, optimizer, device)
        avg_val_loss = val_one_epoch(mlp, val_loader, criterion, device)

        scheduler.step(avg_val_loss)
        # print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(mlp.state_dict(), best_model)

        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)

    print(f"Loading best model with val loss {best_val_loss:.4f}, epoch {best_epoch}")
    mlp.load_state_dict(torch.load(best_model))

    return mlp, np.array(train_history), np.array(val_history)


@torch.no_grad()
def calculate_likelihood_ratio(mlp, data_loader, device):
    mlp.float().to(device)
    mlp.eval()
    Xs, Ws = [], []
    for X_batch, _, W_batch in data_loader:
        X_batch = X_batch.float().to(device)
        W_batch = W_batch.float().to(device)

        logit = mlp.log_likelihood_ratio(X_batch)

        Xs.append(X_batch.cpu())
        Ws.append(W_batch* torch.exp(logit))

    return torch.cat(Xs).numpy().flatten(), torch.cat(Ws).cpu().numpy().flatten()
