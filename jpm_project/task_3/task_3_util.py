import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np


def display_df_summary(df: pd.DataFrame):
    """
    Display main information and summary statistics for a DataFrame.
    """
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    print("\nDataset Info:")
    df.info()

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nBasic Statistics:")
    print(df.describe())

    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# --- PyTorch MLP utility ---

def create_mlp(input_dim, hidden_dims, output_dim=1, activation=nn.ReLU):
    """
    Create a simple MLP model for tabular data.
    Args:
        input_dim (int): Number of input features
        hidden_dims (list of int): List with number of units in each hidden layer
        output_dim (int): Number of output units (default 1 for binary classification)
        activation (nn.Module): Activation function class (default nn.ReLU)
    Returns:
        nn.Module: PyTorch MLP model
    """
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu'):
    """
    Train a PyTorch model with DataLoader objects for train and validation sets.
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: loss function
        optimizer: optimizer
        epochs: number of epochs
        device: 'cpu' or 'cuda'
    Returns:
        train_losses, val_losses, best_model_weights, model
    """
    best_val_loss = float('inf')
    best_model_weights = None
    train_losses = []
    val_losses = []
    model.to(device)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict()

        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Time: {end_time - start_time:.2f}s")

    return train_losses, val_losses, best_model_weights, model
  
def evaluate_mlp(model, data_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy().flatten())
    return np.array(all_preds), np.array(all_targets)




