# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:49:53 2026

@author: codett
"""

import os
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import random
import numpy as np

import matplotlib.pyplot as plt

# Single message-passing layer.
# Represent one iteration of message passing.

class PowerFlowMPNN(MessagePassing):
    
    def __init__(self, hidden_dim=128):
        super().__init__(aggr='add') 
        
        self.message_net = nn.Sequential(
            nn.Linear(2*hidden_dim+2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.update_net = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
    
    def message(self, h_i, h_j, edge_attr):
        x = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.message_net(x)
    
    def update(self, aggr_out, h):
        x = torch.cat([h, aggr_out], dim=-1)
        return h + self.update_net(x) # Residual connection
    
    def forward(self, h, edge_index, edge_attr):
        return self.propagate(
            edge_index,
            h=h,
            edge_attr=edge_attr)
    
class PowerFlowGNN(nn.Module):

    def __init__(
        self,
        in_dim=8,
        edge_dim=2,
        hidden_dim=128,
        out_dim=4,
        num_layers=5):
        
        super().__init__()
        
        # Node encoder
        self.encoder = nn.Linear(in_dim, hidden_dim)
        
        # Message-passing layers
        self.layers = nn.ModuleList([
            PowerFlowMPNN(hidden_dim)
            for _ in range(num_layers)])
        
        # Readout Network
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim))
        
    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.readout(h)
    
def train_model(
        model,
        dataset,
        save_filepath,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        device='cpu',
        loss_weights=[1, 1, 1, 1]):
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(dataset['X'], dtype=torch.float32)
    Y = torch.tensor(dataset['Y'], dtype=torch.float32)
    edge_index = torch.tensor(dataset['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(dataset['edge_attr'], dtype=torch.float32)
    
    masks = X[:, :, 4:8]
    weights = torch.tensor(loss_weights, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X))
        
        for i in tqdm(range(0, len(X), batch_size),
                       desc=f"Epoch {epoch+1}/{epochs}"):
            
            idx = indices[i:i+batch_size]
            x, y = X[idx].to(device), Y[idx].to(device)
            mask = masks[idx].to(device)
            ea = edge_attr[idx].to(device)
            
            optimizer.zero_grad()
            pred = model(x, edge_index.to(device), ea)
            
            loss = ((pred - y)**2 * (1 - mask) * weights.to(device)).sum()
            loss /= ((1 - mask) * weights.to(device)).sum()
            
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), save_filepath)
    
def train_model_old(model, dataset, save_filepath, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X = torch.tensor(dataset['X'], dtype=torch.float32)
    Y = torch.tensor(dataset['Y'], dtype=torch.float32)
    
    edge_index = torch.tensor(dataset['edge_index'])
    edge_attr = torch.tensor(dataset['edge_attr'], dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X))
        total_loss = 0
        
        for i in tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            idx = indices[i:i+batch_size]
            x, y = X[idx].to(device), Y[idx].to(device)
            ea = edge_attr[idx].to(device)
            ei = edge_index.to(device)
            
            optimizer.zero_grad()
            pred = model(x, ei, ea)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    torch.save(model.state_dict(), save_filepath)
    
def test_model(
        model,
        test_dataset,
        results_filepath,
        batch_size=32,
        device='cpu',
        include_knowns=True):
    
    model = model.to(device)
    model.eval()

    X = torch.tensor(test_dataset['X'], dtype=torch.float32)
    Y = torch.tensor(test_dataset['Y'], dtype=torch.float32)
    edge_index = torch.tensor(test_dataset['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(test_dataset['edge_attr'], dtype=torch.float32)

    preds, targets = [], []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i:i+batch_size].to(device)
            y = Y[i:i+batch_size].to(device)
            ea = edge_attr[i:i+batch_size].to(device)
            pred = model(x, edge_index.to(device), ea)
            preds.append(pred.cpu())
            targets.append(y.cpu())
            
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    var_names = np.array(['P', 'V', 'Q', 'Theta'])
    masks = X[:, :, 4:8]
    
    if include_knowns:
        eval_preds, eval_targets = preds, targets
        unknown = torch.ones_like(masks, dtype=torch.bool)
    else:
        unknown = masks == 0
        eval_preds, eval_targets = preds.clone(), targets.clone()
        eval_preds[~unknown] = float('nan')
        eval_targets[~unknown] = float('nan')
        
    metrics = {}
    bus_metrics = np.full((Y.shape[1], 4), np.nan)
    
    for i, name in enumerate(var_names):
        if include_knowns:
            pred, target = preds[:, :, i], targets[:, :, i]
        else:
            pred = eval_preds[:, :, i][unknown[:, :, i]]
            target = eval_targets[:, :, i][unknown[:, :, i]]

        mse = F.mse_loss(pred, target).item()
        rmse = np.sqrt(mse)
        mae = F.l1_loss(pred, target).item()

        ss_res = torch.sum((target - pred)**2).item()
        ss_tot = torch.sum((target - target.mean())**2).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        bias = torch.mean(pred - target).item()

        slope = np.polyfit(
            target.numpy(),
            pred.numpy(),
            1
        )[0]

        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Mean Bias': bias,
            'Regression Slope': slope
        }

        for bus in range(Y.shape[1]):
            mask = unknown[:, bus, i]
            if mask.any():
                bus_metrics[bus, i] = torch.sqrt(
                    F.mse_loss(
                        preds[:, bus, i][mask],
                        targets[:, bus, i][mask]
                    )
                ).item()

    results = {
        'preds': eval_preds.numpy(),
        'targets': eval_targets.numpy(),
        'Y_labels': var_names,
        'metrics': metrics,
        'bus_metrics': bus_metrics,
        'metrics_labels': np.array([
            'P_RMSE', 'V_RMSE', 'Q_RMSE', 'Theta_RMSE'
        ])
    }

    np.save(results_filepath, results, allow_pickle=True)
    return results
    
def test_model_old(
        model,
        test_dataset,
        results_filepath,
        batch_size=32,
        device='cpu',
        include_knowns=True):

    model = model.to(device)
    model.eval()

    X = torch.tensor(test_dataset['X'], dtype=torch.float32)
    Y = torch.tensor(test_dataset['Y'], dtype=torch.float32)
    edge_index = torch.tensor(test_dataset['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(test_dataset['edge_attr'], dtype=torch.float32)

    preds, targets = [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i:i+batch_size].to(device)
            y = Y[i:i+batch_size].to(device)
            ea = edge_attr[i:i+batch_size].to(device)
            pred = model(x, edge_index.to(device), ea)
            preds.append(pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    var_names = np.array(['P', 'V', 'Q', 'Theta'])
    masks = X[:, :, 4:8]

    if include_knowns:
        eval_preds, eval_targets = preds, targets
        unknown = torch.ones_like(masks, dtype=torch.bool)
    else:
        unknown = masks == 0
        eval_preds, eval_targets = preds.clone(), targets.clone()
        eval_preds[~unknown] = float('nan')
        eval_targets[~unknown] = float('nan')

    metrics = {}
    bus_metrics = np.full((Y.shape[1], 4), np.nan)

    for i, name in enumerate(var_names):
        if include_knowns:
            pred, target = preds[:, :, i], targets[:, :, i]
        else:
            pred = eval_preds[:, :, i][unknown[:, :, i]]
            target = eval_targets[:, :, i][unknown[:, :, i]]

        mse = F.mse_loss(pred, target).item()
        rmse = np.sqrt(mse)
        mae = F.l1_loss(pred, target).item()
        ss_res = torch.sum((target - pred)**2).item()
        ss_tot = torch.sum((target - target.mean())**2).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        for bus in range(Y.shape[1]):
            mask = unknown[:, bus, i]
            if mask.any():
                bus_metrics[bus, i] = torch.sqrt(
                    F.mse_loss(
                        preds[:, bus, i][mask],
                        targets[:, bus, i][mask])).item()

    results = {
        'preds': eval_preds.numpy(),
        'targets': eval_targets.numpy(),
        'Y_labels': var_names,
        'metrics': metrics,
        'bus_metrics': bus_metrics,
        'metrics_labels': np.array([
            'P_RMSE', 'V_RMSE', 'Q_RMSE', 'Theta_RMSE'])
    }

    np.save(results_filepath, results, allow_pickle=True)
    return results

def print_metrics(metrics):
    for name, values in metrics.items():
        print(f'\n{name}:')
        for metric, value in values.items():
            print(f'  {metric}: {value:.6f}')
            
def plot_distributions(targets, preds, variable_list, bus=None):
    for i, variable in enumerate(variable_list):
        if bus is None:
            true = targets[:, :, i].ravel()
            pred = preds[:, :, i].ravel()
        else:
            true = targets[:, bus-1, i]
            pred = preds[:, bus-1, i]

        if np.all(np.isnan(true)) or np.all(np.isnan(pred)):
            continue

        plt.figure()
        plt.hist(true, bins=50, alpha=0.5, label='True')
        plt.hist(pred, bins=50, alpha=0.5, label='Predicted')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.title(
            f'{variable} Distribution' +
            (f' — Bus {bus}' if bus is not None else '')
        )
        plt.legend()
        plt.show()
    
def plot_distributions_old(targets, preds, variable_list, bus=None):
    for i, variable in enumerate(variable_list):
        plt.figure()
        plt.hist(targets[:, :, i].ravel(), bins=50, alpha=0.5, label='True')
        plt.hist(preds[:, :, i].ravel(), bins=50, alpha=0.5, label='Predicted')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.title(f'{variable} Distribution')
        plt.legend()
        plt.show()
        
def plot_bus_metrics(bus_metrics, metrics_labels):
    plt.figure()
    buses = np.arange(1, len(bus_metrics) + 1)

    for i, label in enumerate(metrics_labels):
        plt.plot(buses, bus_metrics[:, i], marker='o', label=label)

    plt.xlabel('Bus Number')
    plt.ylabel('RMSE (pu)')
    plt.xticks(buses)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def plot_scatterplots(targets, preds, variable_list, bus_list):
    var_indices = {'P': 0, 'V': 1, 'Q': 2, 'Theta': 3}

    for bus in bus_list:
        for variable in variable_list:
            i = var_indices[variable]

            true = targets[:, bus-1, i]
            pred = preds[:, bus-1, i]

            mask = ~np.isnan(true) & ~np.isnan(pred)
            if not np.any(mask):
                continue

            x = np.arange(len(true))[mask]
            true, pred = true[mask], pred[mask]

            true_coeff = np.polyfit(x, true, 1)
            pred_coeff = np.polyfit(x, pred, 1)

            x_fit = np.linspace(x.min(), x.max(), 200)
            true_fit = np.polyval(true_coeff, x_fit)
            pred_fit = np.polyval(pred_coeff, x_fit)

            plt.figure(figsize=(8, 5))
            plt.scatter(x, true, s=15, alpha=0.25, label='True')
            plt.scatter(x, pred, s=15, alpha=0.25, label='Predicted')
            plt.plot(x_fit, true_fit, '--', linewidth=2.5, label='True best fit')
            plt.plot(x_fit, pred_fit, '-', linewidth=2.5, label='Predicted best fit')

            plt.xlabel('Sample Index')
            plt.ylabel(f'{variable} (pu)')
            plt.title(f'{variable} — Bus {bus}')
            plt.legend()
            plt.tight_layout()
            plt.show()
    
def plot_pred_vs_true(targets, preds, variable_list, bus_list):
    var_indices = {'P':0, 'V':1, 'Q':2, 'Theta':3}
    for bus in bus_list:
        for variable in variable_list:
            i = var_indices[variable]
            true = targets[:, bus-1, i]
            pred = preds[:, bus-1, i]
            mask = ~np.isnan(true) & ~np.isnan(pred)
            if not np.any(mask): continue
            true, pred = true[mask], pred[mask]
            lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            plt.figure(figsize=(6, 6))
            plt.scatter(true, pred, s=15, alpha=0.25)
            plt.plot(lims, lims, 'k--', linewidth=2, label='Identity')
            plt.xlabel('True (pu)')
            plt.ylabel('Predicted (pu)')
            plt.title(f'{variable} — Bus {bus}')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    train_data_filepath = os.path.join(data_dir, 'case14_train3.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_test3.npy')
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_filepath = os.path.join(results_dir, 'case14_results.npy')

    train_data = np.load(train_data_filepath, allow_pickle=True).item()
    test_data = np.load(test_data_filepath, allow_pickle=True).item()
    
    model = PowerFlowGNN()
    save_filepath = 'case14_model.pth'
    
    train_model(model, train_data, save_filepath)
    model.load_state_dict(torch.load(save_filepath, weights_only=True))
    
    test_model(model, test_data, results_filepath, include_knowns=False)
    results = np.load(results_filepath, allow_pickle=True).item()
    
    print_metrics(results['metrics'])
    
    # plot_bus_metrics(results['bus_metrics'], results['metrics_labels'])
    
    # plot_distributions(results['targets'], results['preds'],['P', 'V', 'Q', 'Theta'],bus=3)
    
    # plot_scatterplots(results['targets'], results['preds'], ['Q', 'Theta'], [2,3])
    
    # plot_pred_vs_true(results['targets'], results['preds'], ['Q', 'Theta'], [2,3])
    
    print('')