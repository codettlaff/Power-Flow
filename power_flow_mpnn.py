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
    
def convert_to_per_unit(data, bases):
    S_base = bases['S_base']
    V_base = bases['V_base']

    data = data.copy()
    data[..., 0] /= S_base  # P
    data[..., 1] /= V_base  # V
    data[..., 2] /= S_base  # Q

    return data


def convert_to_absolute(data, bases):
    S_base = bases['S_base']
    V_base = bases['V_base']

    data = data.copy()
    data[..., 0] *= S_base  # P
    data[..., 1] *= V_base  # V
    data[..., 2] *= S_base  # Q
    # data[..., 3] = Theta unchanged

    return data

def predict_model(model, dataset, batch_size=32, device='cpu'):
    model = model.to(device).eval()
    
    X = dataset['X']
    Y = dataset['Y']
    edge_index = torch.tensor(dataset['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(dataset['edge_attr'], dtype=torch.float32).to(device)
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            preds.append(model(x, edge_index, edge_attr).cpu().numpy())
            
    return np.concatenate(preds), Y

def test_model(
        model,
        test_dataset,
        results_filepath,
        per_unit=False,
        batch_size=32,
        device='cpu',
        include_knowns=True):
    
    preds, targets = predict_model(model, test_dataset, batch_size, device)
    var_names = np.array(['P', 'V', 'Q', 'Theta'])
    masks = test_dataset['X'][:, :, 4:8].astype(bool)
    unknown = masks if not include_knowns else np.ones_like(masks, dtype=bool)
    
    # Convert units if necessary
    bases = test_dataset['bases']
    if test_dataset['per_unit'] != per_unit:
        convert = convert_to_per_unit if per_unit else convert_to_absolute 
        preds = convert(preds, bases)
        targets = convert(targets, bases)
        
    metrics = {}
    bus_metrics = np.full((targets.shape[1], 4), np.nan)
    
    for i, name in enumerate(var_names):
        pred = preds[:, :, i][unknown[:, :, i]]
        target = targets[:, :, i][unknown[:, :, i]]
        error = pred - target
        mse = np.mean(error ** 2)
        metrics[name] = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': np.mean(np.abs(error)),
            'R2': 1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2),
            'Mean Bias': np.mean(error),
            'Regression Slope': np.polyfit(target, pred, 1)[0]}
        
        for bus in range(targets.shape[1]):
            mask = unknown[:, bus, i]
            if mask.any():
                bus_metrics[bus, i] = np.sqrt(
                    np.mean((preds[:, bus, i][mask] -
                             targets[:, bus, i][mask]) ** 2))
                
    results = {
        'preds': preds,
        'targets': targets,
        'Y_labels': var_names,
        'metrics': metrics,
        'bus_rmse': bus_metrics,
        'metrics_labels': np.array(['P_RMSE', 'V_RMSE', 'Q_RMSE', 'Theta_RMSE']),
        'per_unit': per_unit,
        'bases': bases}
    return results
            
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
    train_data_filepath = os.path.join(data_dir, 'case14_train1.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_test1.npy')
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_filepath = os.path.join(results_dir, 'case14_results1.npy')

    train_data = np.load(train_data_filepath, allow_pickle=True).item()
    test_data = np.load(test_data_filepath, allow_pickle=True).item()
    
    model = PowerFlowGNN()
    save_filepath = 'case14_model1.pth'
    
    # train_model(model, train_data, save_filepath)
    model.load_state_dict(torch.load(save_filepath, weights_only=True))
    
    # test_model(model, test_data, results_filepath, include_knowns=False)
    results = np.load(results_filepath, allow_pickle=True).item()
    
    # print_metrics(results['metrics'])
    
    # plot_bus_metrics(results['bus_metrics'], results['metrics_labels'])
    
    plot_distributions(results['targets'], results['preds'],['P', 'V', 'Q', 'Theta'], bus=1)
    
    plot_scatterplots(results['targets'], results['preds'], ['P', 'Q'], [1])
    
    plot_pred_vs_true(results['targets'], results['preds'], ['P', 'Q'], [1])
    
    print('')