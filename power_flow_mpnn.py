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
    
def train_model(model, dataset, save_filepath, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
    
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
    batch_size=32,
    device='cpu'):
    
    model = model.to(device)
    loader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    
    preds = []
    targets = []
    
    with torch.no_grad():
        
        for data in loader:
            data = data.to(device)
            pred = model(data)
            preds.append(pred.cpu())
            targets.append(data.y.cpu())
            
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    
    metrics = {}
    var_names = ['V', 'Theta', 'P', 'Q']
    
    for i, name in enumerate(var_names):
        
        target = targets[:, i]
        pred = preds[:, i]
        
        mse = F.mse_loss(pred, target).item()
        rmse = np.sqrt(mse)
        mae = F.l1_loss(pred, target).item()
        
        value_range = (target.max() - target.min()).item()
        nrmse = rmse / value_range if value_range > 0 else np.nan
        
        ss_res = torch.sum((target - pred)**2).item()
        ss_tot = torch.sum((target - target.mean())**2).item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'NRMSE': nrmse,
            'R2': r2}
        
    return preds_np, targets_np, metrics
            
if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    train_data_filepath = os.path.join(data_dir, 'case14_train.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_test.npy')

    train_data = np.load(train_data_filepath, allow_pickle=True).item()
    test_data = np.load(test_data_filepath, allow_pickle=True).item()
    
    model = PowerFlowGNN()
    save_filepath = 'case14_model.pth'
    
    train_model(model, train_data, save_filepath)
    model.load_state_dict(torch.load(save_filepath, weights_only=True))
    test_loss = test_model(model, test_data)