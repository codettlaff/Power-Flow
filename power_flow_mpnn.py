# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:49:53 2026

@author: codett
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import random

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
        
    def forward(self, data):
        h = self.encoder(data.x)
        for layer in self.layers:
            h = layer(
                h,
                data.edge_index,
                data.edge_attr)
        return self.readout(h)
    
def train_model(
    model,
    dataset,
    save_filepath,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    device='cpu'):
    
    model = model.to(device)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr)
    
    for epoch in range(epochs):
        
        model.train()
        total_loss = 0.0
        
        for data in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    torch.save(model.state_dict(), save_filepath)
            
def test_model(
    model,
    dataset,
    batch_size=32,
    device='cpu'):
    
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        
        for data in loader:
            
            data = data.to(device)
            pred = model(data)
            loss = F.mse_loss(pred, data.y)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(loader)
    return avg_loss
            
if __name__ == '__main__':

    train_data = torch.load('case14_train.pt', weights_only=False)
    test_data = torch.load('case14_test.pt', weights_only=False)
    
    model = PowerFlowGNN()
    
    save_filepath = 'case14_model.pth'
    
    # train_model(model, train_data, save_filepath) 
    model.load_state_dict(torch.load(save_filepath, weights_only=True))
    test_loss = test_model(model, test_data)         
        