# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:54:48 2026

@author: codett
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# what is the difference between a torch vector and an np array?
# What's the nn.Module class that's being inherited?
# what is nn.Sequential?
# what does it mean to add batch dimension to edge features and why is this required?
# what does torch cat dim=-1 do?
# what does scatter_add do?
# what does the encoder that turns the node features into the hidden representation do?
# how exactly does nn.ModuleList work?

class PowerFlowMPNN(nn.Module):
    
    def __init__(self, hidden_dim=128, edge_dim=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Message = f(h_i, h_j, edge_features)
        # Inputs:
        # Source node state: hidden_dim
        # Desination node state: hidden_dim
        # Edge features: edge_dim
        # Total input dimension = 2 * hidden_dim + edge_dim
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        # Node uses incoming messages and current state to update itself
        # Input:
        # Current node state: hidden_dim
        # Aggregated messages: hidden_dim
        # Total input dimension = 2 * hidden_dim
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
    def compute_messages(self, h, edge_index, edge_attr):
        """
        COMPUTE ONE MESSAGE FOR EVERY DIRECTED EDGE

        Parameters
        ----------
        h : [batch, num_nodes, hidden_dim]
        edge_index : [2, num_edges]
        edge_attr : [num_edges, edge_dim]

        Returns
        -------
        messages : [batch, num_edges, hidden_dim]
        None
        """
        
        source = edge_index[0]
        destination = edge_index[1]
        
        # Node states at the two ends of every edge
        h_source = h[:, source, :]
        h_destination = h[:, destination, :]
        
        # Add batch dimension to edge features
        edge_features = edge_attr.unsqueeze(0)
        edge_features = edge_features.expand(h.size(0), -1, -1)
        
        # Combine source state, destination state, edge features
        message_input = torch.cat([h_source, h_destination, edge_features], dim=-1)
        return self.message_net(message_input)
    
    def aggregate_messages(self, messages, edge_index, num_nodes):
        """
        SUM INCOMING MESSAGES AT EACH DESTINATION NODE.

        Parameters
        ----------
        messages : [batch, num_edges, hidden_dim]
        edge_index : [2, num_edges]
        num_nodes : int

        Returns
        -------
        aggregated : [batch, num_nodes, hidden_dim]
        """
        
        destination = edge_index[1]
        
        aggregated = torch.zeros(
            messages.size(0),
            num_nodes,
            self.hidden_dim,
            device=messages.device,
            dtype=messages.dtype)
        
        aggregated.scatter_add_(
            dim=1,
            index=destination
                .view(1, -1, 1)
                .expand_as(messages),
            src=messages)
        
        return aggregated
    
    def update_nodes(self, h, aggregated_messages):
        """
        UPDATE EACH NODE USING ITS CURRENT STATE AND MESSAGES FROM ITS NEIGHBORS

        Parameters
        ----------
        h : [batch, num_nodes, hidden_dim]
        aggregated_messages : [batch, num_nodes, hidden_dim]

        Returns
        -------
        h : [batch, num_nodes, hidden_dim]
        """
        
        update_input = torch.cat([h, aggregated_messages], dim=-1)
        update = self.update_net(update_input)
        
        # Residual connection: new_state = old_state + learned_update
        return h + update
    
    def forward(self, h, edge_index, edge_attr):
        """
        COMPUTES MESSAGES AND UPDATES NODES

        Parameters
        ----------
        h : [batch, num_nodes, hidden_dim]
        edge_index : [2, num_edges]
        edge_attr : [2, num_edges]

        Returns
        -------
        h : [batch, num_nodes, hidden_dim]
        """
        
        # 1. Compute messages along every edge
        messages = self.compute_messages(h, edge_index, edge_attr)
        
        # 2. Sum incoming messages at each node
        aggregated_messages = self.aggregate_messages(messages, edge_index, num_nodes=h.size(1))
        
        # 3. Update node representations
        h = self.update_nodes(h, aggregated_messages)
        
        return h
    
class PowerFlowGNN(nn.Module):
    
    def __init__(
            self,
            in_dim=8,
            hidden_dim=128,
            out_dim=4,
            edge_dim=2,
            num_layers=5):
        super().__init__()
        
        # Encode original node features into hidden representation used by the GNN.
        # Input: P, V, Q, Theta, mP, mV, mQ, mTheta
        # Output: hidden_dim features per node
        self.encoder = nn.Linear(in_dim, hidden_dim)
        
        # Repeated message-passing layers
        # Each layer allows information to propogate one step further in the network.
        self.message_passing_layers = nn.ModuleList([
            PowerFlowMPNN(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim)
            for _ in range(num_layers)])
        
        # Convert the final hidden representation back into [P, V, Q, Theta]
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim))
        
    def forward(self, x, edge_index, edge_attr):
        """
        PERFORM MESSAGE PASSING AND PREDICTS POWER-FLOW VARIABLES

        Parameters
        ----------
        x : Node features [batch, num_nodes, in_dim]
        edge_index : Directed graph connectivity [2, num_edges]
        edge_attr : Edge features [num_edges, edge_dim]

        Returns
        -------
        Node predictions [batch, num_nodes, out_dim]
        """
        
        # Step 1. Encode input features
        h = self.encoder(x)
        
        # Step 2. Propagate information through the graph
        for layer in self.message_passing_layers: 
            h = layer(h, edge_index, edge_attr)
            
        # Step 3. Predict P, V, Q, Theta
        output = self.readout(h)
        
        return output
    
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
    edge_index = torch.tensor(dataset['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(dataset['edge_attr'], dtype=torch.float32).to(device)
    masks = X[:, :, 4:8]
    weights = torch.tensor(loss_weights, dtype=torch.float32).to(device)
    
    loss_history = []
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X))
        
        for start in tqdm(range(0, len(X), batch_size), desc=f'Epoch {epoch + 1}/{epochs}'):
            idx = indices[start:start + batch_size]
            x = X[idx].to(device)
            y = Y[idx].to(device)
            mask = masks[idx].to(device)
            
            optimizer.zero_grad()
            pred = model(x, edge_index, edge_attr)
            
            # Compute weighted MSE only for unknown variables.
            unknown = 1 - mask
            loss = ((pred - y) ** 2 * unknown * weights).sum()
            loss /= (unknown * weights).sum()
            
            loss.backward()
            optimizer.step()
            
        loss_history.append(loss.item())
            
    torch.save(model.state_dict(), save_filepath)
    return loss_history

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
    return data

def predict(
        model,
        dataset,
        batch_size=32,
        device='cpu',
        loss_weights=[1, 1, 1, 1]):
    
    model = model.to(device).eval()
    X = torch.tensor(dataset['X'], dtype=torch.float32)
    Y = torch.tensor(dataset['Y'], dtype=torch.float32)
    edge_index = torch.tensor(dataset['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(dataset['edge_attr'], dtype=torch.float32).to(device)
    
    masks = X[:, :, 4:8]
    weights = torch.tensor(loss_weights, dtype=torch.float32).to(device)
    
    preds = []
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i: i + batch_size].to(device)
            y = Y[i: i + batch_size].to(device)
            mask = masks[i: i + batch_size].to(device)
            
            pred = model(x, edge_index, edge_attr)
            preds.append(pred.cpu().numpy())
            
            # Testing loss calculation
            weights = torch.tensor(loss_weights, dtype=torch.float32, device=device).view(1, 1, 4)
            unknown = (1 - mask).float()
            loss = ((pred - y) ** 2 * unknown * weights).sum()
            loss /= (unknown * weights).sum()
            
            total_loss += loss.item()
            total_batches += 1
            
    testing_loss = total_loss / total_batches
    return np.concatenate(preds), Y.numpy(), testing_loss

def compute_metrics(preds, targets, mask):
    
    var_names = np.array(['P', 'V', 'Q', 'Theta'])
    mask = ~mask
    
    metrics = {}
    for i, name in enumerate(var_names):
        
        pred = preds[:, :, i][mask[:, :, i]]
        target = targets[:, :, i][mask[:, :, i]]
        error = pred - target
        bias = np.mean(error)
        mse = np.mean(error ** 2)
        denom = np.sum((target - target.mean()) ** 2)
        if denom > 0: R2 = 1 - np.sum(error ** 2) / denom
        else: R2 = np.nan
        metrics[name] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'bias': bias,
            'R2': R2}
        
    return metrics

def compute_bus_metrics(preds, targets, mask):
    
    var_names = ['P', 'V', 'Q', 'Theta']
    mask = ~mask
    
    bus_metrics = {}
    for bus in range(preds.shape[1]):
        metrics = {}
        for i, name in enumerate(var_names):
            valid = mask[:, bus, i]
            if not valid.any(): continue
            
            pred = preds[:, bus, i][valid]
            target = targets[:, bus, i][valid]
            error = pred - target
            bias = np.mean(error)
            mse = np.mean(error ** 2)
            denom = np.sum((target - target.mean()) ** 2)
            if denom > 0: R2 = 1 - np.sum(error ** 2) / denom
            else: R2 = np.nan
            metrics[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'bias': bias,
                'R2': R2}
        bus_metrics[bus] = metrics
    return bus_metrics

def print_metrics(metrics):
    print(f"{'Variable':<10} {'MSE':>12} {'RMSE':>12} {'Bias':>12} {'R²':>12}")
    print("-" * 60)

    for name, values in metrics.items():
        print(
            f"{name:<10} "
            f"{values['mse']:>12.6f} "
            f"{values['rmse']:>12.6f} "
            f"{values['bias']:>12.6f} "
            f"{values['R2']:>12.6f}"
        )

def test_model(
        model,
        test_dataset,
        results_filepath,
        per_unit=False,
        batch_size=32,
        device='cpu',
        include_knowns=True):
    
    preds, targets, testing_loss = predict(model, test_dataset, batch_size, device)
    
    # Convert units if necessary
    bases = test_dataset['bases']
    if test_dataset['per_unit'] != per_unit:
        convert = convert_to_per_unit if per_unit else convert_to_absolute 
        preds = convert(preds, bases)
        targets = convert(targets, bases)
        
    var_names = np.array(['P', 'V', 'Q', 'Theta'])
    masks = test_dataset['X'][:, :, 4:8].astype(bool)
    evaluation_mask = (
        np.ones_like(masks, dtype=bool)
        if include_knowns else ~masks)
    
    metrics = {}
    bus_rmse = np.full((targets.shape[1], 4), np.nan)
    
    for i, name in enumerate(var_names):
        
        pred = preds[:, :, i][evaluation_mask[:, :, i]]
        target = targets[:, :, i][evaluation_mask[:, :, i]]

        error = pred - target
        mse = np.mean(error ** 2)
        
        metrics[name] = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': np.mean(np.abs(error)),
            'R2': (
                1 - np.sum(error ** 2) /
                np.sum((target - target.mean()) ** 2)
                if np.sum((target - target.mean()) ** 2) > 0
                else np.nan),
            'Mean Bias': np.mean(error)}
        
        # RMSE for each bus.
        for bus in range(targets.shape[1]):
            mask = evaluation_mask[:, bus, i]

            if mask.any():
                bus_rmse[bus, i] = np.sqrt(
                    np.mean(
                        (preds[:, bus, i][mask] -
                         targets[:, bus, i][mask]) ** 2))
                
    results = {
        'preds': preds,
        'targets': targets,
        'Y_labels': var_names,
        'metrics': metrics,
        'bus_rmse': bus_rmse,
        'metrics_labels': np.array(
            ['P_RMSE', 'V_RMSE', 'Q_RMSE', 'Theta_RMSE']),
        'per_unit': per_unit,
        'bases': bases}
    
    np.save(results_filepath, results, allow_pickle=True)
    
def plot_loss_history(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
def plot_bus_metrics(bus_metrics):
    variables = ['P', 'V', 'Q', 'Theta']
    metrics = ['mse', 'rmse', 'bias', 'R2']

    bus_indices = sorted(bus_metrics.keys())

    for variable in variables:
        for metric in metrics:
            values = [
                bus_metrics[bus].get(variable, {}).get(metric, np.nan)
                for bus in bus_indices
            ]

            plt.figure()
            plt.bar(bus_indices, values)
            plt.xlabel('Bus Index')
            plt.ylabel(metric.upper())
            plt.title(f'{variable} {metric.upper()} by Bus')
            plt.xticks(bus_indices)
            plt.grid(axis='y')
            plt.show()
    
if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    train_data_filepath = os.path.join(data_dir, 'case14_32sample_train.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_32sample_test.npy')
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_filepath = os.path.join(models_dir, '32_sample_mdl.npy')
    
    results_dir = os.path.join(base_dir, 'results')
    results_folderpath = os.path.join(results_dir, '32sample')
    os.makedirs(results_folderpath, exist_ok=True)
    results_filepath = os.path.join(results_folderpath, 'case14_results.npy')
    loss_history_filepath = os.path.join(results_folderpath, 'loss_history.npy')
    
    train_data = np.load(train_data_filepath, allow_pickle=True).item()
    test_data = np.load(test_data_filepath, allow_pickle=True).item()
    
    # Make sure data is in per_unit, test conversion
    test_conversion = False
    if test_conversion:
        per_unit = test_data['per_unit']
        Y, bases = test_data['Y'], test_data['bases']
        Y_roundtrip = convert_to_per_unit(convert_to_absolute(Y,bases), bases)
        error = np.max(np.abs(Y_roundtrip - Y))
    
    # Model and Training Parameters
    num_layers = 5
    model = PowerFlowGNN(num_layers=num_layers)
    epochs = 500
    lr = 1e-3
    loss_weights = [1, 1, 1, 1]
    
    model = PowerFlowGNN(num_layers=num_layers)
    
    train = False # already done
    if train: 
        loss_history = train_model(
            model, 
            train_data,
            model_filepath,
            epochs=epochs,
            batch_size=32,
            lr=lr,
            device='cpu',
            loss_weights=loss_weights)
        np.save(loss_history_filepath, loss_history)
    
    model.load_state_dict(torch.load(model_filepath, weights_only=True))
    # loss_history = np.load(loss_history_filepath)
    # plot_loss_history(loss_history)
    
    mask = test_data['X'][:, :, 4:8].astype(bool)
    bases = test_data['bases']
    
    preds_pu, targets_pu, testing_loss = predict(model, test_data, batch_size=32, device='cpu')
    metrics_pu = compute_metrics(preds_pu, targets_pu, mask)
    bus_metrics_pu = compute_bus_metrics(preds_pu, targets_pu, mask)
    
    preds_absolute = convert_to_absolute(preds_pu, bases)
    targets_absolute = convert_to_absolute(targets_pu, bases)
    metrics_absolute = compute_metrics(preds_absolute, targets_absolute, mask)
    bus_metrics_absolute = compute_bus_metrics(preds_absolute, targets_absolute, mask)
    
    print('Per-Unit Metrics:\n')
    print_metrics(metrics_pu)
    print('Absolute Metrics:\n')
    print_metrics(metrics_absolute)
    
    print('Per-Unit Slack-Bus Metrics:\n')
    print_metrics(bus_metrics_pu[0])
    print('Absolute Slack-Bus Metrics:\n')
    print_metrics(bus_metrics_absolute[0])
    
    plot_bus_metrics(bus_metrics_absolute)
    
    print('')