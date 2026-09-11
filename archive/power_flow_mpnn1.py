# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:25:32 2026
Author: Casey Dettlaff
PowerFlowNet-style message-passing neural network.

Data structure:
    X : size([samples, buses, 8]) [P, V, Q, Theta, mP, mV, mQ, mTheta]
    Y : size([samples, buses, 4]) [P, V, Q, Theta]

Architecture:
Partially observed node features + binary mask ->
Mask encoder ->
Input projection ->
PowerFlowConv x (L - 1)
-- Message Network
-- Residual addition
-- Update Network
-- ReLU + Dropout
Final readout network ->
Fully observed node features 
"""

import os
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv

class PowerFlowMessageNetwork(nn.Module):
    """
    Implements edge-aware message network MPL
    For every directed edge i -> j:
        message_ij = MLP([h_i, h_j, e_ij])
        
    Messages are summed at their destination node.
    This corresponds to two-layer MLP in PowerFlowNet Eq.(10)
    """
    
    def __init__(
        self,
        nfeature_dim,
        efeature_dim,
        hidden_dim,
        output_dim):
        super().__init__()
        
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Two-layer MLP
        # input : [source_state, destination_state, edge_features] ->
        # Linear ->
        # ReLU ->
        # Linear ->
        # output : [message]
        self.network = nn.Sequential(
            nn.Linear(2 * nfeature_dim + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, h, edge_index, edge_attr):
        """
        Compute one learned message for every directed edge.
        Parameters:
            h : size([batch, num_nodes, nfeature_dim]) : Node states
            edge_index : size([2, num_edges]) : [from_node, to_node]
            edge_attr : size([num_edges, efeature_dim]) : [r, x]
        Returns:
            messages : size([batch, num_edges, output_dim]) TODO: should this have hidden_dim?
        """
        
        source = edge_index[0]
        destination = edge_index[1]
        
        h_source = h[:, source, :]
        h_destination = h[:, destination, :]
        
        edge_features = edge_attr.unsqueeze(0)
        edge_features = edge_features.expand(h.size(0), -1, -1)
        
        # Construct input : [h_i, h_j, e_ij]
        message_input = torch.cat([h_source, h_destination, edge_features], dim=-1)
        
        return self.network(message_input)
        
class PowerFlowMessagePassing(nn.Module):
    """
    Performs message aggregation.
    """
    
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, messages, edge_index, num_nodes):
        """
        Sum incoming messages at each destination node.
        Parameters: 
            messages : size([batch, num_edges, output_dim]) TODO: should this have hidden_dim?
            edge_index : size([2, num_edges]) : [from_node, to_node]
            num_nodes : int 
        Returns:
            aggregated : [batch, num_nodes, output_dim]
        """
        
        destination = edge_index[1]
        batch_size = messages.size(0)
        output_dim = messages.size(2)
        
        aggregated = torch.zeros(
            batch_size,
            num_nodes,
            output_dim,
            device = messages.device,
            dtype = messages.dtype)
        
        for node in range(num_nodes):
            incoming = destination == node
            if incoming.any():
                aggregated[:, node, :] = (messages[:, incoming, :].sum(dim=1))
        
        return aggregated
    
# PowerFlowNet replaces the update network with residual addition followed by TAGConv.
# TAGConv is a graph convolution / update operation.
# It takes the node states after the residual addition and propagates them over a larger neighborhood.
class PowerFlowUpdateNetwork(nn.Module):
    """
    Update Network
    Implements high-order node update in PowerFlowNet using TAGConv.
    First, add aggregated message to the current node state:
        h_hat = h + message
    The resulting state is then passed through a K-hop TAGConv:
        h_new = TAGConv_K(h_hat)
    TAGConv provides the higher-order graph update.
    A small wrapper is used because the project stores a batch as:
        [batch, num_nodes, hiddem_dim]
    While PyTorch Geometric TAGConv operates on:
        [num_nodes, hidden_dim]
    The batch is therefore temporarily represented as a disconnected collection of identical graphs.
    """
    
    def __init__(
        self,
        hidden_dim,
        K):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.K = K
        
        # PowerFlowNet uses a K-hop TAGConv after message passing.
        self.tag_conv = TAGConv(
            in_channels = hidden_dim,
            out_channels = hidden_dim,
            K=K)
        
    @staticmethod 
    def make_batched_edge_index(
        edge_index,
        batch_size,
        num_nodes):
        """
        Repeat the same graph for every sample in the batch.
        Graph 0 uses node indices [0, ..., N-1]
        Graph 1 uses node indices [N, ..., 2N-1]
        ect.
        
        This lets TAGConv process the entire batch in one operation.
        Without allowing information to pass between different samples.
        """
        
        device = edge_index.device
        
        offsets = (
            torch.arange(
                batch_size,
                device=device,
                dtype=edge_index.dtype)
            * num_nodes)
        
        offsets = (
            torch.arange(
                batch_size,
                device=device,
                dtype=edge_index.dtype)
            * num_nodes)
        
        offsets = offsets.view(-1, 1, 1)
        
        batched_edge_index = (
            edge_index.unsqueeze(0) + offsets)
        
        batched_edge_index = (
            batched_edge_index
            .permute(1, 0, 2)
            .reshape(2, -1))
        
        return batched_edge_index
    
    def forward(self, h, edge_index):
        """
        Apply K-hop TAGConv independently to every graph in the batch.
        """
        
        batch_size, num_nodes, hidden_dim = h.shape
        
        batched_edge_index = self.make_batched_edge_index(
            edge_index,
            batch_size,
            num_nodes)
        
        h_flat = h.reshape(
            batch_size * num_nodes,
            hidden_dim)
        
        h_flat = self.tag_conv(
            h_flat,
            batched_edge_index)
        
        return h_flat.reshape(
            batch_size,
            num_nodes,
            hidden_dim)
    
class PowerFlowConv(nn.Module):
    """
    One PowerFlowConv Layer
    PowerFlowNet combines two graph operations:
        1. Message network: Edge-aware one-hop message passing.
        2. Update network: Residual addition followed by K-hop TAGConv.
        
    Final PowerFlowNet layer is handled separately by the readout network.
    Final layer does not use TAGConv.
    """
    
    def __init__(
        self,
        hidden_dim,
        edge_dim,
        K,
        dropout_rate):
        super().__init__()
        
        # Message Network
        self.message_network = PowerFlowMessageNetwork(
            nfeature_dim=hidden_dim,
            efeature_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim)
        
        # Sum incoming edge messages at each node
        self.message_aggregation = PowerFlowMessagePassing(output_dim=hidden_dim)
        
        # Update network
        self.update_network = PowerFlowUpdateNetwork(
            hidden_dim=hidden_dim,
            K=K)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, h, edge_index, edge_attr):
        """
        Perform one PowerFlowConv operation.
        1. Compute edge-aware messages.
        2. Aggregate incoming messages.
        3. Add the message to the current node state.
        4. Apply the K-hop TAGConv.
        5. Apply ReLU and dropout.
        """
        
        # 1. Message Network
        messages = self.message_network(
            h,
            edge_index,
            edge_attr)
        
        # 2. Message Aggregation
        aggregated_messages = self.message_aggregation(
            messages,
            edge_index,
            num_nodes=h.size(1))
        
        # 3. Residual Addition
        h = h + aggregated_messages
        
        # 4. Update Network: K-hop TAGConv
        h = self.update_network(
            h,
            edge_index)
        
        # 5. Apply ReLU and dropout
        h = torch.relu(h)
        h = self.dropout(h)
        return h

# PowerFlowNet does not have a seperate Readout Neural Network
# Instead it's just another esge-aware message passing layer.
class PowerFlowReadoutNetwork(nn.Module):
    """
    Readout Network
    Final PowerFlowNet layer has no TAGConv after final message-passing operation.
    Produces [P, V, Q, Theta]
    """
    
    def __init__(
            self,
            hidden_dim,
            edge_dim,
            output_dim):
        super().__init__()
        
        # Readout / Final Message Network
        self.message_network = PowerFlowMessageNetwork(
            nfeature_dim = hidden_dim,
            efeature_dim = edge_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim)
        
        self.message_aggregation = PowerFlowMessagePassing(output_dim=output_dim)
        
    def forward(self, h, edge_index, edge_attr):
        """
        Produce the final node-level [P, V, Q, Theta] prediction.
        """
        
        messages = self.message_network(
            h,
            edge_index,
            edge_attr)
        
        output = self.message_aggregation(
            messages,
            edge_index,
            num_nodes = h.size(1))
        
        return output
    
class PowerFlowMaskEncoder(nn.Module):
    """
    Mask Encoder
    Maps the four binary feature masks to learned four-dimensional embedding:
        mask -> Linear -> ReLU -> Linear -> mask embedding
    The embedding is added to the observed / zero-filled node features before the input projection.
    """
    
    def __init__(
        self,
        feature_dim,
        hidden_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim))
        
    def forward(self, mask):
        return self.network(mask)
        
class PowerFlowGNN(nn.Module):
    """
    Architecture:
        [P, V, Q, Theta] + mask ->
        Mask encoder ->
        Input projection ->
        PowerFlowConv layer 1:
            -- Message network
            -- Message aggregation
            -- Residual addition
            -- Update network (TAGConv)
            -- ReLU + Dropout
        -> ... ->
        PowerFlowConv layer L-1 ->
        Final readout network:
            -- Message network
            -- Message aggregation
        -> Final output : [P, V, Q, Theta]
    """
    
    def __init__(
        self, 
        in_dim=8,
        hidden_dim=128,
        out_dim=4,
        edge_dim=2,
        num_layers=4,
        K=3,
        dropout_rate=0.2):
        super().__init__()
        
        self.in_dim = in_dim
        self.feature_dim = 4
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.K = K
        self.dropout_rate = dropout_rate
        
        # Mask Encoder
        self.mask_encoder = PowerFlowMaskEncoder(
            feature_dim=self.feature_dim,
            hidden_dim = hidden_dim)
        
        # Input Projection
        self.input_proj = nn.Linear(
            self.feature_dim,
            hidden_dim)
        
        # L-1 Intermediate PowerFlowConv Layers
        self.message_passing_layers = nn.ModuleList([
            PowerFlowConv(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                K=K,
                dropout_rate=dropout_rate)
            for _ in range(num_layers -1)])
        
        # Final Readout
        self.readout = PowerFlowReadoutNetwork(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            output_dim=out_dim)
        
    @staticmethod
    def ensure_bidirectional_graph(edge_index, edge_attr):
        """
        Make sure every physical branch has both directed edges.
        The project's dataset generator already stores both directions.
        If a dataset contains only one direction, add the reverse edge.
        """
        
        if edge_index.numel() == 0:
            return edge_index, edge_attr
        
        source = edge_index[0]
        destination = edge_index[1]
        
        # Check whether every edge has a corresponding reverse edge.
        edge_pairs = set(zip(
            source.detach().cpu().tolist(),
            destination.detach().cpu().tolist()))
        
        already_bidirectional = all(
            (dst, src) in edge_pairs
            for src, dst in edge_pairs)

        if already_bidirectional:
            return edge_index, edge_attr

        reverse_edge_index = torch.stack(
            [edge_index[1], edge_index[0]],
            dim=0)

        edge_index = torch.cat(
            [edge_index, reverse_edge_index],
            dim=1)

        edge_attr = torch.cat(
            [edge_attr, edge_attr],
            dim=0)

        return edge_index, edge_attr
    
    def forward(self, x, edge_index, edge_attr):
        """
        Perform a PowerFlowNet-style forward pass.
        Parameters:
            x : size([batch, num_nodes, 8]) : [P, V, Q, Theta, mP, mV, mQ, mTheta]
            edge_index : size([2, num_edges]) : Directed graph connectivity [from_node, to_node]
            edge_attr : size([2, num_edges]) : [r, x]
        Returns:
            output : size([batch, num_nodes, 4]) : [P, V, Q, Theta]
        """
        
        # 1. Separate features and mask
        node_features = x[:, :, :self.feature_dim]
        mask = x[:, :, self.feature_dim:]
        
        # 2. Mask encoder
        mask_embedding = self.mask_encoder(mask)
        x = node_features + mask_embedding
        
        # 3. Input projection
        h = self.input_proj(x)
        
        # Ensure graph is bidirectional before message passing.
        edge_index, edge_attr = self.ensure_bidirectional_graph(
            edge_index,
            edge_attr)
        
        # PowerFlowNet uses only resistance and reactance
        edge_attr = edge_attr[:, :self.edge_dim]
        
        # 4. PowerFlowConv Stack
        for layer in self.message_passing_layers:
            h = layer(
                h,
                edge_index,
                edge_attr)
        
        # 5. Final readout
        output = self.readout(
            h,
            edge_index,
            edge_attr)
        return output
    
def compute_loss(
        pred,
        target,
        mask,
        loss_weights,
        unknown_only=False):
    """
    Compute the training loss.
    Train with ordinary MSE over the complete output graph.
    """
    
    if unknown_only:
        # Mask = 1 for known values.
        unknown = 1.0 - mask
        loss = ((pred-target)**2 * unknown * loss_weights)
        denominator = (unknown * loss_weights).sum()
        
    else:
        loss = ((pred-target)**2 * loss_weights)
        denominator = (torch.ones_like(target) * loss_weights).sum()
        
    return loss.sum() / denominator
    
def train_model(
        model,
        train_dataset,
        save_filepath,
        val_dataset=None,
        epochs=1000,
        batch_size=128,
        lr=1e-3,
        device='cpu',
        loss_weights=(1, 1, 1, 1),
        early_stopping=False,
        patience=20,
        unknown_only=False,
        use_one_cycle=True):
    
    model = model.to(device)
    
    # PowerFlowNet uses AdamW with lr = 0.001
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr)
    
    # Training data
    X_train = torch.tensor(
        train_dataset['X'],
        dtype=torch.float32)
    
    Y_train = torch.tensor(
        train_dataset['Y'],
        dtype=torch.float32)
    
    train_mask = X_train[:, :, 4:8]
    
    # Validation data
    if val_dataset is not None:
        X_val = torch.tensor(
            val_dataset['X'],
            dtype=torch.float32)
        Y_val = torch.tensor(
            val_dataset['Y'],
            dtype=torch.float32)
        val_mask = X_val[:, :, 4:8]
        
    # Graph data
    edge_index = torch.tensor(train_dataset['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(train_dataset['edge_attr'], dtype=torch.float32).to(device)
    
    # Loss weights
    weights = torch.tensor(
        loss_weights,
        dtype=torch.float32,
        device=device).view(1, 1, 4)
    
    # PowerFlowNet uses OneCyleLR during training.
    if use_one_cycle:
        steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch)
        
    loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    for epoch in range(epochs):
        
        # Training 
        model.train()
        
        indices = torch.randperm(len(X_train))
        epoch_loss = 0.0
        num_batches = 0
        
        for start in tqdm(
            range(0, len(X_train), batch_size),
            desc=f'Epoch {epoch + 1}/{epochs}'):
            
            idx = indices[start:start + batch_size]
            
            x = X_train[idx].to(device)
            y = Y_train[idx].to(device)
            mask = train_mask[idx].to(device)
            
            optimizer.zero_grad()
            
            pred = model(
                x,
                edge_index,
                edge_attr)
            
            loss = compute_loss(
                pred,
                y,
                mask,
                weights,
                unknown_only=unknown_only)
            
            loss.backward()
            optimizer.step()
            
            if use_one_cycle:
                scheduler.step()
                
            epoch_loss += loss.item()
            num_batches += 1
            
        epoch_loss /= num_batches
        loss_history.append(epoch_loss)
        
        # Validation
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(
                    X_val.to(device),
                    edge_index,
                    edge_attr)
                val_loss = compute_loss(
                    val_pred,
                    Y_val.to(device),
                    val_mask.to(device),
                    weights,
                    unknown_only=unknown_only)
                
            val_loss = val_loss.item()
            val_loss_history.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(
                    model.state_dict())
                patience_counter = 0
                
            else: patience_counter += 1
            
            if early_stopping and patience_counter >= patience: break
        
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_filepath)
    
    if val_dataset is not None:
        return loss_history, val_loss_history
    return loss_history
        
# Unit Conversion
def convert_to_per_unit(data, bases):
    S_base = bases['S_base']
    V_base = bases['V_base']

    data = data.copy()

    data[..., 0] /= S_base
    data[..., 1] /= V_base
    data[..., 2] /= S_base

    return data


def convert_to_absolute(data, bases):
    S_base = bases['S_base']
    V_base = bases['V_base']

    data = data.copy()

    data[..., 0] *= S_base
    data[..., 1] *= V_base
    data[..., 2] *= S_base

    return data

# Prediction

def predict(
    model,
    dataset,
    batch_size=32,
    device='cpu',
    loss_weights=[1, 1, 1, 1]):
    
    model = model.to(device).eval()
    
    X = torch.tensor(
        dataset['X'],
        dtype=torch.float32)
    
    Y = torch.tensor(
        dataset['Y'],
        dtype=torch.float32)
    
    edge_index = torch.tensor(
        dataset['edge_index'],
        dtype=torch.long).to(device)
    
    edge_attr = torch.tensor(
        dataset['edge_attr'],
        dtype=torch.float32).to(device)
    
    masks = X[:, :, 4:8]
    
    weights = torch.tensor(
        loss_weights,
        dtype=torch.float32,
        device=device).view(1, 1, 4)
    
    preds = []
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        
        for i in range(0, len(X), batch_size):
            
            x = X[i:i + batch_size].to(device)
            y = Y[i:i + batch_size].to(device)
            mask = masks[i:i + batch_size].to(device)
            
            pred = model(
                x, 
                edge_index,
                edge_attr)
            
            preds.append(pred.cpu().numpy())
            
            # Testing loss: unknown variables only
            unknown = (1.0 - mask).float()
            
            loss = ((pred - y)**2 * unknown * weights).sum()
            loss /= (unknown * weights).sum()
            
            total_loss += loss.item()
            total_batches += 1
            
    testing_loss = total_loss / total_batches
    
    return (
        np.concatenate(preds),
        Y.numpy(),
        testing_loss)

# Metrics
def compute_metrics(preds, targets, mask):
    
    var_names = np.array(['P', 'V', 'Q', 'Theta'])
    mask = ~ mask
    metrics = {}
    
    for i, name in enumerate(var_names):
        
        pred = preds[:, :, i][mask[:, :, i]]
        target = targets[:, :, i][mask[:, :, i]]
        
        error = pred - target
        bias = np.mean(error)
        mse = np.mean(error ** 2)
        
        denom = np.sum((target - target.mean()) ** 2)
        if denom > 0: R2 = (1 - np.sum(error ** 2) / denom)
        else: R2 = np.nan
        
        metrics[name] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'bias': bias,
            'R2': R2}
        
    return metrics

def compute_bus_metrics(preds, targets, mask):
    
    var_names = ['P', 'V', 'Q', 'Theta']
    mask = ~ mask
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
            
            if denom > 0: R2 = (1 - np.sum(error ** 2) / denom)
            else: R2 = np.nan
            
            metrics[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'bias': bias,
                'R2': R2}
            
        bus_metrics[bus] = metrics
    return bus_metrics


def print_metrics(metrics):

    print(
        f"{'Variable':<10} "
        f"{'MSE':>12} "
        f"{'RMSE':>12} "
        f"{'Bias':>12} "
        f"{'R²':>12}")

    print("-" * 60)

    for name, values in metrics.items():

        print(
            f"{name:<10} "
            f"{values['mse']:>12.6f} "
            f"{values['rmse']:>12.6f} "
            f"{values['bias']:>12.6f} "
            f"{values['R2']:>12.6f}")

# Plotting
def plot_loss_history(loss_history):

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
def plot_bus_metrics(bus_metrics):

    variables = [
        'P',
        'V',
        'Q',
        'Theta']

    metrics = [
        'mse',
        'rmse',
        'bias',
        'R2']

    for variable in variables:

        for metric in metrics:

            valid_buses = [
                bus
                for bus in sorted(bus_metrics)
                if variable in bus_metrics[bus]
                and metric in bus_metrics[bus][variable]]

            values = [
                bus_metrics[bus][variable][metric]
                for bus in valid_buses]

            plt.figure()
            plt.bar(valid_buses, values)
            plt.xlabel('Bus Index')
            plt.ylabel(metric.upper())
            plt.title(f'{variable} {metric.upper()} by Bus')
            plt.xticks(valid_buses)
            plt.grid(axis='y')
            plt.show()

# Main
if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    
    train_data_filepath = os.path.join(data_dir, 'case14_20000sample_train.npy')
    val_data_filepath = os.path.join(data_dir, 'case14_20000sample_val.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_20000sample_test.npy')
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_filepath = os.path.join(models_dir, 'powerflownet_model.pt')
    
    results_dir = os.path.join(base_dir, 'results')
    results_folderpath = os.path.join(results_dir, '100sample')
    os.makedirs(results_folderpath, exist_ok=True)
    results_filepath = os.path.join(results_folderpath, 'case14_results.npy')
    
    train_loss_history_filepath = os.path.join(results_folderpath, 'train_loss_history.npy')
    val_loss_history_filepath = os.path.join(results_folderpath, 'val_loss_history.npy')
    
    train_data = np.load(train_data_filepath, allow_pickle=True).item()
    val_data = np.load(val_data_filepath, allow_pickle=True).item()
    test_data = np.load(test_data_filepath, allow_pickle=True).item()
    
    # Model and PowerFlowNet Configuration
    hidden_dim = 128
    num_layers = 4
    K = 3 
    dropout_rate = 0.2
    
    model = PowerFlowGNN(
        in_dim=8,
        hidden_dim=hidden_dim,
        out_dim=4,
        edge_dim=2,
        num_layers=num_layers,
        K=K,
        dropout_rate=dropout_rate)
    
    # PowerFlowNet standard training settings:
    # AdamW, lr=0.001, OneCycleLR, batch size=128, MSE loss.
    epochs = 500
    batch_size = 128
    lr = 1e-3
    loss_weights = [1, 1, 1, 1]
    
    # PowerFlowNet Paper MSE training objective uses Knowns
    unknown_only = False
    
    train = True
    if train:
        train_loss_history, val_loss_history = train_model(
            model,
            train_data,
            model_filepath,
            val_dataset=val_data,
            early_stopping=True,
            patience=20,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device='cpu',
            loss_weights=loss_weights,
            unknown_only=unknown_only,
            use_one_cycle=True)
        
        np.save(
            train_loss_history_filepath,
            train_loss_history)
        
        np.save(
            val_loss_history_filepath,
            val_loss_history)
        
    # Load trained model
    model.load_state_dict(
        torch.load(
            model_filepath,
            weights_only=True))
    
    train_loss_history = np.load(
        train_loss_history_filepath)
    
    val_loss_history = np.load(
        val_loss_history_filepath)
    
    plot_loss_history(
        train_loss_history)
    
    plot_loss_history(
        val_loss_history)
    
    # Test Model
    mask = test_data['X'][:, :, 4:8].astype(bool)
    bases = test_data['bases']
    
    preds_pu, targets_pu, testing_loss = predict(
        model,
        test_data, 
        batch_size=32,
        device='cpu')
    
    metrics_pu = compute_metrics(
        preds_pu,
        targets_pu,
        mask)
    
    bus_metrics_pu = compute_bus_metrics(
        preds_pu,
        targets_pu,
        mask)

    preds_absolute = convert_to_absolute(
        preds_pu,
        bases)

    targets_absolute = convert_to_absolute(
        targets_pu,
        bases)

    metrics_absolute = compute_metrics(
        preds_absolute,
        targets_absolute,
        mask)

    bus_metrics_absolute = compute_bus_metrics(
        preds_absolute,
        targets_absolute,
        mask)
    
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