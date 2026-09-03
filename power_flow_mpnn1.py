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
from torch.geometric.nn import TAGConv

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
            
            return self.netowork(message_input)
        
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
        
        aggregated = torch.zeros(
            messages.size(0),
            num_nodes,
            self.output_dim,
            device = messages.device,
            dtype = messages.dtype)
        
        aggregated.scatter_add_(
            dim=-1,
            index=destination
                .view(1, -1, 1)
                .expand_as(messages),
            src=messages)
        
        return aggregated
    
# TODO: This is still confusing.
# The update network should fulfill the role of using the aggregated messages to update the node state.
# Is that happeneing here?
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
        
        self.message_aggregation = PowerFlowMessagePassing(output = output_dim)
        
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
        
        def foward(self, mask):
            return self.network(mask)
        
# TODO: confused why the message network and message aggregation are a part of the readout network.
# TODO: shouldnt the readout network be a network not a layer?
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
            feature_dim=self.feature_sim,
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
        
            
                                                