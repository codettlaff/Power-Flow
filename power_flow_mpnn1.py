# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:54:48 2026

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
        edge_features = edge_attr.unsqueeze()
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
            