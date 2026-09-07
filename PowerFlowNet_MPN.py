# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:53:03 2026
Author: Casey Dettlaff
"""

import torch
import torch.nn as nn
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing, TAGConv

class EdgeAggregation(MessagePassing):
    """
    Edge-aware one-hop message-passing block used by PowerFlowNet
    For each directed edge (i,j), the message MLP receives: [x_i, x_j, e_ij]
    Messages are summed at the destination node.
    This matches the message-passing component used by the PowerFlowNet repository implementation.
    """
    
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.edge_aggr = nn.Sequential(
            nn.Linear(2 * nfeature_dim + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        
    def message(self, x_i, x_j, edge_attr):
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1))
    
    def forward(self, x, edge_index, edge_attr):
        
        row, col = edge_index
        deg = degree(
            col,
            x.size(0),
            dtype=x.dtype)
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = self.propagate(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            norm=norm)
        
        return out
    
    def forward_old(self, x, edge_index, edge_attr):
        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr)
    
class PowerFlowNetMPN(nn.Module):
    """
    PowerFlowNet architecture used for the replication experiment.
    Architecture:
        node features + learned mask embedding ->
        PowerFlowConv layer 1 (edge-aware MP -> TAGConv) ->
        PowerFlowConv layer 2 (edge-aware MP -> TAGConv) ->
        PowerFlowConv layer 3 (edge-aware MP -> TAGConv) ->
        PowerFlowConv layer 4 (edge-aware MP only) ->
        output
        
    The paper's standard configuration is L=4 and K=3.
    The repository implementation exposes the model as MaskEmbdMultiMPN.
    This class makes that architecture the primary MPN model used by train.py.
    """
    
    def __init__(
        self,
        nfeature_dim,
        efeature_dim,
        output_dim,
        hidden_dim,
        n_gnn_layers,
        K,
        dropout_rate):
        
        super().__init__()
        
        self.input_proj = nn.Linear(nfeature_dim, hidden_dim)
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        
        # The PowerFlowNet paper uses a learned two-layer mask encoder.
        # This maps the F-dimensional binary mask back to F dimensions.
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim))
        
        # Each intermediate PowerFlowConv consists of:
        # 1. Edge-aware message-passing
        # 2. Redisdual addition
        # 3. K-hop TAGConv
        # The final layer contains message-passing / readout operation, without TAGConv.
        self.message_layers = nn.ModuleList()
        self.tag_convs = nn.ModuleList()
        
        if n_gnn_layers == 1:
            self.message_layers.append(
                EdgeAggregation(
                    nfeature_dim=hidden_dim,
                    efeature_dim=2,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim))
        else: 
            # L-1 intermediate message-passing blocks.
            for _ in range(n_gnn_layers - 1):
                self.message_layers.append(
                    EdgeAggregation(
                        nfeature_dim=hidden_dim,
                        efeature_dim=2,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim))
                
                self.tag_convs.append(
                    TAGConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        K=K))

            # Final message-passing/readout layer; no TAGConv follows it.
            self.message_layers.append(
                EdgeAggregation(
                    nfeature_dim=hidden_dim,
                    efeature_dim=2,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim))
            
        self.dropout = nn.Dropout(dropout_rate)
        
    @property
    def layers(self):
        return self.message_layers
        
    @staticmethod
    def is_directed(edge_index):
        """
        Detect whether edge_index contains only one direction of a branch.
        The dataset normally already stores both directions.
        If it does not, the reverse edges are added before message passing.
        """
        if edge_index.numel() == 0: return False
        src = edge_index[0, 0]
        dst = edge_index[1, 0]
        return src not in edge_index[1, edge_index[0] == dst]
    
    @classmethod 
    def undirect_graph(cls, edge_index, edge_attr):
        if cls.is_directed(edge_index):
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
        
    def forward(self, data):
        
        # Project-specific data layout:
        # columns 0 : 4 = one-hot bus type.
        # columns 4 : 4 + nfeature_dim = node features
        # columns 4 + nfeature_dim : = mask
        x = data.x[:, 4:4 + self.nfeature_dim]
        mask = data.x[:, -self.nfeature_dim:]
        
        # Mask encoder:
        # X^0_i = x_i + mask_embedding(m_i)
        x = x + self.mask_embd(mask)
        x = self.input_proj(x)
        edge_index, edge_attr = self.undirect_graph(data.edge_index, data.edge_attr)
        
        # Project's edge representation contains five processed attributes.
        # PowerFlowNet only uses resistance / reactance.
        # Current project keeps those as the first two columns.
        edge_attr = edge_attr[:, :2]
        
        # n_gnn_layers == 1 is retained for paper's ablation case
        if self.n_gnn_layers == 1:
            return self.layers[0](
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr)
        
        # Intermediate PowerFlowConv blocks.
        layer_index = 0
        
        for i in range(self.n_gnn_layers - 1):
            
            # One-hop edge-aware message passing
            message = self.message_layers[i](
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr)
            
            # Residual addition described in Eq. (10)-(11).
            x = x + message
            x = self.tag_convs[i](x, edge_index)
            
            # ReLU + droput at the end of the intermediate layer.
            x = torch.relu(x)
            x = self.dropout(x)
            layer_index += 2
            
        # Final PowerFlowConv: message passing only, no TAGConv
        return self.message_layers[-1](
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr)