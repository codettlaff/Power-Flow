# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:52:39 2026

@author: codett
"""

import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# x_i = [P,V,Q,Theta,mP,mV,mQ,mTheta]
# e_ij = [r, x]

# A. Node Encoder: x_i \in \mathbb{R}^8 -> h_i \in \mathbb{R}^128
# B. Message Network: [h_i, h_j, e_ij] -> m_ij
# C. Update Networ: [h_i, \sum_{ij \in N(i)} m_ij] -> h_i

# PyTorch Geometric's MessagePassing propogate() gives:
# for every edge (i,j): message[i,j] = message_network(h[i], h[j], edge[i,j])
# for every node i: aggregate[i] = sum(message[i,j] for j in N(i))
# for every node i: h_new[i] = update_network(h[i], aggregate[i])

class PowerFlowMPNN(MessagePassing):
    
    def __init__(self, hidden_dim=128, edge_dim=2):
        super().__init__(aggre='add')
        
        # Message Network
        # Input:
        #   h_i: receiving node hidden feature vector
        #   h_j: sending node hidden feature vector
        #   edge_ij: [r_ij, x_ij]
        # Input Dimension: 128 + 128 + 2 = 258
        # Output: Message vector of dimension 128
        
        self.message_network = nn.Sequential(
            nn.Linear(
                2 * hidden_dim + edge_dim,
                hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim))
        
        
        