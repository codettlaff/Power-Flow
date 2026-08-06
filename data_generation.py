# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:40:39 2026

@author: codett
"""

import copy
import numpy as np

import torch
import pandapower as pp
import pandapower.networks as pn
from torch_geometric.data import Data

def perturb_loads(net, global_scale=1.0, local_var=0.1):
    net = copy.deepcopy(net)
    scale = global_scale * np.random.uniform(
        1 - local_var,
        1 + local_var,
        len(net.load))
    net.load['p_mw'] *= scale
    net.load['q_mvar'] *= scale
    return net

def perturb_lines(net, r_var=0.1, x_var=0.1):
    net = copy.deepcopy(net)
    net.line['r_ohm_per_km'] *= np.random.uniform(
        1-r_var,
        1+r_var,
        len(net.line))
    net.line['x_ohm_per_km'] *= np.random.uniform(
        1-x_var,
        1+x_var,
        len(net.line))
    return net

def build_dataset(net):
    
    pp.runpp(net)
    
    # Build node features x = [P, V, Q, theta, mP, mV, mQ, mTheta]
    n = len(net.bus)
    x = torch.zeros((n,8), dtype=torch.float)
    
    # PQ buses
    for idx in net.load.bus:
        p = net.load.loc[net.load.bus == idx, 'p_mw'].sum()
        q = net.load.loc[net.load.bus == idx, 'q_mvar'].sum()
        x[idx] = torch.tensor([
            p,
            0.0,
            q,
            0.0,
            1,0,1,0])
        
    # PV buses
    for _, gen in net.gen.iterrows():
        x[int(gen.bus)] = torch.tensor([
            gen.p_mw,
            gen.vm_pu,
            0.0,
            0.0,
            1,1,0,0])
        
    # Slack bus
    for _, ext in net.ext_grid.iterrows():
        x[int(ext.bus)] = torch.tensor([
            0.0,
            ext.vm_pu,
            0.0,
            ext.va_degree,
            0,1,0,1])
        
    # Edge List

    edge_index = torch.tensor([
        net.line.from_bus.values,
        net.line.to_bus.values,
        ], dtype=torch.long)

    edge_attr = torch.tensor(
        net.line[['r_ohm_per_km', 'x_ohm_per_km']].values,
        dtype=torch.float)

    # Make graph undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    # Targets
    # y = [V, theta, P, Q]

    res = net.res_bus
    y = torch.tensor(
        res[['vm_pu', 'va_degree', 'p_mw', 'q_mvar']].values,
        dtype=torch.float)

    # PyTorch Geometric Object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y)

    return data
    
net = pn.case14() # IEEE 14-bus transmission system
net2 = perturb_loads(net)
net3 = perturb_lines(net)
dataset = build_dataset(net3)

print('')
# Generating multiple data points
# Randomly scale the loads while keeping network topology fixed.
# Want 5,000-10,000 samples for case 14.
