# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:40:39 2026

@author: codett
"""

import copy
import numpy as np
from tqdm import tqdm
import random

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

def create_sample(net):
    
    pp.runpp(net)
    S_base = net.sn_mva
    
    # Build node features x = [P, V, Q, theta, mP, mV, mQ, mTheta]
    n = len(net.bus)
    x = torch.zeros((n,8), dtype=torch.float)
    
    # PQ buses
    for idx in net.load.bus:
        p = net.load.loc[net.load.bus == idx, 'p_mw'].sum() / S_base
        q = net.load.loc[net.load.bus == idx, 'q_mvar'].sum() / S_base
        x[idx] = torch.tensor([
            p,
            0.0,
            q,
            0.0,
            1,0,1,0])
        
    # PV buses
    for _, gen in net.gen.iterrows():
        x[int(gen.bus)] = torch.tensor([
            gen.p_mw / S_base,
            gen.vm_pu,
            0.0,
            0.0,
            1, 1, 0, 0])
        
    # Slack bus
    for _, ext in net.ext_grid.iterrows():
        x[int(ext.bus)] = torch.tensor([
            0.0,
            ext.vm_pu,
            0.0,
            np.deg2rad(ext.va_degree),
            0, 1, 0, 1])
        
    # Edge list
    edge_index = torch.tensor([
        net.line.from_bus.values,
        net.line.to_bus.values,
    ], dtype=torch.long)
    
    edge_attr = torch.tensor(
        net.line[['r_ohm_per_km', 'x_ohm_per_km']].values,
        dtype=torch.float)
    
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    
    # Targets: [V, theta, P, Q]
    y = torch.tensor(
        np.column_stack([
            net.res_bus.vm_pu.values,
            np.deg2rad(net.res_bus.va_degree.values),
            net.res_bus.p_mw.values / S_base,
            net.res_bus.q_mvar.values / S_base]),
        dtype=torch.float)
    
    # Targets: [V, theta, P, Q]
    y = torch.tensor(
        np.column_stack([
            net.res_bus.vm_pu.values,
            np.deg2rad(net.res_bus.va_degree.values),
            net.res_bus.p_mw.values / S_base,
            net.res_bus.q_mvar.values / S_base
        ]),
        dtype=torch.float)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y)
    
def generate_dataset(
    net,
    n_samples,
    vary_loads=True,
    vary_lines=True,
    global_scale=(0.7,1.3),
    local_var=0.1,
    r_var=0.1,
    x_var=0.1):

    dataset = []

    for _ in tqdm(range(n_samples), desc="Generating samples"):

        net_i = copy.deepcopy(net)

        if vary_loads:
            net_i = perturb_loads(
                net_i,
                global_scale=np.random.uniform(*global_scale),
                local_var=local_var)
        if vary_lines:
            net_i = perturb_lines(
                net_i,
                r_var=r_var,
                x_var=x_var)
        dataset.append(create_sample(net_i))
    return dataset

def train_test_split(dataset, train_ratio=0.8):
    
    dataset = dataset.copy()
    random.shuffle(dataset)
    split = int(train_ratio * len(dataset))
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]
    return train_dataset, test_dataset

net = pn.case14() # IEEE 14-bus transmission system
n_samples = int(5e3)
dataset = generate_dataset(net, n_samples) # Returns list of PyTorch Geometric Data Objects

dataset = torch.load('case14.pt', weights_only=False)

train_data, test_data = train_test_split(dataset)

torch.save(train_data, 'case14_train.pt')
torch.save(test_data, 'case14_test.pt')

print('')
