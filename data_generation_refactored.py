# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 08:49:20 2026

@author: Casey
"""

import os
import copy
import numpy as np
from tqdm import tqdm
import random

import pandapower as pp
import pandapower.networks as pn

def perturb_loads(net, global_scale=1.0, local_var=0.1):
    net = copy.deepcopy(net)
    scale = global_scale * np.random.uniform(
        1 - local_var, 1 + local_var, len(net.load))
    net.load['p_mw'] *= scale
    net.load['q_mvar'] *= scale
    return net

def perturb_lines(net, global_scale=1.0, r_var=0.1, x_var=0.1):
    net = copy.deepcopy(net)
    net.line['r_ohm_per_km'] *= global_scale * np.random.uniform(
        1 - r_var, 1 + r_var, len(net.line))
    net.line['x_ohm_per_km'] *= global_scale * np.random.uniform(
        1 - x_var, 1 + x_var, len(net.line))
    return net

def create_sample(net):
    pp.runpp(net)
    S_base = net.sn_mva
    n = len(net.bus)
    
    # Node features: [P, V, Q, theta, mP, mV, mQ, mTheta]
    X = np.zeros((n, 8), dtype=np.float32)
    
    # PQ Buses
    for idx in net.load.bus:
        p = net.load.loc[net.load.bus == idx, 'p_mw'].sum() / S_base
        q = net.load.loc[net.load.bus == idx, 'q_mvar'].sum() / S_base
        X[idx] = [
            p,
            0,
            q,
            0, 
            1, 0, 1, 0]
        
    # PV Buses
    for _, gen in net.gen.iterrows():
        X[int(gen.bus)] = [
            gen.p_mw / S_base,
            gen.vm_pu,
            0,
            0,
            1, 1, 0, 0]
        
    # Slack Bus
    for _, ext in net.ext_grid.iterrows():
        X[int(ext.bus)] = [
            0,
            ext.vm_pu,
            0,
            np.deg2rad(ext.va_degree),
            0, 1, 0, 1]
        
    # edge_index: defines which buses are connected, shape [2, numer_of_edges].
    # each column is a connection [from_bus, to_bus]
    # duplicate / reverse the edges so that graph is treates as undirected.
    edge_index = np.array([
        net.line.from_bus.values,
        net.line.to_bus.values], dtype=np.int64)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    
    edge_attr = net.line[['r_ohm_per_km', 'x_ohm_per_km']].values.astype(np.float32)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)
    
    # Targets: [V, theta, P, Q]
    Y = np.column_stack([
        net.res_bus.vm_pu.values,
        np.deg2rad(net.res_bus.va_degree.values),
        net.res_bus.p_mw.values / S_base,
        net.res_bus.q_mvar.values / S_base]).astype(np.float32)
    
    return X, Y, edge_index, edge_attr

def generate_dataset(
        net, n_samples, vary_loads=True, vary_lines=True,
        global_scale=(0.7, 1.3), local_var=0.1, r_var=0.1, x_var=0.1):
    
    X, Y, edge_attrs = [], [], []
    for _ in tqdm(range(n_samples), desc='Generating samples'):
        net_i = copy.deepcopy(net)
        
        if vary_loads: net_i = perturb_loads(net_i, np.random.uniform(*global_scale), local_var)
        if vary_lines: net_i = perturb_lines(net_i, r_var, x_var)
        
        x, y, _, edge_attr = create_sample(net_i)
        X.append(x)
        Y.append(y)
        edge_attrs.append(edge_attr)
        
    return {
        'X': np.array(X),
        'Y': np.array(Y),
        'X_labels': np.array(['P', 'V', 'Q', 'theta', 'mP', 'mV', 'mQ', 'mTheta']),
        'Y_labels': np.array(['V', 'theta', 'P', 'Q']),
        'edge_index': create_sample(net)[2],
        'edge_attr': np.array(edge_attrs)}

def train_test_split(dataset, train_ratio=0.8):
    
    n = len(dataset['X'])
    indices = list(range(n))
    random.shuffle(indices)
    split = int(train_ratio * n)
    
    train = {k: v[indices[:split]] if k in ['X', 'Y', 'edge_attr'] else v for k, v in dataset.items()}
    test = {k: v[indices[split:]] if k in ['X', 'Y', 'edge_attr'] else v for k, v in dataset.items()}
    
    return train, test

if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    train_data_filepath = os.path.join(data_dir, 'case14_train.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_test.npy')
    
    n_samples = 5e3
    
    net = pn.case14()
    dataset = generate_dataset(net, n_samples)
    train_data, test_data = train_test_split(dataset)
    
    np.save(train_data_filepath, train_data, allow_pickle=True)
    np.save(test_data_filepath, test_data, allow_pickle=True)
    