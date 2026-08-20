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

def sample_scale(distribution='uniform', scale_range=(0.7, 1.3)):
    if distribution == 'uniform':
        return np.random.uniform(*scale_range)
    elif distribution == 'laplace':
        center = np.mean(scale_range)
        spread = (scale_range[1] - scale_range[0]) / 2
        return center + np.random.laplace(0, spread / 2)
    elif distribution == 'extreme': # Favor lower and upper ends of the range
        center = np.mean(scale_range)
        spread = (scale_range[1] - scale_range[0]) / 2
        return center + spread * np.random.choice([-1,1]) * np.random.uniform(0.5,1.0)

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
    
    # Want one slack bus
    n_ext = len(net.ext_grid) # number of external grid connections.
    slack_bus = int(net.ext_grid.iloc[0].bus)
    
    pv_buses = set(net.gen.bus.astype(int))
    pq_buses = set(range(n)) - pv_buses - {slack_bus}
    
    # Node features: [P, V, Q, Theta, mP, mV, mQ, mTheta]
    X = np.zeros((n, 8), dtype=np.float32)
    
    # PQ Buses
    # Include both load buses and zero-injection buses.
    for idx in pq_buses:
        p = net.load.loc[net.load.bus == idx, 'p_mw'].sum() / S_base
        q = net.load.loc[net.load.bus == idx, 'q_mvar'].sum() / S_base
        X[idx] = [
            p, 0, q, 0,
            1, 0, 1, 0]
        
    # PV Buses
    # Generator buses are treated as PV even if they also have a load.
    for bus in pv_buses:
        gens = net.gen[net.gen.bus == bus]
        p = gens.p_mw.sum() / S_base
        vm_pu = gens.vm_pu.iloc[0] # All generators at a bus should have the same voltage setpoint.
        X[bus] = [
            p, vm_pu, 0, 0,
            1, 1, 0, 0]
        
    # Slack Bus
    ext = net.ext_grid.iloc[0]
    X[slack_bus] = [
        0, ext.vm_pu, 0, np.deg2rad(ext.va_degree),
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
    
    # Targets: [V, Theta, P, Q]
    Y = np.column_stack([
        net.res_bus.p_mw.values / S_base,
        net.res_bus.vm_pu.values,
        net.res_bus.q_mvar.values / S_base,
        np.deg2rad(net.res_bus.va_degree.values)]).astype(np.float32)
        
    return X, Y, edge_index, edge_attr

def generate_dataset(
        net, n_samples, vary_loads=True, vary_lines=True,
        distribution='uniform', scale_range=(0.7, 1.3),
        local_var=0.1, r_var=0.1, x_var=0.1):

    X, Y, edge_attrs = [], [], []

    for _ in tqdm(range(n_samples), desc='Generating samples'):
        net_i = copy.deepcopy(net)

        if vary_loads:
            net_i = perturb_loads(
                net_i, sample_scale(distribution, scale_range), local_var)

        if vary_lines:
            net_i = perturb_lines(
                net_i, sample_scale(distribution, scale_range),
                r_var, x_var)

        x, y, _, edge_attr = create_sample(net_i)
        X.append(x)
        Y.append(y)
        edge_attrs.append(edge_attr)

    return {
        'X': np.array(X),
        'Y': np.array(Y),
        'X_labels': np.array(
            ['P', 'V', 'Q', 'theta', 'mP', 'mV', 'mQ', 'mTheta']),
        'Y_labels': np.array(['P', 'V', 'Q', 'theta']),
        'edge_index': create_sample(net)[2],
        'edge_attr': np.array(edge_attrs)
    }

def generate_dataset_old(
        net, n_samples, vary_loads=True, vary_lines=True,
        global_scale=(0.7, 1.3), local_var=0.1, r_var=0.1, x_var=0.1):
    
    X, Y, edge_attrs = [], [], []
    for _ in tqdm(range(n_samples), desc='Generating samples'):
        net_i = copy.deepcopy(net)
        
        if vary_loads: net_i = perturb_loads(net_i, np.random.uniform(*global_scale), local_var)
        if vary_lines: net_i = perturb_lines(net_i, np.random.uniform(*global_scale), r_var, x_var)
        
        x, y, _, edge_attr = create_sample(net_i)
        X.append(x)
        Y.append(y)
        edge_attrs.append(edge_attr)
        
    return {
        'X': np.array(X),
        'Y': np.array(Y),
        'X_labels': np.array(['P', 'V', 'Q', 'theta', 'mP', 'mV', 'mQ', 'mTheta']),
        'Y_labels': np.array(['P', 'V', 'Q', 'theta']),
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
    os.makedirs(data_dir, exist_ok=True)
    
    train_data_filepath = os.path.join(data_dir, 'case14_train2.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_test2.npy')
    
    n_samples = int(5e3)
    scale_range = (0.1, 2)
    
    net = pn.case14()
    dataset = generate_dataset(net, n_samples)
    train_data, test_data = train_test_split(dataset)
    
    np.save(train_data_filepath, train_data, allow_pickle=True)
    np.save(test_data_filepath, test_data, allow_pickle=True)
    