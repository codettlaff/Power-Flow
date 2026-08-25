# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:56:28 2026

@author: Casey
"""

import os
import copy
import random
import numpy as np
from tqdm import tqdm

import pandapower as pp
import pandapower.networks as pn

def get_system_bases(net):
    S_base = float(net.sn_mva)
    V_base = net.bus.vn_kv.values.astype(np.float32)
    z_base = (V_base[net.line.from_bus.values] ** 2 / S_base).astype(np.float32)
    return S_base, V_base, z_base

def scale_loads(net, scale_factor):
    s = np.random.uniform(*scale_factor, len(net.load))
    net.load[['p_mw', 'q_mvar']] *= s[:, None]

def scale_generator_powers(net, scale_factor):
    s = np.random.uniform(*scale_factor, len(net.gen))
    net.gen['p_mw'] *= s

def scale_generator_voltages(net, scale_factor):
    s = np.random.uniform(*scale_factor, len(net.gen))
    net.gen['vm_pu'] *= s
    
def get_edge_data(net):
    edge_index = np.array([net.line.from_bus, net.line.to_bus], dtype=np.int64)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)

    # Convert Ω/km → Ω → p.u.
    z_ohm = net.line[['r_ohm_per_km', 'x_ohm_per_km']].values * net.line.length_km.values[:, None]
    z_base = net.bus.vn_kv.values[net.line.from_bus.values] ** 2 / net.sn_mva
    edge_attr = (z_ohm / z_base[:, None]).astype(np.float32)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)

    return edge_index, edge_attr

def create_sample(
        net,
        perturb_loads=False,
        load_scale_factor=(0.7,1.3),
        perturb_generator_powers=True,
        gen_powers_scale_factor=(0.7,1.3),
        perturb_generator_voltages=True,
        gen_voltages_scale_factor=(0.98,1.02)):
    
    net = copy.deepcopy(net)
    S_base = net.sn_mva
    
    if perturb_loads: scale_loads(net, load_scale_factor)
    if perturb_generator_powers: scale_generator_powers(net, gen_powers_scale_factor)
    if perturb_generator_voltages: scale_generator_voltages(net, gen_voltages_scale_factor)
    
    pp.runpp(net)
    
    n = len(net.bus)
    X = np.zeros((n, 8), dtype=np.float32)
    
    slack = int(net.ext_grid.iloc[0].bus)
    pv = set(net.gen.bus.astype(int))
    pq = set(range(n)) - pv - {slack}
    
    # PQ buses, including zero-injection buses
    for bus in pq:
        load = net.load[net.load.bus == bus]
        P = load.p_mw.sum() / S_base
        Q = load.q_mvar.sum() / S_base
        X[bus] = [P, 0, Q, 0, 1, 0, 1, 0]
        
    # PV buses
    for bus in pv:
        gen = net.gen[net.gen.bus == bus]
        X[bus] = [
            gen.p_mw.sum() / S_base,
            gen.vm_pu.iloc[0],
            0, 0,
            1, 1, 0, 0]
        
    # Slack Bus
    ext = net.ext_grid.iloc[0]
    X[slack] = [
        0, ext.vm_pu, 0, np.deg2rad(ext.va_degree),
        0, 1, 0, 1]
    
    edge_index, edge_attr = get_edge_data(net)
    
    Y = np.column_stack([
        net.res_bus.p_mw.values / S_base,
        net.res_bus.vm_pu.values,
        net.res_bus.q_mvar.values / S_base,
        np.deg2rad(net.res_bus.va_degree.values)]).astype(np.float32)
    
    return X, Y

def create_dataset(
        net,
        n_samples,
        perturb_loads=True,
        load_scale_factor=(0.7, 1.3),
        perturb_generator_powers=True,
        gen_powers_scale_factor=(0.7,1.3),
        perturb_generator_voltages=True,
        gen_voltages_scale_factor=(0.98,1.02)):
    
    edge_index, edge_attr = get_edge_data(net)
    X, Y = [], []
    
    for _ in tqdm(range(n_samples), desc='Generating samples'):
        x, y = create_sample(
            net,
            perturb_loads,
            load_scale_factor,
            perturb_generator_powers,
            gen_powers_scale_factor,
            perturb_generator_voltages,
            gen_voltages_scale_factor)
        X.append(x)
        Y.append(y)

    return {
        'X': np.array(X),
        'Y': np.array(Y),
        'X_labels': np.array(['P', 'V', 'Q', 'Theta', 'mP', 'mV', 'mQ', 'mTheta']),
        'Y_labels': np.array(['P', 'V', 'Q', 'Theta']),
        'edge_index': edge_index,
        'edge_attr': edge_attr}

def train_test_split(dataset, train_ratio=0.8):
    n = len(dataset['X'])
    indices = list(range(n))
    random.shuffle(indices)
    split = int(train_ratio * n)
    
    train = {k: v[indices[:split]] if k in ['X', 'Y'] else v for k, v in dataset.items()}
    test = {k: v[indices[split:]] if k in ['X', 'Y'] else v for k, v in dataset.items()}
    
    return train, test

if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    train_data_filepath = os.path.join(data_dir, 'case14_train1.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_test1.npy')
    
    # n_samples = int(5e4)
    n_samples = 100 # Test
    load_scale_factor = (0.7, 1.3)
    gen_power_scale_factor = (0.7, 1.3)
    gen_voltage_scale_factor = (0.98, 1.02)
    
    net = pn.case14()
    dataset = create_dataset(
        net,
        n_samples,
        perturb_loads=True,
        load_scale_factor=load_scale_factor,
        perturb_generator_powers=True,
        gen_powers_scale_factor=gen_power_scale_factor,
        perturb_generator_voltages=True,
        gen_voltages_scale_factor=gen_voltage_scale_factor)
        
    train_data, test_data = train_test_split(dataset)
    
    np.save(train_data_filepath, train_data, allow_pickle=True)
    np.save(test_data_filepath, test_data, allow_pickle=True)
        