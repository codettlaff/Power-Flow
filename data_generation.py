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
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.networks as pn

def inspect_net(net):
    if 'res_bus' not in net or net.res_bus.empty: pp.runpp(net, numba=False)
    
    print("=== SYSTEM ===")
    print(f"Base power:       {net.sn_mva:.1f} MVA")
    print(f"Total load:       {net.load.p_mw.sum():.2f} MW")
    print(f"Total generation: {net.res_gen.p_mw.sum():.2f} MW")
    print(f"System losses:    {net.res_line.pl_mw.sum():.2f} MW")
    
    print("\n=== BUS VOLTAGES ===")
    bus_results = net.res_bus[['vm_pu', 'va_degree']].copy()
    bus_results.index.name = 'bus'
    print(bus_results.to_string(float_format=lambda x: f'{x:.4f}'))
    
    print("\n=== LINE LOADING ===")
    line_results = net.res_line[['loading_percent', 'p_from_mw', 'q_from_mvar', 'pl_mw']].copy()
    line_results.index.name = 'line'
    print(line_results.to_string(float_format=lambda x: f'{x:.2f}'))
    print(
        f"\nMaximum line loading: "
        f"{net.res_line.loading_percent.max():.2f}%")
    
    if len(net.trafo):
        print("\n=== TRANSFORMER LOADING ===")
        print(
            net.res_trafo[['loading_percent', 'p_hv_mw', 'q_hv_mvar']]
            .to_string(float_format=lambda x: f'{x:.2f}'))
        print(
            f"\nMaximum transformer loading: "
            f"{net.res_trafo.loading_percent.max():.2f}%")
        
    print('\n')
        
    # Plots
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))

    # Voltage profile
    ax[0].plot(net.res_bus.vm_pu.values, marker='o')
    ax[0].axhline(1.0, linestyle='--')
    ax[0].set_ylabel('Voltage [p.u.]')
    ax[0].set_xlabel('Bus')
    ax[0].set_title('Bus Voltage Profile')
    ax[0].grid(True)

    # Line loading
    ax[1].bar(
        np.arange(len(net.line)),
        net.res_line.loading_percent.values)
    ax[1].axhline(100.0, linestyle='--')
    ax[1].set_ylabel('Loading [%]')
    ax[1].set_xlabel('Line')
    ax[1].set_title('Line Loading')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

def max_load_net(net, step, trials, save_filepath, v_limits=(0.95, 1.05)):
    net = copy.deepcopy(net)

    def test(net):
        try:
            pp.runpp(net, numba=False)
        except pp.LoadflowNotConverged:
            return True, True

        v = net.res_bus.vm_pu
        return v.max() > v_limits[1], v.min() < v_limits[0]

    def inc_load(net, scale):
        base = copy.deepcopy(net)
        for _ in range(trials):
            trial = copy.deepcopy(base)
            bus = np.random.choice(trial.load.index)
            trial.load.loc[bus, ['p_mw', 'q_mvar']] *= 1 + scale
            over, under = test(trial)
            if under:
                return base
            base = trial
        return base

    def inc_gen(net, scale):
        base = copy.deepcopy(net)
        for _ in range(trials):
            trial = copy.deepcopy(base)
            gen = np.random.choice(trial.gen.index)
            trial.gen.loc[gen, 'p_mw'] *= 1 + scale
            over, under = test(trial)
            if over:
                return base
            base = trial
        return base

    for _ in tqdm(range(trials), desc='Trial'):
        net = inc_load(net, step)
        net = inc_gen(net, step)

    pp.to_pickle(net, save_filepath)
    
def get_system_bases(net):
    S_base = float(net.sn_mva)
    V_base = net.bus.vn_kv.values.astype(np.float32)
    z_base = (V_base[net.line.from_bus.values] ** 2 / S_base).astype(np.float32)
    return {
        'S_base': S_base,
        'V_base': V_base,
        'z_base': z_base}

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

# Absolute Values, not per-unit
def create_sample(
        net,
        perturb_loads=False,
        load_scale_factor=(0.7, 1.3),
        perturb_generator_powers=True,
        gen_powers_scale_factor=(0.7, 1.3),
        perturb_generator_voltages=True,
        gen_voltages_scale_factor=(0.98, 1.02)):

    net = copy.deepcopy(net)

    if perturb_loads:
        scale_loads(net, load_scale_factor)
    if perturb_generator_powers:
        scale_generator_powers(net, gen_powers_scale_factor)
    if perturb_generator_voltages:
        scale_generator_voltages(net, gen_voltages_scale_factor)

    pp.runpp(net, numba=False)

    n = len(net.bus)
    X = np.zeros((n, 8), dtype=np.float32)
    Y = np.zeros((n, 4), dtype=np.float32)

    slack = int(net.ext_grid.iloc[0].bus)
    pv = set(net.gen.bus.astype(int))
    pq = set(range(n)) - pv - {slack}

    # PQ buses
    for bus in pq:
        load = net.load[net.load.bus == bus]
        P = load.p_mw.sum()
        Q = load.q_mvar.sum()

        X[bus] = [P, 0, Q, 0, 1, 0, 1, 0]
        Y[bus] = X[bus][:4]

    # PV buses
    for bus in pv:
        gen = net.gen[net.gen.bus == bus]
        load = net.load[net.load.bus == bus]
        P_gen = gen.p_mw.sum()
        P_load = load.p_mw.sum()
        P = P_gen - P_load
        V_kv = gen.vm_pu.iloc[0] * net.bus.vn_kv.iloc[bus]

        X[bus] = [
            P,
            V_kv,
            0,
            0,
            1,
            1,
            0,
            0
        ]
        Y[bus] = X[bus][:4]

    # Slack bus
    ext = net.ext_grid.iloc[0]
    V_kv = ext.vm_pu * net.bus.vn_kv.iloc[slack]

    X[slack] = [
        0,
        V_kv,
        0,
        np.deg2rad(ext.va_degree),
        0,
        1,
        0,
        1
    ]
    Y[slack] = X[slack][:4]

    # Finish constructing Y with unknown values from the power-flow solution.
    # Known values are already present in Y and must not be overwritten.
    unknown_P = X[:, 4] == 0
    unknown_V = X[:, 5] == 0
    unknown_Q = X[:, 6] == 0
    unknown_Theta = X[:, 7] == 0
    
    Y[unknown_P, 0] = net.res_bus.p_mw.values[unknown_P]
    
    Y[unknown_V, 1] = (
        net.res_bus.vm_pu.values[unknown_V]
        * net.bus.vn_kv.values[unknown_V])

    Y[unknown_Q, 2] = net.res_bus.q_mvar.values[unknown_Q]

    Y[unknown_Theta, 3] = np.deg2rad(
        net.res_bus.va_degree.values[unknown_Theta])

    return X, Y

def convert_to_per_unit(data, bases):
    S_base = bases['S_base']
    V_base = bases['V_base']

    data = data.copy()
    data[..., 0] /= S_base  # P
    data[..., 1] /= V_base  # V
    data[..., 2] /= S_base  # Q

    return data

def convert_to_absolute(data, bases):
    S_base = bases['S_base']
    V_base = bases['V_base']

    data = data.copy()
    data[..., 0] *= S_base  # P
    data[..., 1] *= V_base  # V
    data[..., 2] *= S_base  # Q
    # data[..., 3] = Theta unchanged

    return data

def create_dataset(
        net,
        n_samples,
        per_unit=True,
        perturb_loads=True,
        load_scale_factor=(0.7, 1.3),
        perturb_generator_powers=True,
        gen_powers_scale_factor=(0.7, 1.3),
        perturb_generator_voltages=True,
        gen_voltages_scale_factor=(0.98, 1.02)):

    edge_index, edge_attr = get_edge_data(net)
    bases = get_system_bases(net)
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

    X = np.array(X)
    Y = np.array(Y)

    if per_unit:
        X = convert_to_per_unit(X, bases)
        Y = convert_to_per_unit(Y, bases)

    return {
        'X': X,
        'Y': Y,
        'X_labels': np.array(['P', 'V', 'Q', 'Theta', 'mP', 'mV', 'mQ', 'mTheta']),
        'Y_labels': np.array(['P', 'V', 'Q', 'Theta']),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'per_unit': per_unit,
        'bases': bases}

def train_test_split(dataset, train_ratio=0.8):
    n = len(dataset['X'])
    indices = list(range(n))
    random.shuffle(indices)
    split = int(train_ratio * n)
    
    train = {k: v[indices[:split]] if k in ['X', 'Y'] else v for k, v in dataset.items()}
    test = {k: v[indices[split:]] if k in ['X', 'Y'] else v for k, v in dataset.items()}
    
    return train, test

def train_val_test_split(dataset, train_val_test_split=(0.7,0.15,0.15)):
    n = len(dataset['X'])
    indices = list(range(n))
    random.shuffle(indices)
    
    train_ratio, val_ratio, test_ratio = train_val_test_split
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    
    train = {
        k: v[indices[:train_end]] if k in ['X', 'Y'] else v
        for k, v in dataset.items()}
    
    val = {
        k: v[indices[:train_end]] if k in ['X', 'Y'] else v
        for k, v in dataset.items()}
    
    test = {
        k: v[indices[val_end:]] if k in ['X', 'Y'] else v
        for k, v in dataset.items()}
    
    return train, val, test

if __name__ == '__main__':
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    max_load_net_filepath = os.path.join(data_dir, 'case14_max_load_net.p')
    train_data_filepath = os.path.join(data_dir, 'case14_32sample_train.npy')
    test_data_filepath = os.path.join(data_dir, 'case14_32sample_test.npy')
    
    # n_samples = int(5e4) # 50,000 samples
    n_samples = 32 # Overfitting test
    load_scale_factor = (0.8, 1.2)
    gen_power_scale_factor = (0.8, 1.2)
    gen_voltage_scale_factor = (0.99, 1.01)
    
    get_max_load_net = False # Done
    
    net = pn.case14() 
    # inspect_net(net)
    
    if get_max_load_net: max_load_net(net, step=0.1, trials=20, save_filepath=max_load_net_filepath)
    max_load_net = pp.from_pickle(max_load_net_filepath)
    inspect_net(max_load_net)
    
    dataset = create_dataset(
        max_load_net,
        n_samples,
        per_unit=True,
        perturb_loads=True,
        load_scale_factor=load_scale_factor,
        perturb_generator_powers=True,
        gen_powers_scale_factor=gen_power_scale_factor,
        perturb_generator_voltages=True,
        gen_voltages_scale_factor=gen_voltage_scale_factor)
        
    train_data, test_data = train_test_split(dataset)
    
    np.save(train_data_filepath, train_data, allow_pickle=True)
    np.save(test_data_filepath, test_data, allow_pickle=True)
        