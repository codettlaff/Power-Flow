# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:47:12 2026
Author: Casey Dettlaff

PowerFlowNet-stype dataset generation with project's existing dataset format.
Procedure:
    - Start from fresh default PandaPower case for every sample.
    - Set line shunt capacitance to zero.
    - Perturb line resistance, reactance, and length by +/- 20%.
    - Sample generator voltage setpoints uniformly from 1.00 - 1.05 p.u.
    - Sample generator active power from N(Pg, 0.1*|Pg|).
    - Sample load active/reactive power from N(Pd, 0.1*|Pd|) and N(Qd, 0.1*|Qd|).
    - Solve each scenario with Newton-Raphson power-flow method.

Final dataset format:
    - X : (size [samples, buses, 8]) [P, V, Q, Theta, mP, mV, mQ, mTheta]
    - Y : (size [samples, buses, 4]) [P, V, Q, Theta]
    - edge_index : (size [2, directed_edges]) [from_bus, to_bus]
    - edge_attr : (size [samples, directed_edges, 2]) [r, x]
    - per_unit : True/False
    - bases : system bases Dict {'S_base': S_base, 'V_base': V_base, 'z_bases': z_base}
        - S_base : System S base (float)
        - V_base : V base at each node (np.array size [n_nodes])
        - z_bases : impedance base at each node (np.array size [n_nodes])
"""

import os
import copy
import random
import argparse

import numpy as np
from tqdm import tqdm

import pandapower as pp
import pandapower.networks as pn

# Configuration

N_SAMPLES = 20000
LINE_SCALE = (0.8, 1.2)
GEN_VOLTAGE_RANGE = (1.00, 1.05)
GEN_POWER_STD = 0.10
LOAD_POWER_STD = 0.10

TRAIN_RATIO = 0.8
PER_UNIT = True


def get_case(case):
    """Return a fresh PandaPower network for the requested test case."""
    if case == '14':
        net = pn.case14()
    if case == '118':
        net = pn.case118()
    if case == '6470rte':
        net = pn.case6470rte()
    return net

def remove_c_nf(net):
    """Set line shunt capacitance to zero, as in PowerFlowNet."""
    if len(net.line):
        net.line['c_nf_per_km'] = 0.0


def get_system_bases(net):
    """
    Return the system bases used by the project's dataset format.
    Voltage and impedance bases are stored per bus.
    """
    S_base = float(net.sn_mva)
    V_base = net.bus['vn_kv'].values.astype(np.float32)
    z_base = V_base ** 2 / S_base

    return {
        'S_base': S_base,
        'V_base': V_base,
        'z_base': z_base.astype(np.float32)}


def get_line_z(net, per_unit=True):
    """
    Return total line resistance and reactance.
    The total series impedance = PandaPower ohm/km parameters * line length.
    If per_unit=True, return [r_pu, x_pu].
    If per_unit=False, return [r_ohm, x_ohm].
    """
    if len(net.line) == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32))

    r_ohm = (
        net.line['r_ohm_per_km'].values
        * net.line['length_km'].values)
    x_ohm = (
        net.line['x_ohm_per_km'].values
        * net.line['length_km'].values)

    if not per_unit:
        return (
            r_ohm.astype(np.float32),
            x_ohm.astype(np.float32))

    from_bus = net.line['from_bus'].values.astype(int)
    z_base = (net.bus['vn_kv'].values[from_bus] ** 2 / net.sn_mva)

    r_pu = r_ohm / z_base
    x_pu = x_ohm / z_base

    return (
        r_pu.astype(np.float32),
        x_pu.astype(np.float32))


def get_trafo_z(net, per_unit=True):
    """
    Return transformer resistance/reactance.
    If per_unit=True, return [r_pu, x_pu].
    If per_unit=False, return [r_ohm, x_ohm].

    Transformer impedances in PandaPower are represented by vk_percent
    and vkr_percent on the transformer rating base. They are first
    converted to the network S base, then to ohms using the high-voltage
    bus impedance base when absolute units are requested.

    PowerFlowNet preprocessing ignores transformer magnetizing current
    and iron losses.
    """
    if len(net.trafo) == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32))

    z_pu = (
        net.trafo['vk_percent'].values / 100.0
        * 1000.0 / net.sn_mva)

    r_pu = (
        net.trafo['vkr_percent'].values / 100.0
        * 1000.0 / net.sn_mva)

    # Numerical protection for very small round-off errors.
    x_squared = np.maximum(z_pu ** 2 - r_pu ** 2, 0.0)
    x_pu = np.sqrt(x_squared)

    if per_unit:
        return (
            r_pu.astype(np.float32),
            x_pu.astype(np.float32))

    hv_bus = net.trafo['hv_bus'].values.astype(int)
    z_base = (
        net.bus['vn_kv'].values[hv_bus] ** 2
        / net.sn_mva)

    r_ohm = r_pu * z_base
    x_ohm = x_pu * z_base

    return (
        r_ohm.astype(np.float32),
        x_ohm.astype(np.float32))


def get_edge_index(net):
    """
    Build the project's directed edge representation.

    Lines use [from_bus, to_bus]. Transformers use [hv_bus, lv_bus].
    The reverse of each physical branch is then added so that message
    passing has a directed edge in both directions.
    """
    line_edges = np.array([
        net.line['from_bus'].values,
        net.line['to_bus'].values
    ], dtype=np.int64)

    if len(net.trafo):
        trafo_edges = np.array([
            net.trafo['hv_bus'].values,
            net.trafo['lv_bus'].values
        ], dtype=np.int64)

        physical_edges = np.concatenate(
            [line_edges, trafo_edges],
            axis=1)
    else:
        physical_edges = line_edges

    return np.concatenate(
        [physical_edges, physical_edges[::-1]],
        axis=1)


def get_edge_attr(net, per_unit=True):
    """
    Return [r, x] for every directed edge.
    If per_unit=True, impedance is stored in per-unit.
    If per_unit=False, impedance is stored in ohms.
    """
    line_r, line_x = get_line_z(net, per_unit=per_unit)
    line_attr = np.column_stack([line_r, line_x])

    if len(net.trafo):
        trafo_r, trafo_x = get_trafo_z(net, per_unit=per_unit)
        trafo_attr = np.column_stack([trafo_r, trafo_x])
        physical_attr = np.concatenate(
            [line_attr, trafo_attr],
            axis=0)
    else:
        physical_attr = line_attr

    return np.concatenate(
        [physical_attr, physical_attr],
        axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# PowerFlowNet perturbation procedure
# ---------------------------------------------------------------------------

def perturb_network(net):
    """
    Apply the PowerFlowNet sampling procedure to a fresh base network.
    Returns the perturbed network after its input parameters have been
    modified, but before solving the power flow.
    """

    net = copy.deepcopy(net)

    # Save nominal values before perturbing them
    r = net.line['r_ohm_per_km'].values.copy()
    x = net.line['x_ohm_per_km'].values.copy()
    length = net.line['length_km'].values.copy()
    Pg = net.gen['p_mw'].values.copy()
    Pd = net.load['p_mw'].values.copy()
    Qd = net.load['q_mvar'].values.copy()

    # PowerFlowNet: independently perturb each line parameter by +/-20%.
    if len(net.line):
        net.line['r_ohm_per_km'] = np.random.uniform(
            LINE_SCALE[0] * r,
            LINE_SCALE[1] * r)

        net.line['x_ohm_per_km'] = np.random.uniform(
            LINE_SCALE[0] * x,
            LINE_SCALE[1] * x)

        net.line['length_km'] = np.random.uniform(
            LINE_SCALE[0] * length,
            LINE_SCALE[1] * length)

    # PowerFlowNet: generator voltage is uniform on [1.00, 1.05] p.u.
    if len(net.gen):
        net.gen['vm_pu'] = np.random.uniform(
            GEN_VOLTAGE_RANGE[0],
            GEN_VOLTAGE_RANGE[1],
            size=len(net.gen))

        # PowerFlowNet: Pg ~ N(Pg_nominal, 0.1*|Pg_nominal|)
        net.gen['p_mw'] = np.random.normal(
            Pg,
            GEN_POWER_STD * np.abs(Pg))

    # PowerFlowNet: load P/Q ~ N(nominal, 0.1*|nominal|)
    if len(net.load):
        net.load['p_mw'] = np.random.normal(
            Pd,
            LOAD_POWER_STD * np.abs(Pd))

        net.load['q_mvar'] = np.random.normal(
            Qd,
            LOAD_POWER_STD * np.abs(Qd))

    return net


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def create_sample(net, init='flat', per_unit=True):
    """
    Generate one X/Y sample from a perturbed network.

    X follows the project's existing 8-feature representation:
        [P, V, Q, Theta, mP, mV, mQ, mTheta]

    Y contains the complete solved state:
        [P, V, Q, Theta]

    Known node quantities are copied directly into X and Y. Unknown
    quantities are filled into Y from the Newton-Raphson solution.

    Edge attributes use the same unit convention as X/Y:
        per_unit=True  -> [r_pu, x_pu]
        per_unit=False -> [r_ohm, x_ohm]
    """
    
    net = perturb_network(net)

    # PowerFlowNet uses Newton-Raphson for the ground-truth solution.
    pp.runpp(
        net,
        algorithm='nr',
        init=init,
        numba=False)

    S_base = float(net.sn_mva)
    V_base = net.bus['vn_kv'].values.astype(np.float32)
    n = len(net.bus)

    X = np.zeros((n, 8), dtype=np.float32)
    Y = np.zeros((n, 4), dtype=np.float32)

    slack = int(net.ext_grid.iloc[0].bus)
    pv_buses = set(net.gen.bus.astype(int))
    pq_buses = set(range(n)) - pv_buses - {slack}

    # -----------------------------------------------------------------------
    # PQ buses: P and Q are known.
    # P/Q represent net injections where P_net = P_generation - P_load
    # -----------------------------------------------------------------------
    for bus in pq_buses:
        load = net.load[net.load.bus == bus]

        P_load = load['p_mw'].sum()
        Q_load = load['q_mvar'].sum()

        X[bus] = [
            -P_load / S_base,
            0.0,
            -Q_load / S_base,
            0.0,
            1.0, 0.0, 1.0, 0.0]

    # -----------------------------------------------------------------------
    # PV buses: P and V are known.
    # -----------------------------------------------------------------------
    for bus in pv_buses:
        gens = net.gen[net.gen.bus == bus]
        loads = net.load[net.load.bus == bus]

        P_gen = gens['p_mw'].sum()
        P_load = loads['p_mw'].sum()

        P_net = P_gen - P_load
        V_pu = gens['vm_pu'].iloc[0]

        X[bus] = [
            P_net / S_base,
            V_pu,
            0.0,
            0.0,
            1.0, 1.0, 0.0, 0.0
        ]

    # -----------------------------------------------------------------------
    # Slack bus: V and Theta are known.
    # -----------------------------------------------------------------------
    ext = net.ext_grid.iloc[0]
    X[slack] = [
        0.0,
        ext.vm_pu,
        0.0,
        np.deg2rad(ext.va_degree),
        0.0, 1.0, 0.0, 1.0]

    # Start Y with the known values. Unknown values are then filled from the
    # Newton-Raphson result.
    Y[:, :] = X[:, :4]

    known_P = X[:, 4].astype(bool)
    known_V = X[:, 5].astype(bool)
    known_Q = X[:, 6].astype(bool)
    known_theta = X[:, 7].astype(bool)

    # P
    unknown_P = ~known_P
    Y[unknown_P, 0] = net.res_bus.p_mw.values[unknown_P] / S_base

    # V
    unknown_V = ~known_V
    Y[unknown_V, 1] = net.res_bus.vm_pu.values[unknown_V]

    # Q
    unknown_Q = ~known_Q
    Y[unknown_Q, 2] = net.res_bus.q_mvar.values[unknown_Q] / S_base

    # Theta
    unknown_theta = ~known_theta
    Y[unknown_theta, 3] = np.deg2rad(
        net.res_bus.va_degree.values[unknown_theta])

    edge_index = get_edge_index(net)
    edge_attr = get_edge_attr(net, per_unit=True)

    if not per_unit:
        # X/Y are initially constructed in per-unit. 
        # Convert back to absolute units.
        bases = get_system_bases(net)
        X, edge_attr = convert_to_absolute(
            X,
            bases,
            edge_attr=edge_attr,
            edge_index=edge_index)
        Y = convert_to_absolute(Y, bases)

    return X, Y, edge_index, edge_attr


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def get_edge_z_bases(bases, edge_index):
    """
    Return the impedance base associated with each directed edge.
    For transformer branches this convention preserves the side of the network
    to which the directed edge is attached.
    """
    z_base = np.asarray(bases['z_bases'])
    source_bus = np.asarray(edge_index[0], dtype=np.int64)
    return z_base[source_bus]

def convert_to_per_unit(
        data,
        bases,
        edge_attr=None,
        edge_index=None):
    """
    Convert [P, V, Q, Theta] fields to per-unit.

    If edge_attr is supplied, its [r, x] columns are also converted from
    ohms to per-unit using the impedance base of each directed edge.

    Returns data when edge_attr is not supplied. Otherwise returns
    (data, edge_attr).
    """
    S_base = bases['S_base']
    V_base = np.asarray(bases['V_base'])

    data = data.copy()
    data[..., 0] /= S_base
    data[..., 1] /= V_base
    data[..., 2] /= S_base

    if edge_attr is None:
        return data

    edge_attr = edge_attr.copy()
    z_base = get_edge_z_bases(bases, edge_index)

    edge_attr[..., 0] /= z_base
    edge_attr[..., 1] /= z_base

    return data, edge_attr


def convert_to_absolute(
        data,
        bases,
        edge_attr=None,
        edge_index=None):
    """
    Convert [P, V, Q, Theta] fields from per-unit to absolute units.

    P and Q are converted to MW/Mvar, V is converted to kV, and Theta is
    unchanged.

    If edge_attr is supplied, its [r, x] columns are also converted from
    per-unit to ohms using the impedance base of each directed edge.

    Returns data when edge_attr is not supplied. Otherwise returns
    (data, edge_attr).
    """
    S_base = bases['S_base']
    V_base = np.asarray(bases['V_base'])

    data = data.copy()
    data[..., 0] *= S_base
    data[..., 1] *= V_base
    data[..., 2] *= S_base

    if edge_attr is None:
        return data

    edge_attr = edge_attr.copy()
    z_base = get_edge_z_bases(bases, edge_index)

    edge_attr[..., 0] *= z_base
    edge_attr[..., 1] *= z_base

    return data, edge_attr


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def create_dataset(
        net,
        n_samples,
        per_unit=PER_UNIT):
    """
    Generate a dataset using PowerFlowNet-style sampling.

    Failed Newton-Raphson cases are discarded and generation continues until
    exactly n_samples successful cases have been collected.
    """

    
    base_net = copy.deepcopy(net)
    remove_c_nf(base_net) # Remove line capacitance
    bases = get_system_bases(net)
    
    # Get Base Case solution for Warm Start
    pp.runpp(base_net, algorithm='nr', init='flat', numba=False)

    X = []
    Y = []
    edge_attrs = []

    successful_samples = 0
    attempted_samples = 0

    with tqdm(total=n_samples, desc='Generating samples') as progress:
        while successful_samples < n_samples:
            attempted_samples += 1

            try:
                x, y, edge_index, edge_attr = create_sample(
                    base_net,
                    init='results',
                    per_unit=per_unit)
            except pp.LoadflowNotConverged:
                continue

            X.append(x)
            Y.append(y)
            edge_attrs.append(edge_attr)

            successful_samples += 1
            progress.update(1)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    edge_attr = np.asarray(edge_attrs, dtype=np.float32)

    return {
        'X': X,
        'Y': Y,
        'X_labels': np.array([
            'P', 'V', 'Q', 'Theta',
            'mP', 'mV', 'mQ', 'mTheta'
        ]),
        'Y_labels': np.array([
            'P', 'V', 'Q', 'Theta'
        ]),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'per_unit': per_unit,
        'bases': bases
    }


def train_val_test_split(dataset, train_val_test_split=(0.7, 0.15, 0.15)):
    """Randomly split X, Y, and per-sample edge_attr into train/validation/test sets."""
    n = len(dataset['X'])
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(train_val_test_split[0] * n)
    val_end = train_end + int(train_val_test_split[1] * n)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    sample_fields = ['X', 'Y', 'edge_attr']

    train = {
        k: v[train_indices] if k in sample_fields else v
        for k, v in dataset.items()
    }

    val = {
        k: v[val_indices] if k in sample_fields else v
        for k, v in dataset.items()
    }

    test = {
        k: v[test_indices] if k in sample_fields else v
        for k, v in dataset.items()
    }

    return train, val, test

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Configuration
    CASE = '14'
    SAMPLES = N_SAMPLES
    TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)
    PER_UNIT = True

    print(f'Generating {SAMPLES} samples')
    print(f'Case: IEEE {CASE}')
    print('Sampling method: PowerFlowNet')
    print(f'Line parameter scale: {LINE_SCALE[0]:.2f}-{LINE_SCALE[1]:.2f}')
    print(
        'Generator voltage range: '
        f'{GEN_VOLTAGE_RANGE[0]:.2f}-{GEN_VOLTAGE_RANGE[1]:.2f} p.u.')
    print(f'Generator/load power standard deviation: {GEN_POWER_STD:.0%}')

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    train_data_filepath = os.path.join(
        data_dir, f'case{CASE}_PowerFlowNet_{SAMPLES}samples_train.npy')
    val_data_filepath = os.path.join(
        data_dir, f'case{CASE}_PowerFlowNet_{SAMPLES}samples_val.npy')
    test_data_filepath = os.path.join(
        data_dir, f'case{CASE}_PowerFlowNet_{SAMPLES}samples_test.npy')

    net = get_case(CASE)

    dataset = create_dataset(
        net,
        SAMPLES,
        per_unit=PER_UNIT)

    train_data, val_data, test_data = train_val_test_split(
        dataset,
        train_val_test_split)

    np.save(train_data_filepath, train_data, allow_pickle=True)
    np.save(val_data_filepath, val_data, allow_pickle=True)
    np.save(test_data_filepath, test_data, allow_pickle=True)

    print('\nDataset generated successfully.')
    print(f'X shape:          {dataset["X"].shape}')
    print(f'Y shape:          {dataset["Y"].shape}')
    print(f'edge_index shape: {dataset["edge_index"].shape}')
    print(f'edge_attr shape:  {dataset["edge_attr"].shape}')
    print(f'Train samples:    {len(train_data["X"])}')
    print(f'Test samples:     {len(test_data["X"])}')
    print(f'Saved train data: {train_data_filepath}')
    print(f'Saved val data: {val_data_filepath}')
    print(f'Saved test data:  {test_data_filepath}')
