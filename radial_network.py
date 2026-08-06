# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:14:32 2026

@author: codett
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

import opendssdirect as dss
from dss import DSS
dss_engine = DSS()

class RadialNetwork:

    def __init__(self, source_dss_filepath):

        # Node Attributes
        self.nodes = [] # List of node indices [0,1,2,...,n]
        self.bus_names = [] # List of bus names
        self.root = 0 # Idx of root node
        self.V = [] # List of node voltages, including root node
        self.P = [] # List of real power injections, including root node
        self.Q = [] # List of reactive power injections, including root node
        self.bus_to_node = {} # Dict such that bus_to_node[bus_name] = node index

        # Branch Attributes
        self.branches = [] # List of node indices [(1,2),(2,3),...,(5,6)]
        self.r0 = {} # Dict of zero-seq branch resistance, indexed by (i,j)
        self.r1 = {} # Dict of pos-seq branch resistance, indexed by (i,j)
        self.x0 = {} # Dict of zero-seq branch reactance, indexed by (i,j)
        self.x1 = {} # Dict of pos-seq branch reactance, indexed by (i,j)
        self.p = {} # Dict of real branch power flow, indexed by (i,j)
        self.q = {} # Dict of reactive branch power flow, indexed by (i,j)
        self.i_mag = {} # Dict of branch current flow magnitude, indexed by (i,j)
        self.i_ang = {} # Dict of branch current flow angle, indexed by (i,j)
        self.v_drop_mag = {} # Dict of branch voltage drop magnitude, indexed by (i,j)
        self.v_drop_ang = {} # Dict of branch voltage drop angle, indexed by (i,j)
        self.branch_idx_to_name = {}
        self.branch_name_to_index = {} # Dict such that branch_name_to_index[branch_name] = branch index.

        self.lines = [] # List of Branch indices where the branch is a line. Sorted upstream -> downstream.
        self.transformers = [] # List of Branch indices where the branch is a transformer. Sorted upstream -> downstream.
        self.transformer_ratios = {} # Dict of transformer ratios (V_i:V_j) indexed by (i,j).

        # Network Structure
        self.children = {} # Dictionary such that children[node_idx] = list of node indices for all child nodes
        self.parent = {} # Dictionary such that parent[node_dx] = node index of parent node

        # Per-Unit
        self.S_base = 0.0 # System MVA Base
        self.V_base = [] # Voltage Base for each voltage region (Node-Wise)
        self.I_base = {} # Current Base for each voltage region (Branch-Wise)
        self.Z_base = {} # Impedance Base for each impedance region (Branch-Wise)

        self.load_from_dss(source_dss_filepath)

    # Set containing all nodes along path from root to node
    def C(self, i):
        path = []
        current = i
        while True:
            path.append(current)
            if current == self.root: break
            current = self.parent[current]
        path.reverse()
        return path

    # Set containing all nodes downstream of node i
    def D(self, i):
        stack = [i]
        downstream = []
        while stack:
            node = stack.pop()
            downstream.append(node)
            stack.extend(self.children.get(node, []))
        return downstream

    # Set containing all nodes upstream of node i
    def U(self, i):
        upstream = []
        current = i
        while current in self.parent:
            current = self.parent[current]
            upstream.append(current)
        return upstream

    # Set containing all lines along path from root to node
    def L(self, i):
        path = self.C(i)
        return [(path[k],path[k+1]) for k in range(len(path)-1)]

    # Set containing all lines connected to node i
    def M(self, i): return [(u,v) for (u,v) in self.lines if u==i or v==i]

    def load_topology_from_dss(self, dss_filepath):

        # Nodes
        bus_names = dss.Circuit.AllBusNames()
        if not bus_names:
            dss.Text.Command("Solve")
            bus_names = dss.Circuit.AllBusNames()
        self.nodes = list(range(len(bus_names)))
        self.bus_to_node_map = {name: idx for idx, name in enumerate(bus_names)}
        self.node_to_bus_map = {idx: name for idx, name in enumerate(bus_names)}

        # Lines
        self.branches = []
        self.branch_name_to_index = {}
        dss.Lines.First()
        while True:
            bus1 = dss.Lines.Bus1().split('.')[0]
            bus2 = dss.Lines.Bus2().split('.')[0]
            i = self.bus_to_node_map[bus1]
            j = self.bus_to_node_map[bus2]
            self.branches.append((i, j))
            self.lines.append((i, j))
            self.branch_idx_to_name[(i,j)] = dss.Lines.Name()
            self.branch_name_to_index[dss.Lines.Name()] = (i,j)
            if not dss.Lines.Next(): break

        # Transformers
        self.transformers = []
        self.transformer_ratios = {}
        if dss.Transformers.First() > 0:
            while True:
                buses = dss.CktElement.BusNames()
                bus1 = buses[0].split('.')[0]
                bus2 = buses[1].split('.')[0]

                i = self.bus_to_node_map[bus1]
                j = self.bus_to_node_map[bus2]
                self.transformers.append((i, j))
                self.branches.append((i, j))
                self.branch_idx_to_name[(i,j)] = dss.Transformers.Name()
                self.branch_name_to_index[dss.Transformers.Name()] = [(i, j)]

                dss.Transformers.Wdg(1)
                kv1 = dss.Transformers.kV()
                dss.Transformers.Wdg(2)
                kv2 = dss.Transformers.kV()
                ratio = kv1 / kv2
                self.transformer_ratios[(i, j)] = ratio

                if dss.Transformers.Next() == 0:
                    break

        # Tree Structure
        self.children = {i: [] for i in self.nodes}
        self.parent = {}
        for (i, j) in self.branches:
            self.children[i].append(j)
            self.parent[j] = i

    def calculate_bases(self, S_base=100e6, V_base_root=None):

        self.S_base = S_base

        # Initialize V_base at all nodes.
        V_base = self.V0 if not V_base_root else V_base_root
        self.V_base = [V_base] * len(self.nodes)

        if V_base_root: self.V[self.root] = self.V0 / V_base_root
        else: self.V[self.root] = 1.0

        # Propagate through Transformers
        for (i,j) in self.transformers:
            ratio = self.transformer_ratios[(i,j)]

            # Update base voltage downstream of transformer
            new_V_base = self.V_base[i] * (1 / ratio)
            downstream_nodes = self.D(j)
            for node in downstream_nodes: self.V_base[node] = new_V_base

        # Compute branch-wise I_base and Z_base
        self.I_base = {}
        self.Z_base = {}

        for (i, j) in self.branches:
            Vb = self.V_base[i]  # use upstream node base voltage
            self.I_base[(i, j)] = self.S_base / Vb
            self.Z_base[(i, j)] = (Vb ** 2) / self.S_base

    def load_branch_impedance_from_dss(self, dss_filepath):
        self.r0, self.r1, self.x0, self.x1 = {}, {}, {}, {}

        # Lines
        if dss.Lines.First() > 0:
            while True:
                bus1 = dss.Lines.Bus1().split('.')[0]
                bus2 = dss.Lines.Bus2().split('.')[0]
                i = self.bus_to_node_map[bus1]
                j = self.bus_to_node_map[bus2]
                length = dss.Lines.Length()
                self.r0[(i,j)] = dss.Lines.R0() * length / self.Z_base[(i,j)]
                self.r1[(i,j)] = dss.Lines.R1() * length / self.Z_base[(i,j)]
                self.x0[(i,j)] = dss.Lines.X0() * length / self.Z_base[(i,j)]
                self.x1[(i,j)] = dss.Lines.X1() * length / self.Z_base[(i,j)]
                if dss.Lines.Next() == 0: break

        # Transformers
        if dss.Transformers.First() > 0:
            while True:
                buses = dss.CktElement.BusNames()
                bus1 = buses[0].split('.')[0]
                bus2 = buses[1].split('.')[0]
                i = self.bus_to_node_map[bus1]
                j = self.bus_to_node_map[bus2]

                # Transformer Bases
                dss.Transformers.Wdg(1)
                V_tf_base = dss.Transformers.kV() * 1e3
                S_tf_base = dss.Transformers.kVA() * 1e3
                Z_tf_base = (V_tf_base ** 2) / S_tf_base

                # Transformer Impedance
                x_pu_tf = dss.Transformers.Xhl() / 100 # Given in Per-Unit (%)
                r_pu_tf = dss.Transformers.R() / 100 # Given in Per-Unit (%)
                x_true = x_pu_tf * Z_tf_base
                r_true = r_pu_tf * Z_tf_base
                x_pu = x_true / self.Z_base[(i,j)]
                r_pu = r_true / self.Z_base[(i,j)]

                # Positive Sequence Leakage Reactance
                self.r1[(i,j)], self.x1[(i,j)] = r_pu, x_pu

                # Assume Zero Sequence Impedance is zero.
                self.r0[(i,j)], self.x0[(i,j)] = 0.0, 0.0

                if dss.Transformers.Next() == 0: break

    def load_loads_from_dss(self, dss_filepath):

        # Loads
        self.P, self.Q = {}, {}
        dss.Loads.First()
        while True:
            load_name = dss.Loads.Name()
            dss.Circuit.SetActiveElement(load_name)
            bus_name = dss.CktElement.BusNames()[0].split('.')[0]
            node = self.bus_to_node_map[bus_name]
            peak_kw = dss.Loads.kW()
            peak_kvar = dss.Loads.kvar()
            shape_name = dss.Loads.Daily()
            dss.LoadShape.Name(shape_name)
            self.time_resolution = dss.LoadShape.SInterval()
            self.P[node] = np.array([peak_kw * s * 1e3 for s in dss.LoadShape.PMult()]) / self.S_base
            Qmult = dss.LoadShape.QMult() if dss.LoadShape.QMult() != [0.0] else dss.LoadShape.PMult()
            self.Q[node] = np.array([peak_kvar * s * 1e3 for s in Qmult]) / self.S_base
            if not dss.Loads.Next(): break

        self.T = min(min(len(v) for v in self.P.values()), min(len(v) for v in self.Q.values()))

        # Assign Zero power to Nodes without Loads
        self.P[self.root] = np.zeros(self.T)
        self.Q[self.root] = np.zeros(self.T)
        for node in self.nodes:
            if not node in self.P: self.P[node] = np.zeros(self.T)
            if not node in self.Q: self.Q[node] = np.zeros(self.T)

    def load_from_dss(self, dss_filepath):

        dss.Text.Command("Clear")
        dss.Text.Command(f"Compile [{dss_filepath}]")
        if not dss.Circuit.AllBusNames():
            dss.Text.Command("Solve")

        self.load_topology_from_dss(dss_filepath) # Nodes, Branches, Transformers, Tree Structure

        # Source
        dss.Vsources.First()
        bus = dss.CktElement.BusNames()[0].split('.')[0]
        dss.Circuit.SetActiveBus(bus)
        self.root = self.bus_to_node_map[bus]
        self.V = [0.0] * len(self.nodes) # Initialize node voltages.
        self.V0 = dss.Bus.VMagAngle()[0] # Source true voltage (not per-unit).

        self.calculate_bases()

        self.load_branch_impedance_from_dss(dss_filepath)

        self.load_loads_from_dss(dss_filepath)

    def export_to_dss(self, dss_filepath, circuit_name='RadialNetwork'):

        with open(dss_filepath, 'w') as f:

            f.write(f'New Circuit.{circuit_name} phases=1 basekv={self.V0/1e3} pu=1.0\n')
            f.write(f'New Vsource.Vsource bus1={self.node_to_bus_map[self.root]} phases=1 basekv={self.V0 / 1e3} pu={self.V[self.root]}\n\n')

            # Lines
            for (i,j) in self.lines:
                f.write(
                    f"New Line.{self.branch_idx_to_name[(i, j)]} "
                    f"phases=1 "
                    f"bus1={self.node_to_bus_map[i]} bus2={self.node_to_bus_map[j]} "
                    f"r1={self.r1[(i, j)] * self.Z_base[(i, j)]} x1={self.x1[(i, j)] * self.Z_base[(i, j)]} r0={self.r0[(i, j)] * self.Z_base[(i, j)]} x0={self.x0[(i, j)] * self.Z_base[(i, j)]} "
                    f"length=1 units=km\n")
            f.write("\n")

            # Transformers
            for (i,j) in self.transformers:
                ratio = self.transformer_ratios[(i,j)]
                kv_primary = self.V_base[i] / 1e3
                kv_secondary = self.V_base[j] / 1e3
                f.write(
                    f"New Transformer.XF_{i}_{j} "
                    f"phases=1 windings=2 "
                    f"buses=[{self.node_to_bus_map[i]}, {self.node_to_bus_map[j]}] "
                    f"conns=[wye, wye] "
                    f"kvs=[{kv_primary}, {kv_secondary}] "
                    f"kvas=[{self.S_base / 1e3}, {self.S_base / 1e3}] " # TODO: Replace this
                    f"%r={self.r1[(i,j)] * 100} xhl={self.x1[(i,j)] * 100}\n" # TODO: Replace this.
                )
            f.write("\n")

            # Cache Unique Load Shapes
            load_kw, load_kvar = {}, {}
            shape_map = {}
            shape_counter = 0
            node_to_shape = {}

            # Load-Shapes
            for node in self.nodes:
                max_p = np.max(self.P[node])
                max_q = np.max(self.Q[node])
                load_kw[node] = max_p * self.S_base / 1e3
                load_kvar[node] = max_q * self.S_base / 1e3
                P_series_scaled = self.P[node] / max_p if max_p != 0 else np.zeros_like(self.P[i])
                Q_series_scaled = self.Q[node] / max_q if max_q != 0 else np.zeros_like(self.Q[i])
                key = (
                    tuple(np.round(P_series_scaled, decimals=3)),
                    tuple(np.round(Q_series_scaled, decimals=3))
                )
                if key not in shape_map:
                    shape_name = f"LS_{shape_counter}"
                    shape_map[key] = shape_name
                    shape_counter += 1
                    P_mult_str = " ".join(str(p) for p in P_series_scaled)
                    Q_mult_str = " ".join(str(q) for q in Q_series_scaled)
                    f.write(
                        f"New LoadShape.{shape_name} "
                        f"npts={self.T} "
                        f"Sinterval={self.time_resolution} "
                        f"Pmult=({P_mult_str})\n"
                        f"Qmult=({Q_mult_str})\n"
                    )
                else:
                    shape_name = shape_map[key]
                node_to_shape[node] = shape_name
            f.write("\n")

            # Loads
            for node in self.nodes:
                f.write(
                    f"New Load.Load_{node} "
                    f"bus1={self.node_to_bus_map[node]} "
                    f"phases=1 "
                    f"conn=wye "
                    f"model=1 "
                    f"kV={self.V_base[node] / 1e3} "
                    f"kW={load_kw[node]} "
                    f"kVar={load_kvar[node]} "
                    f"Daily={node_to_shape[node]}\n")
            f.write("\n")

            # Simulation Setup
            f.write("Set mode=Daily\n")
            f.write(f"Set number={self.T}\n")
            f.write(f"Set stepsize={self.time_resolution}\n")

    def lin_dist_flow(self, per_unit=True):

        V = defaultdict(list) # Node Voltage (p.u.)
        v = defaultdict(list) # Branch Voltage Drop (p.u.)
        p = defaultdict(list) # Branch Real Power Flow (p.u.)
        q = defaultdict(list) # Branch Reactive Power Flow (p.u.)
        i_branch = defaultdict(list) # Branch Current (p.u.)

        V[self.root] = np.ones(self.T) * self.V[self.root] # Root Voltage

        # Branch Power-Flow and Squared-Voltage Drop
        for t in range(self.T):
            v_sq_drop = {}
            for (i,j) in self.branches:
                p_ij = sum(self.P[h][t] for h in self.D(j))
                q_ij = sum(self.Q[h][t] for h in self.D(j))
                p[(i,j)].append(p_ij)
                q[(i,j)].append(q_ij)
                r = self.r1[(i,j)]
                x = self.x1[(i,j)]
                v_sq_drop[(i,j)] = 2 * (r * p_ij + x * q_ij) # LDF Voltage Equation (per-unit)

            # Node Voltages
            V_t = {self.root: self.V[self.root]}
            for node in self.nodes:
                if node == self.root: continue
                V_sq = (self.V[self.root]**2 - sum(v_sq_drop[(h,k)] for (h,k) in self.L(node)))
                V_t[node] = np.sqrt(max(V_sq, 0.0))
                V[node].append(V_t[node])

            # Branch Voltage Drops and Current
            for (i, j) in self.branches:
                v_ij = V_t[i] - V_t[j]
                S_ij = np.hypot(p[(i, j)][-1], q[(i, j)][-1])
                i_ij = S_ij / V_t[i]
                v[(i,j)].append(v_ij)
                i_branch[(i, j)].append(i_ij)

        # Convert Lists to Arrays
        V = {k: np.asarray(v) for k, v in V.items()}
        v = {k: np.asarray(v) for k, v in v.items()}
        i_branch = {k: np.asarray(v) for k, v in i_branch.items()}
        p = {k: np.asarray(v) for k, v in p.items()}
        q = {k: np.asarray(v) for k, v in q.items()}

        if not per_unit:
            V = {node: values * self.V_base[node] for node, values in V.items()}
            v = {(i,j): values * self.V_base[i] for (i,j), values in v.items()}
            i_branch = {(i,j): values * self.I_base[(i,j)] for (i,j), values in i_branch.items()}
            p = {(i,j): values * self.S_base[(i,j)] for (i,j), values in p.items()}
            q = {(i,j): values * self.S_base[(i,j)] for (i,j), values in q.items()}

        return {
            "V": V,
            "v": v,
            "i": i_branch,
            "p": p,
            "q": q
        }


    def solve_dss(self, dss_filepath, per_unit=True):

        V = defaultdict(list) # Node Voltage Injection
        v = defaultdict(list) # Branch Voltage Drop
        p = defaultdict(list) # Branch Real Power Flow
        q = defaultdict(list) # Branch Reactive Power Flow
        i_branch = defaultdict(list) # Branch Current Flow

        dss.Text.Command("Clear")
        dss.Text.Command(f"Compile [{dss_filepath}]")

        # Add Monitors
        i=self.root
        j=self.children[i][0]
        if (i, j) in self.lines:
            element = "Line"
            element_name = self.branch_idx_to_name[(i, j)]
        elif (i, j) in self.transformers:
            element = "Transformer"
            element_name = self.branch_idx_to_name[(i, j)]
        dss.Text.Command(f"New Monitor.V_root element={element}.{element_name} mode=0 terminal=1")
        for (i,j) in self.transformers:
            transformer = self.branch_idx_to_name[(i, j)]
            dss.Text.Command(f"New Monitor.pq_{i}_{j} element=Transformer.{transformer} mode=1 terminal=1 ppolar=no")  # Power Mode, From Bus
            dss.Text.Command(f"New Monitor.vi_{i}_{j} element=Transformer.{transformer} mode=0 terminal=2")  # Voltage / Current Mode, From Bus
        for (i,j) in self.lines:
            line = self.branch_idx_to_name[(i,j)]
            dss.Text.Command(f"New Monitor.pq_{i}_{j} element=Line.{line} mode=1 terminal=1 ppolar=no")  # Power Mode, From Bus
            dss.Text.Command(f"New Monitor.vi_{i}_{j} element=Line.{line} mode=0 terminal=2")  # Voltage / Current Mode, From Bus

        dss.Text.Command("Solve")

        # Read Monitor Data
        dss.Monitors.Name("v_root")
        V[self.root] = dss.Monitors.Channel(1) # Voltage Magnitude
        for (i,j) in self.branches:
            dss.Monitors.Name(f"pq_{i}_{j}")
            p[(i,j)] = dss.Monitors.Channel(1) * 1e3 # Reads in kW
            q[(i,j)] = dss.Monitors.Channel(2) * 1e3 # Reads in kVar
            dss.Monitors.Name(f"vi_{i}_{j}")
            V[j] = dss.Monitors.Channel(1) # Reads in Volts
            if dss.Monitors.NumChannels() <= 4: i_branch[(i,j)] = dss.Monitors.Channel(3) # Reads in Amps from Line Monitor
            else: i_branch[(i,j)] = dss.Monitors.Channel(4) # Read in Amps from Transformer Monitor
        for (i,j) in self.lines:
            v[(i,j)] = (V[i] - V[j])
        dss.Text.Command("Clear") # Remove Monitors

        # Convert to numpy arrays
        V = {k: np.asarray(v) for k, v in V.items()}
        v = {k: np.asarray(v) for k, v in v.items()}
        i_branch = {k: np.asarray(v) for k, v in i_branch.items()}
        p = {k: np.asarray(v) for k, v in p.items()}
        q = {k: np.asarray(v) for k, v in q.items()}

        if per_unit:
            V = { node: values / self.V_base[node] for node, values in V.items()}
            v = {(i,j): values / self.V_base[i] for (i,j), values in v.items()}
            i_branch = {(i,j): values / self.I_base[(i,j)] for (i,j), values in i_branch.items()}
            p = {(i,j): values / self.S_base for (i,j), values in p.items()}
            q = {(i,j): values / self.S_base for (i,j), values in q.items()}

        return {
            'V': V,
            'v': v,
            'i': i_branch,
            'p': p,
            'q': q}

    def display_results(self, results, t=0, show=False, return_results=False, save=False, save_folderpath=None):

        # Node Table
        node_data = []
        for i in self.nodes:
            node_data.append({
                'node': i,
                'V': results['V'][i][t],
                'P': self.P[i][t],
                'Q': self.Q[i][t]})
        df_nodes = pd.DataFrame(node_data).sort_values(by='node')

        # Line Table
        line_data = []
        for (i, j) in self.lines:
            line_data.append({
                'from': i,
                'to': j,
                'r1': self.r1[(i, j)],
                'x1': self.x1[(i, j)],
                'r0': self.r0[(i, j)],
                'x0': self.x0[(i, j)],
                'p_flow': results['p'][(i, j)][t],
                'q_flow': results['q'][(i, j)][t],
                'i_flow': results['i'][(i, j)][t],
                'v_drop': results['v'][(i, j)][t]})
        df_lines = pd.DataFrame(line_data).sort_values(by=['from', 'to'])

        # Display
        if show:
            print(f"\n=== NODE STATES (t={t}) ===")
            print(df_nodes.to_string(index=False))
            print(f"\n=== LINE STATES (t={t}) ===")
            print(df_lines.to_string(index=False))

        if save:
            df_nodes.to_csv(os.path.join(save_folderpath, "node_data.csv"))
            df_lines.to_csv(os.path.join(save_folderpath, "line_data.csv"))

        if return_results: return node_data, line_data

def compute_error(results1, results2, normalize_error=False):
    def error_fn(a, b):
        num = np.linalg.norm(a - b)
        if not normalize_error:
            return num  # absolute error
        den = num + np.linalg.norm(b)
        return 0.0 if np.isclose(den, 0.0) else num / den

    errors = {}
    for key in ['V', 'v', 'i', 'p', 'q']:
        errors[key] = {}
        keys1 = set(results1[key].keys())
        keys2 = set(results2[key].keys())
        common_keys = keys1 & keys2
        for k in common_keys:
            a = np.array(results1[key][k])
            b = np.array(results2[key][k])
            errors[key][k] = error_fn(a, b)
    return errors

def display_error(errors):

    # Node Errors
    node_data = [{'node': i, 'V_error': errors['V'][i]} for i in sorted(errors['V'])]
    df_nodes = pd.DataFrame(node_data)

    # Branch Errors
    branch_data = []
    for (i,j) in sorted(errors['v']):
        branch_data.append({
            'from': i,
            'to': j,
            'v_error': errors['v'][(i,j)],
            'i_error': errors['i'][(i,j)],
            'p_error': errors['p'][(i,j)],
            'q_error': errors['q'][(i,j)]})
    df_branches = pd.DataFrame(branch_data)

    print("\n=== NODE ERRORS ===")
    print(df_nodes.to_string(index=False))
    print("\n=== BRANCH ERRORS ===")
    print(df_branches.to_string(index=False))

# Testing

if __name__ == '__main__':

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    data_dir = os.path.join(base_dir, 'data')

    # dss_filepath = os.path.join(data_dir, 'ieee_123bus', 'Master.dss')
    dss_filepath = os.path.join(data_dir, 'simple_radial.dss')
    dss_filepath2 = os.path.join(data_dir, 'simple_radial2.dss')

    network = RadialNetwork(source_dss_filepath=dss_filepath)
    network.export_to_dss(dss_filepath2, 'circuit2')
    network2 = RadialNetwork(source_dss_filepath=dss_filepath2)

    results1 = network.solve_dss(dss_filepath)
    results2 = network2.solve_dss(dss_filepath2)

    results1_ldf = network.lin_dist_flow()
    results2_ldf = network2.lin_dist_flow()

    # network.display_results(results1, show=True)

    errors1 = compute_error(results1, results1_ldf)
    errors2 = compute_error(results2, results2_ldf)

    display_error(errors1)
    display_error(errors2)

    errors3 = compute_error(results1, results2)
    display_error(errors3)

    print('')


# To Do: We won't know for certain that the transformer impedance scaling is correct until we run power flow and compare results.
# To Do: Simplify by integrating dss-Python