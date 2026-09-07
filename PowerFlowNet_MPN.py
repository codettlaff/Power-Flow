# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:53:03 2026
Author: Casey Dettlaff

Train PowerFlowNetMPN on generated data.
The generated dataset stores:
    X : [samples, buses, 8] = [P, V, Q, Theta, mP, mV, mQ, mTheta]
    Y : [samples, buses, 4] = [P, V, Q, Theta]
    edge_index : [2, directed_edges]
    edge_attr : [samples, directed_edges, 2] = [r, x]
    per_unit : True / False
    
The four bus-type features are reconstructed:
    PQ -> [1, 0, 0, 0]
    PV -> [0, 1, 0, 0]
    Slack -> [0, 0, 1, 0]

Training loss is weighted MSE on unknown quantities only.

"""

import argparse
import copy
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing, TAGConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

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
    
# ========== Train Model ========== #

def load_dataset(filepath):
    dataset = np.load(filepath, allow_pickle=True).item()
    return dataset

def build_bus_type_features(x):
    """
    Reconstruct the 4-column bus-type block expected by PowerFlowNetMPN.
    """
    mask = x[:, 4:8]
    pq = (
        (mask[:, 0] == 1)
        & (mask[:, 1] == 0)
        & (mask[:, 2] == 1)
        & (mask[:, 3] == 0))
    pv = (
        (mask[:, 0] == 1)
        & (mask[:, 1] == 1)
        & (mask[:, 2] == 0)
        & (mask[:, 3] == 0))
    slack = (
        (mask[:, 0] == 0)
        & (mask[:, 1] == 1)
        & (mask[:, 2] == 0)
        & (mask[:, 3] == 1))
    
    bus_type = np.zeros((len(x), 4), dtype=np.float32)
    bus_type[pq, 0] = 1.0
    bus_type[pv, 1] = 1.0
    bus_type[slack, 2] = 1.0
    
    return bus_type

def make_pyg_dataset(dataset):
    """
    Convert the project's dictionary dataset into a list of PyG Data Objects'
    """
    X = np.asarray(dataset['X'], dtype=np.float32)
    Y = np.asarray(dataset['Y'], dtype=np.float32)
    edge_index = torch.as_tensor(dataset['edge_index'], dtype=torch.long)
    graphs = []
    
    for i in range(len(X)):
        x_state = X[i]
        bus_type = build_bus_type_features(x_state)
        
        # Model expects:
        # [4 bus-type features, 4 state features, 4 mask features]
        model_x = np.concatenate(
            [bus_type, x_state[:, :4], x_state[:, 4:8]], axis=1)
        
        graphs.append(
            Data(
                x=torch.from_numpy(model_x),
                y=torch.from_numpy(Y[i]),
                edge_index=edge_index.clone(),
                edge_attr=torch.as_tensor(
                    dataset['edge_attr'][i], dtype=torch.float32)))
    return graphs

def fix_undirect_graph_bug():
    """
    The supplied PowerFlowNet_MPN.py returns None from undirect_graph() when
    the graph is already bidirectional. data_generation1.py already stores
    both directions, so replace it with a version that always returns the
    graph and attributes.
    """
    def undirect_graph(cls, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return edge_index, edge_attr

        pairs = set(zip(
            edge_index[0].detach().cpu().tolist(),
            edge_index[1].detach().cpu().tolist(),
        ))

        is_bidirectional = all(
            (dst, src) in pairs
            for src, dst in pairs
        )

        if is_bidirectional:
            return edge_index, edge_attr

        reverse_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0
        )
        edge_index = torch.cat(
            [edge_index, reverse_edge_index], dim=1
        )
        edge_attr = torch.cat(
            [edge_attr, edge_attr], dim=0
        )
        return edge_index, edge_attr

    PowerFlowNetMPN.undirect_graph = classmethod(undirect_graph)
    
def weighted_unknown_mse(pred, target, data, loss_weights):
    """
    Weighted MSE over unknown variables only.

    data.x[:, -4:] contains [mP, mV, mQ, mTheta].
    """
    mask = data.x[:, -4:]
    unknown = 1.0 - mask

    weights = torch.as_tensor(
        loss_weights, dtype=pred.dtype, device=pred.device
    ).view(1, 4)

    squared_error = (pred - target) ** 2
    weighted = squared_error * unknown * weights

    denominator = (unknown * weights).sum().clamp_min(1.0)
    return weighted.sum() / denominator

def run_epoch(model, loader, optimizer, device, loss_weights, train):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for data in loader:
            data = data.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            pred = model(data)
            loss = weighted_unknown_mse(
                pred, data.y, data, loss_weights
            )

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)

def evaluate(model, loader, device, loss_weights):
    """Return unknown-only MSE and MAE over the complete dataset."""
    model.eval()

    weights = torch.as_tensor(
        loss_weights, dtype=torch.float32, device=device
    ).view(1, 4)

    total_squared = torch.zeros(4, device=device)
    total_absolute = torch.zeros(4, device=device)
    total_weight = torch.zeros(4, device=device)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)

            unknown = 1.0 - data.x[:, -4:]
            error = pred - data.y

            total_squared += ((error ** 2) * unknown).sum(dim=0)
            total_absolute += (error.abs() * unknown).sum(dim=0)
            total_weight += unknown.sum(dim=0)

    mse = total_squared / total_weight.clamp_min(1.0)
    mae = total_absolute / total_weight.clamp_min(1.0)

    weighted_mse = (mse * weights.squeeze(0)).sum() / weights.sum()

    return weighted_mse.item(), mse.cpu().numpy(), mae.cpu().numpy()


def train_model(
    model,
    train_dataset,
    save_filepath,
    val_dataset=None,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    device="cpu",
    loss_weights=(1, 1, 1, 1),
    early_stopping=False,
    patience=20,
    num_workers=0,
):
    """
    Train PowerFlowNetMPN.

    This follows the supplied older train_model() structure, but uses
    torch_geometric Data/DataLoader because PowerFlowNetMPN.forward()
    accepts a PyG Data object and each sample has its own edge_attr.
    """
    model = model.to(device)

    train_graphs = make_pyg_dataset(train_dataset)
    val_graphs = make_pyg_dataset(val_dataset) if val_dataset is not None else None

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = None
    if val_graphs is not None:
        val_loader = DataLoader(
            val_graphs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            train=True,
        )
        loss_history.append(train_loss)

        message = f"Epoch {epoch + 1:4d}/{epochs} | train loss: {train_loss:.8e}"

        if early_stopping and val_loader is not None:
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                loss_weights=loss_weights,
                train=False,
            )
            val_loss_history.append(val_loss)
            message += f" | val loss: {val_loss:.8e}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(message + " | early stopping")
                break

        print(message)

    if early_stopping and val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(save_filepath)), exist_ok=True)
    torch.save(model.state_dict(), save_filepath)

    if early_stopping and val_loader is not None:
        return model, loss_history, val_loss_history

    return model, loss_history


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
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
    return data

def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    
    n_samples = 100

    train_data_filepath = os.path.join(
        data_dir, f'case14_PowerFlowNet_{n_samples}samples_train.npy')
    val_data_filepath = os.path.join(
        data_dir, f'case14_PowerFlowNet_{n_samples}samples_val.npy')
    test_data_filepath = os.path.join(
        data_dir, f'case14_PowerFlowNet_{n_samples}samples_test.npy')

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_filepath = os.path.join(
        models_dir, f'case14_PowerFlowNet_{n_samples}samples_mdl.pt')

    results_dir = os.path.join(base_dir, 'results')
    results_folderpath = os.path.join(
        results_dir, f'case14_PowerFlowNet_{n_samples}samples')
    os.makedirs(results_folderpath, exist_ok=True)

    results_filepath = os.path.join(
        results_folderpath, 'case14_results.npy')
    train_loss_history_filepath = os.path.join(
        results_folderpath, 'train_loss_history.npy')
    val_loss_history_filepath = os.path.join(
        results_folderpath, 'val_loss_history.npy')

    train_data = load_dataset(train_data_filepath)
    val_data = load_dataset(val_data_filepath)
    test_data = load_dataset(test_data_filepath)

    # -----------------------------------------------------------------------
    # Model and Training Parameters
    # -----------------------------------------------------------------------
    nfeature_dim = 4
    efeature_dim = 2
    output_dim = 4

    hidden_dim = 128
    num_layers = 4
    K = 3
    dropout_rate = 0.0

    epochs = 500
    lr = 1e-3
    batch_size = 32
    loss_weights = [1, 1, 1, 1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PowerFlowNetMPN(
        nfeature_dim=nfeature_dim,
        efeature_dim=efeature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=num_layers,
        K=K,
        dropout_rate=dropout_rate)

    P_loss_weights = [1, 0, 0, 0]
    V_loss_weights = [0, 1, 0, 0]
    Q_loss_weights = [0, 0, 1, 0]
    Theta_loss_weights = [0, 0, 0, 1]

    train = True

    # The supplied model expects a PyG Data object. The generated data already
    # contains both directions for every branch, so patch the model's graph
    # helper before training/evaluation.
    fix_undirect_graph_bug()

    if train:
        model, train_loss_history, val_loss_history = train_model(
            model,
            train_data,
            model_filepath,
            early_stopping=True,
            val_dataset=val_data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            loss_weights=loss_weights)

        np.save(
            train_loss_history_filepath,
            train_loss_history)

        np.save(
            val_loss_history_filepath,
            val_loss_history)

    model.load_state_dict(
        torch.load(
            model_filepath,
            map_location=device,
            weights_only=True))

    train_loss_history = np.load(train_loss_history_filepath)
    val_loss_history = np.load(val_loss_history_filepath)

    # -----------------------------------------------------------------------
    # Test
    # -----------------------------------------------------------------------
    test_graphs = make_pyg_dataset(test_data)
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False)

    weighted_test_loss, mse_pu, mae_pu = evaluate(
        model,
        test_loader,
        device,
        loss_weights)

    # Use the original X mask, as in the older model.
    mask = test_data['X'][:, :, 4:8].astype(bool)
    bases = test_data['bases']

    print('Per-Unit Metrics:\n')
    print(f'Weighted MSE: {weighted_test_loss:.8e}\n')

    labels = ['P', 'V', 'Q', 'Theta']
    for i, label in enumerate(labels):
        print(
            f'{label:>5s} | '
            f'MSE: {mse_pu[i]:.8e} | '
            f'MAE: {mae_pu[i]:.8e}')

    # The model predictions/targets can be reconstructed here for the
    # same per-unit -> absolute-unit comparison used by the older model.
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            preds.append(model(data).cpu().numpy())
            targets.append(data.y.cpu().numpy())

    preds_pu = np.concatenate(preds, axis=0)
    targets_pu = np.concatenate(targets, axis=0)
    
    # PyG batches all nodes from all samples into one node dimension.
    # Restore [samples, buses, 4] before using the dataset's unit conversion.
    n_samples = len(test_data['X'])
    n_buses = test_data['X'].shape[1]
    
    preds_pu = preds_pu.reshape(n_samples, n_buses, 4)
    targets_pu = targets_pu.reshape(n_samples, n_buses, 4)
    
    # The model only predicts unknown quantities. Replace the predictions
    # for known quantities with their exact target values.
    mask = test_data['X'][:, :, 4:8].astype(bool)
    preds_pu[mask] = targets_pu[mask]

    preds_absolute = convert_to_absolute(preds_pu, bases)
    targets_absolute = convert_to_absolute(targets_pu, bases)

    # Absolute-unit metrics, computed using the same unknown-variable mask.
    unknown = (~mask).astype(np.float32)

    squared_absolute = (
        (preds_absolute - targets_absolute) ** 2
        * unknown)
    absolute_error = (
        np.abs(preds_absolute - targets_absolute)
        * unknown)

    denominator = np.maximum(unknown.sum(axis=(0, 1)), 1.0)

    mse_absolute = squared_absolute.sum(axis=(0, 1)) / denominator
    mae_absolute = absolute_error.sum(axis=(0, 1)) / denominator

    print('\nAbsolute Metrics:\n')
    for i, label in enumerate(labels):
        print(
            f'{label:>5s} | '
            f'MSE: {mse_absolute[i]:.8e} | '
            f'MAE: {mae_absolute[i]:.8e}')

    # Save the main test results in the same spirit as the older script.
    results = {
        'preds_pu': preds_pu,
        'targets_pu': targets_pu,
        'preds_absolute': preds_absolute,
        'targets_absolute': targets_absolute,
        'mask': mask,
        'mse_pu': mse_pu,
        'mae_pu': mae_pu,
        'mse_absolute': mse_absolute,
        'mae_absolute': mae_absolute,
        'weighted_test_loss': weighted_test_loss,
    }

    np.save(results_filepath, results, allow_pickle=True)

    print(f'\nSaved model: {model_filepath}')
    print(f'Saved results: {results_filepath}')
    print(f'Saved train loss history: {train_loss_history_filepath}')
    print(f'Saved validation loss history: {val_loss_history_filepath}')


if __name__ == "__main__":
    main()
    
