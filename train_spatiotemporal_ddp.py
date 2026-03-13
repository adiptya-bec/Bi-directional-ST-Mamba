import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler, autocast
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Data paths
    "pressure_csv": "/kaggle/input/datasets/adiptyaghosh/master3sec-filtered-0-3-0-6/master_pressure_3sec_nonzero_filtered.csv",
    "u_velocity_csv": "/kaggle/input/datasets/adiptyaghosh/master3sec-filtered-0-3-0-6/master_x_velocity_3sec_nonzero_filtered.csv",
    "v_velocity_csv": "/kaggle/input/datasets/adiptyaghosh/master3sec-filtered-0-3-0-6/master_y_velocity_3sec_nonzero_filtered.csv",
    "chunk_size": None,
    "timestep_duration": 0.02,

    # Spatial
    "k_neighbors": 20,

    # Temporal
    "input_sequence_length": 15,
    "prediction_horizon_steps": 5,

    # Model
    "model_type": "SpatioTemporalTransformer",
    "spatial_embed_dim": 64,
    "hidden_size": 256,
    "num_layers": 3,
    "nhead": 8,
    "dropout": 0.2,

    # Training
    "batch_size": 128,
    "learning_rate": 5e-4,
    "epochs": 80,
    "patience": 10,
    "grad_clip": 0.5,
    "loss_weights": {"pressure": 0.3, "u_velocity": 0.35, "v_velocity": 0.35},

    # Speed optimizations
    "use_amp": True,
    "preload_to_gpu": True,
    "cache_neighbors": True,
    "cache_neighbors_device": "cuda",
    "cache_vram_fraction": 0.70,

    # DataLoader tuning
    "num_workers": 2,
    "pin_memory": True,

    # Split ratios
    "train_ratio": 0.60,
    "val_ratio": 0.20,

    # Output
    "output_dir": "outputs",
    "checkpoint_dir": "checkpoints",
}

# ── cuDNN benchmark ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ── Helper: estimate VRAM needed for neighbor cache ───────────────────────────
def estimate_neighbor_cache_gb(n_cells, k, n_timesteps, n_vars=3, dtype_bytes=4):
    bytes_needed = n_cells * k * n_timesteps * n_vars * dtype_bytes
    return bytes_needed / (1024 ** 3)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SpatioTemporalDataset(Dataset):
    """
    Sliding-window dataset for spatiotemporal flow forecasting.

    Each sample contains:
      - center_seq   : (seq_len, 3)     — the target cell's P/U/V history
      - neighbor_seq : (seq_len, k, 3)  — its k neighbors' P/U/V history
      - target       : (pred_steps, 3)  — ground truth future values
    """

    def __init__(self, data_normalized, neighbor_indices, seq_length, pred_steps,
                 start_t, end_t, device=None, preload_to_gpu=False,
                 cache_neighbors=False, cache_neighbors_device="cuda",
                 cache_vram_fraction=0.70):
        """
        Parameters
        ----------
        data_normalized  : np.ndarray, shape (n_cells, n_timesteps, 3)
        neighbor_indices : np.ndarray, shape (n_cells, k)
        seq_length       : int
        pred_steps       : int
        start_t          : int — inclusive start of valid time range
        end_t            : int — exclusive end of valid time range
        device           : torch.device
        preload_to_gpu   : bool
        cache_neighbors  : bool
        cache_neighbors_device : "cuda" or "cpu"
        cache_vram_fraction    : float (max fraction of VRAM for cache)
        """
        super().__init__()
        self.device = device

        data_tensor = torch.FloatTensor(data_normalized)
        nbr_tensor = torch.LongTensor(neighbor_indices)

        if preload_to_gpu and device is not None:
            data_tensor = data_tensor.to(device)
            nbr_tensor = nbr_tensor.to(device)

        self.data = data_tensor
        self.neighbor_indices = nbr_tensor
        self.seq_length = seq_length
        self.pred_steps = pred_steps
        self.start_t = start_t

        self.n_cells = data_normalized.shape[0]
        self.n_valid_windows = max(0, end_t - start_t - seq_length - pred_steps + 1)

        self.cache_neighbors = cache_neighbors
        self.cache_neighbors_device = cache_neighbors_device

        if cache_neighbors:
            if cache_neighbors_device == "cuda" and device is not None and torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                est_gb = estimate_neighbor_cache_gb(self.n_cells, neighbor_indices.shape[1], data_normalized.shape[1])
                print(f"Estimated neighbor cache: {est_gb:.2f} GB (VRAM total: {total_gb:.2f} GB)")
                if est_gb > total_gb * cache_vram_fraction:
                    print("[Warning] Neighbor cache too large for GPU; falling back to CPU cache.")
                    self.cache_neighbors_device = "cpu"

            self.neighbor_data = self.data[self.neighbor_indices]
            if self.cache_neighbors_device == "cuda" and device is not None:
                if not self.neighbor_data.is_cuda:
                    self.neighbor_data = self.neighbor_data.to(device)
            else:
                self.neighbor_data = self.neighbor_data.cpu()

    def __len__(self):
        return self.n_cells * self.n_valid_windows

    def __getitem__(self, idx):
        cell_idx = idx // self.n_valid_windows
        window_idx = idx % self.n_valid_windows
        t_start = self.start_t + window_idx

        center_seq = self.data[cell_idx, t_start:t_start + self.seq_length, :]

        if self.cache_neighbors:
            neighbor_seq = self.neighbor_data[cell_idx, :, t_start:t_start + self.seq_length, :]
        else:
            nbr_indices = self.neighbor_indices[cell_idx]
            neighbor_seq = self.data[nbr_indices, t_start:t_start + self.seq_length, :]

        neighbor_seq = neighbor_seq.permute(1, 0, 2)
        target = self.data[cell_idx,
                           t_start + self.seq_length:
                           t_start + self.seq_length + self.pred_steps, :]

        return center_seq, neighbor_seq, target


def load_data(csv_files):
    data = []
    for file in csv_files:
        df = pd.read_csv(file)
        data.append(df.values)
    return np.concatenate(data)


def build_knn_graph(coords, k):
    """
    For each cell, find its k nearest spatial neighbors (excluding self).

    Parameters
    ----------
    coords : np.ndarray, shape (n_cells, 2)
    k      : int — number of neighbors

    Returns
    -------
    neighbor_indices : np.ndarray, shape (n_cells, k)
    """
    print(f"Building kNN graph with k={k} ...")
    tree = cKDTree(coords)
    # Query k+1 because the first result is the point itself
    _, indices = tree.query(coords, k=k + 1)
    neighbor_indices = indices[:, 1:]  # exclude self
    print(f"  neighbor_indices shape: {neighbor_indices.shape}")
    return neighbor_indices


def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


def main(args):
    # Setup for distributed training
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare the data
    data = load_data(args.csv_files)
    data = normalize_data(data)
    dataset = SpatioTemporalDataset(
        data_normalized=data,
        neighbor_indices=build_knn_graph(data[:, :2], CONFIG["k_neighbors"]),
        seq_length=CONFIG["input_sequence_length"],
        pred_steps=CONFIG["prediction_horizon_steps"],
        start_t=0,
        end_t=data.shape[1],
        device=device,
        preload_to_gpu=CONFIG["preload_to_gpu"],
        cache_neighbors=CONFIG["cache_neighbors"],
        cache_neighbors_device=CONFIG["cache_neighbors_device"],
        cache_vram_fraction=CONFIG["cache_vram_fraction"],
    )
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    # Model initialization
    model = SpatioTemporalTransformer()  # This should be defined
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            with autocast(enabled=CONFIG["use_amp"]):
                outputs = model(batch.to(device))
                loss = criterion(outputs)  # Define criterion
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step >= args.max_train_steps_per_epoch:
                break

    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SpatioTemporal model with DDP')
    parser.add_argument('--csv_files', nargs='+',
                        default=[CONFIG["pressure_csv"], CONFIG["u_velocity_csv"], CONFIG["v_velocity_csv"]],
                        help='Input CSV files')
    parser.add_argument('--window_size', type=int, default=10, help='Size of sliding window')
    parser.add_argument('--batch_size', type=int, default=CONFIG["batch_size"], help='Batch size')
    parser.add_argument('--epochs', type=int, default=CONFIG["epochs"], help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=CONFIG["learning_rate"], help='Learning rate')
    parser.add_argument('--max_train_steps_per_epoch', type=int, default=100, help='Max steps per epoch')
    parser.add_argument('--save_model', action='store_true', help='Save the model after training')
    args = parser.parse_args()
    main(args)
