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
    "lambda_spatial_smooth": 0.1,

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
      - center_seq      : (seq_len, 3)        — the target cell's P/U/V history
      - neighbor_seq    : (seq_len, k, 3)     — its k neighbors' P/U/V history
      - target          : (pred_steps, 3)     — ground truth future values
      - neighbor_targets: (k, pred_steps, 3)  — ground truth future values for neighbors
      - neighbor_dists  : (k,)                — spatial distances to each neighbor
    """

    def __init__(self, data_normalized, neighbor_indices, neighbor_distances,
                 seq_length, pred_steps, start_t, end_t, device=None,
                 preload_to_gpu=False, cache_neighbors=False,
                 cache_neighbors_device="cuda", cache_vram_fraction=0.70):
        """
        Parameters
        ----------
        data_normalized    : np.ndarray, shape (n_cells, n_timesteps, 3)
        neighbor_indices   : np.ndarray, shape (n_cells, k)
        neighbor_distances : np.ndarray, shape (n_cells, k)
        seq_length         : int
        pred_steps         : int
        start_t            : int — inclusive start of valid time range
        end_t              : int — exclusive end of valid time range
        device             : torch.device
        preload_to_gpu     : bool
        cache_neighbors    : bool
        cache_neighbors_device : "cuda" or "cpu"
        cache_vram_fraction    : float (max fraction of VRAM for cache)
        """
        super().__init__()
        self.device = device

        data_tensor = torch.FloatTensor(data_normalized)
        nbr_tensor = torch.LongTensor(neighbor_indices)
        dist_tensor = torch.FloatTensor(neighbor_distances)

        if preload_to_gpu and device is not None:
            data_tensor = data_tensor.to(device)
            nbr_tensor = nbr_tensor.to(device)
            dist_tensor = dist_tensor.to(device)

        self.data = data_tensor
        self.neighbor_indices = nbr_tensor
        self.neighbor_distances = dist_tensor
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

        nbr_indices = self.neighbor_indices[cell_idx]
        neighbor_targets = self.data[nbr_indices,
                                     t_start + self.seq_length:
                                     t_start + self.seq_length + self.pred_steps, :]

        neighbor_dists = self.neighbor_distances[cell_idx]

        return center_seq, neighbor_seq, target, neighbor_targets, neighbor_dists


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
    neighbor_indices   : np.ndarray, shape (n_cells, k)
    neighbor_distances : np.ndarray, shape (n_cells, k)
    """
    print(f"Building kNN graph with k={k} ...")
    tree = cKDTree(coords)
    # Query k+1 because the first result is the point itself
    distances, indices = tree.query(coords, k=k + 1)
    neighbor_indices = indices[:, 1:]    # exclude self
    neighbor_distances = distances[:, 1:]
    print(f"  neighbor_indices shape: {neighbor_indices.shape}")
    return neighbor_indices, neighbor_distances


def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


# ── Distance-Weighted Spatial Encoder ────────────────────────────────────────
class DistanceWeightedSpatialEncoder(nn.Module):
    """
    Encodes k-nearest-neighbor flow variables into a spatial embedding using
    distance-weighted aggregation instead of naive mean-pooling.

    A learned distance-scaling network determines effective neighbor weights so
    that closer neighbors contribute more to the spatial representation, preserving
    spatial gradient information across the mesh.
    """

    def __init__(self, input_features=3, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dist_scale = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def forward(self, neighbor_seq, neighbor_dists):
        # neighbor_seq   : (B, seq_len, k, input_features)
        # neighbor_dists : (B, k)
        embedded = self.mlp(neighbor_seq)               # (B, seq_len, k, embed_dim)
        d = neighbor_dists.unsqueeze(-1)                # (B, k, 1)
        raw_w = self.dist_scale(d)                      # (B, k, 1)
        inv_w = 1.0 / (raw_w + 1e-6)                   # closer → larger weight
        weights = inv_w / (inv_w.sum(dim=1, keepdim=True) + 1e-8)  # (B, k, 1)
        weights = weights.unsqueeze(1)                  # (B, 1, k, 1)
        pooled = (embedded * weights).sum(dim=2)        # (B, seq_len, embed_dim)
        return pooled


# ── Model Definitions ─────────────────────────────────────────────────────────
class SpatioTemporalLSTM(nn.Module):
    """
    Spatiotemporal LSTM:
      1. DistanceWeightedSpatialEncoder aggregates neighbor features → spatial embedding
      2. Concatenate center-cell features with spatial embedding
      3. LSTM processes the combined sequence
      4. Decoder MLP maps last hidden state → (pred_steps, 3)
    """

    def __init__(self, input_features=3, spatial_embed_dim=64, hidden_size=256,
                 num_layers=3, pred_steps=15, dropout=0.2):
        super().__init__()
        self.pred_steps = pred_steps
        self.input_features = input_features
        self.spatial_encoder = DistanceWeightedSpatialEncoder(input_features, spatial_embed_dim)
        combined_dim = input_features + spatial_embed_dim
        self.lstm = nn.LSTM(
            combined_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_steps * input_features),
        )

    def forward(self, center_seq, neighbor_seq, neighbor_dists):
        spatial_embed = self.spatial_encoder(neighbor_seq, neighbor_dists)
        combined = torch.cat([center_seq, spatial_embed], dim=-1)
        lstm_out, _ = self.lstm(combined)
        last_hidden = lstm_out[:, -1, :]
        output = self.decoder(last_hidden)
        return output.view(-1, self.pred_steps, self.input_features)


class SpatioTemporalTransformer(nn.Module):
    """
    Spatiotemporal Transformer:
      1. DistanceWeightedSpatialEncoder aggregates neighbor features → spatial embedding
      2. Concatenate center-cell features with spatial embedding
      3. Linear projection to d_model + learned positional encoding
      4. Transformer encoder processes the sequence
      5. Decoder MLP maps last-position output → (pred_steps, 3)
    """

    def __init__(self, input_features=3, spatial_embed_dim=64, d_model=256,
                 nhead=8, num_layers=3, pred_steps=15, dropout=0.2):
        super().__init__()
        self.pred_steps = pred_steps
        self.input_features = input_features
        self.spatial_encoder = DistanceWeightedSpatialEncoder(input_features, spatial_embed_dim)
        combined_dim = input_features + spatial_embed_dim
        self.input_projection = nn.Linear(combined_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_steps * input_features),
        )

    def forward(self, center_seq, neighbor_seq, neighbor_dists):
        spatial_embed = self.spatial_encoder(neighbor_seq, neighbor_dists)
        combined = torch.cat([center_seq, spatial_embed], dim=-1)
        x = self.input_projection(combined)
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        last_output = x[:, -1, :]
        output = self.decoder(last_output)
        return output.view(-1, self.pred_steps, self.input_features)


def create_model(cfg):
    """Factory function — create model by name."""
    model_type = cfg["model_type"]
    pred_steps = cfg["prediction_horizon_steps"]
    spatial_embed_dim = cfg["spatial_embed_dim"]
    hidden_size = cfg["hidden_size"]
    num_layers = cfg["num_layers"]
    dropout = cfg["dropout"]

    if model_type == "SpatioTemporalLSTM":
        model = SpatioTemporalLSTM(
            input_features=3,
            spatial_embed_dim=spatial_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            pred_steps=pred_steps,
            dropout=dropout,
        )
    elif model_type == "SpatioTemporalTransformer":
        model = SpatioTemporalTransformer(
            input_features=3,
            spatial_embed_dim=spatial_embed_dim,
            d_model=hidden_size,
            nhead=cfg["nhead"],
            num_layers=num_layers,
            pred_steps=pred_steps,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Choose 'SpatioTemporalLSTM' or 'SpatioTemporalTransformer'."
        )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created {model_type} with {n_params:,} trainable parameters.")
    return model


# ── Spatial Smoothness Loss ───────────────────────────────────────────────────
def spatial_smoothness_loss(
    predictions: torch.Tensor,
    neighbor_targets: torch.Tensor,
    neighbor_dists: torch.Tensor,
) -> torch.Tensor:
    """
    Penalizes large prediction differences between a cell and its spatial neighbors.

    Weights discrepancies by proximity so that closer neighbors contribute more
    to the regularization signal, encouraging smooth pressure gradients.

    Parameters
    ----------
    predictions      : (B, pred_steps, 3)     — model outputs for center cells
    neighbor_targets : (B, k, pred_steps, 3)  — ground truth future for each neighbor
    neighbor_dists   : (B, k)                 — spatial distances to each neighbor

    Returns
    -------
    loss : scalar tensor
    """
    diff_sq = (predictions.unsqueeze(1) - neighbor_targets) ** 2   # (B, k, pred_steps, 3)
    inv_d = 1.0 / (neighbor_dists + 1e-6)                          # (B, k)
    weights = inv_d / (inv_d.sum(dim=1, keepdim=True) + 1e-8)      # (B, k), normalized
    weights = weights.unsqueeze(-1).unsqueeze(-1)                   # (B, k, 1, 1)
    return (diff_sq * weights).sum(dim=1).mean()


def main(args):
    # Setup for distributed training
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare the data
    data = load_data(args.csv_files)
    data = normalize_data(data)

    # Build kNN graph — returns both indices and distances
    neighbor_indices, neighbor_distances = build_knn_graph(data[:, :2], CONFIG["k_neighbors"])

    dataset = SpatioTemporalDataset(
        data_normalized=data,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
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
    model = create_model(CONFIG)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    mse_criterion = nn.MSELoss()
    lambda_smooth = CONFIG["lambda_spatial_smooth"]

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            center_seq, neighbor_seq, target, neighbor_targets, neighbor_dists = batch
            center_seq = center_seq.to(device)
            neighbor_seq = neighbor_seq.to(device)
            target = target.to(device)
            neighbor_targets = neighbor_targets.to(device)
            neighbor_dists = neighbor_dists.to(device)

            with autocast(enabled=CONFIG["use_amp"]):
                outputs = model(center_seq, neighbor_seq, neighbor_dists)
                mse_loss = mse_criterion(outputs, target)
                smooth_loss = spatial_smoothness_loss(outputs, neighbor_targets, neighbor_dists)
                loss = mse_loss + lambda_smooth * smooth_loss

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
