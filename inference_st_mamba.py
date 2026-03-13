#!/usr/bin/env python3
"""
Inference pipeline for the Bi-Directional Spatial-Temporal Mamba (ST-Mamba) model.

Performs autoregressive prediction for a user-specified number of future
timesteps and exports the results to CSV.

Usage
-----
  # Using a trained checkpoint and real data
  python inference_st_mamba.py \
      --checkpoint checkpoints/best_checkpoint.pth \
      --pressure_csv  data/pressure.csv \
      --u_velocity_csv data/u_velocity.csv \
      --v_velocity_csv data/v_velocity.csv \
      --n_steps 20

  # Quick test with automatically generated mock data
  python inference_st_mamba.py --n_steps 5 --use_mock
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Model and data configuration — must match the checkpoint's training config."""

    # ── Data paths ─────────────────────────────────────────────────────────────
    pressure_path: str = ""
    u_vel_path:    str = ""
    v_vel_path:    str = ""
    output_dir:    str = "inference_outputs"
    checkpoint_dir: str = "checkpoints"

    # ── Mesh / patch ───────────────────────────────────────────────────────────
    n_nodes:    int = 800_000
    patch_size: int = 64
    n_channels: int = 5          # p, u, v, x_coord, y_coord

    # ── Temporal window ────────────────────────────────────────────────────────
    time_in:       int = 20      # input window length (must match training)
    time_out:      int = 1       # single-step prediction (must match training)
    rollout_steps: int = 10      # overridden by --n_steps

    # ── Train / val / test split ───────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio:   float = 0.15

    # ── Model architecture (must match training) ──────────────────────────────
    hidden_dim:    int   = 128
    spatial_layers: int  = 2
    temporal_layers: int = 2
    dropout:       float = 0.15
    mamba_d_state: int   = 16
    mamba_d_conv:  int   = 4
    mamba_expand:  int   = 2

    # ── Inference ──────────────────────────────────────────────────────────────
    use_amp: bool = True
    use_checkpointing: bool = False  # not needed at inference time
    batch_size: int = 1

    # ── Loss (unused at inference but required by Config) ──────────────────────
    epochs: int = 80
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 15_000
    grad_clip: float = 0.5
    spectral_weight: float = 0.10
    energy_weight: float = 0.05
    use_spatial_weighting: bool = True
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "pressure": 0.30,
            "u_velocity": 0.35,
            "v_velocity": 0.35,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hilbert-Curve Sorting
# ─────────────────────────────────────────────────────────────────────────────
def _xy2d(n: int, x: int, y: int) -> int:
    """Convert (x, y) grid coordinates to Hilbert-curve distance."""
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def hilbert_sort_indices(coords: np.ndarray, order: int = 10) -> np.ndarray:
    """Return permutation indices that sort *coords* (N×2) along a 2-D Hilbert curve."""
    grid = 1 << order
    xy = coords.copy().astype(np.float64)
    for dim in range(2):
        lo, hi = xy[:, dim].min(), xy[:, dim].max()
        rng = hi - lo
        if rng < 1e-12:
            xy[:, dim] = 0.0
        else:
            xy[:, dim] = (xy[:, dim] - lo) / rng * (grid - 1)
    ix = xy[:, 0].astype(np.int64).clip(0, grid - 1)
    iy = xy[:, 1].astype(np.int64).clip(0, grid - 1)
    keys = np.array(
        [_xy2d(grid, int(ix[i]), int(iy[i])) for i in range(len(ix))],
        dtype=np.int64,
    )
    return np.argsort(keys)


# ─────────────────────────────────────────────────────────────────────────────
# Data Normalizer
# ─────────────────────────────────────────────────────────────────────────────
class DataNormalizer:
    """Per-variable StandardScaler; coordinates normalised to [0, 1]."""

    def __init__(self) -> None:
        self.scalers: Dict[str, StandardScaler] = {
            v: StandardScaler() for v in ("pressure", "u_velocity", "v_velocity")
        }
        self.coord_min: Optional[np.ndarray] = None
        self.coord_max: Optional[np.ndarray] = None
        self.coord_scale: Optional[np.ndarray] = None

    def fit(
        self,
        pressure_train: np.ndarray,
        u_train: np.ndarray,
        v_train: np.ndarray,
        coords: np.ndarray,
    ) -> None:
        self.scalers["pressure"].fit(pressure_train.reshape(-1, 1))
        self.scalers["u_velocity"].fit(u_train.reshape(-1, 1))
        self.scalers["v_velocity"].fit(v_train.reshape(-1, 1))
        self.coord_min = coords.min(axis=0)
        self.coord_max = coords.max(axis=0)
        rng = self.coord_max - self.coord_min
        rng[rng < 1e-12] = 1.0
        self.coord_scale = rng

    def transform_var(self, arr: np.ndarray, name: str) -> np.ndarray:
        shape = arr.shape
        return self.scalers[name].transform(arr.reshape(-1, 1)).reshape(shape)

    def inverse_transform_var(self, arr: np.ndarray, name: str) -> np.ndarray:
        shape = arr.shape
        return self.scalers[name].inverse_transform(arr.reshape(-1, 1)).reshape(shape)

    def transform_coords(self, coords: np.ndarray) -> np.ndarray:
        return (coords - self.coord_min) / self.coord_scale

    def fit_transform(
        self,
        pressure: np.ndarray,
        u_vel: np.ndarray,
        v_vel: np.ndarray,
        coords: np.ndarray,
        train_end_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.fit(
            pressure[:, :train_end_idx],
            u_vel[:, :train_end_idx],
            v_vel[:, :train_end_idx],
            coords,
        )
        p_n = self.transform_var(pressure, "pressure")
        u_n = self.transform_var(u_vel, "u_velocity")
        v_n = self.transform_var(v_vel, "v_velocity")
        c_n = self.transform_coords(coords)
        return p_n, u_n, v_n, c_n


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_data(
    pressure_path: str, u_vel_path: str, v_vel_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load three variable CSVs (x, y, t0, t1, …) → coords, pressure, u, v."""
    print("Loading CSVs …")
    df_p = pd.read_csv(pressure_path, header=0)
    df_u = pd.read_csv(u_vel_path, header=0)
    df_v = pd.read_csv(v_vel_path, header=0)
    coords = df_p.iloc[:, :2].values.astype(np.float32)
    pressure = df_p.iloc[:, 2:].values.astype(np.float32)
    u_vel = df_u.iloc[:, 2:].values.astype(np.float32)
    v_vel = df_v.iloc[:, 2:].values.astype(np.float32)
    T = min(pressure.shape[1], u_vel.shape[1], v_vel.shape[1])
    pressure, u_vel, v_vel = pressure[:, :T], u_vel[:, :T], v_vel[:, :T]
    print(f"  Loaded — nodes: {coords.shape[0]:,}, timesteps: {T}")
    return coords, pressure, u_vel, v_vel


def make_mock_data(
    n_nodes: int = 10_000, n_timesteps: int = 40, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic CFD-like data for testing."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 1, (n_nodes, 2)).astype(np.float32)
    n_bl = n_nodes // 10
    base[:n_bl, 1] = rng.uniform(0, 0.05, n_bl)
    coords = base
    t = np.linspace(0, 2 * np.pi, n_timesteps, dtype=np.float32)
    x, y = coords[:, 0:1], coords[:, 1:2]
    pressure = (
        100_000
        + 5_000 * np.sin(2 * np.pi * x)
        + 3_000 * np.cos(2 * np.pi * y)
        + 1_000 * np.sin(t[np.newaxis, :])
        + rng.normal(0, 100, (n_nodes, n_timesteps)).astype(np.float32)
    ).astype(np.float32)
    u_vel = (
        20
        + 10 * np.cos(np.pi * y)
        + 2 * np.sin(t[np.newaxis, :])
        + rng.normal(0, 0.5, (n_nodes, n_timesteps)).astype(np.float32)
    ).astype(np.float32)
    v_vel = (
        5 * np.sin(np.pi * x)
        + 1 * np.cos(t[np.newaxis, :])
        + rng.normal(0, 0.3, (n_nodes, n_timesteps)).astype(np.float32)
    ).astype(np.float32)
    print(f"  [Mock] Generated {n_nodes:,} nodes × {n_timesteps} timesteps.")
    return coords, pressure, u_vel, v_vel


# ─────────────────────────────────────────────────────────────────────────────
# Model Components (must match the architecture used during training)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mamba_ssm import Mamba as _MambaSSM

    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class MambaLikeBlock(nn.Module):
    """Single-direction SSM block (mamba_ssm or GRU fallback)."""

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ) -> None:
        super().__init__()
        if _MAMBA_AVAILABLE:
            self._impl = _MambaSSM(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            d_inner = d_model * expand
            self._impl = nn.Sequential(nn.Linear(d_model, d_inner), nn.GELU())
            self._gru = nn.GRU(d_inner, d_model, batch_first=True)
        self._use_mamba = _MAMBA_AVAILABLE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mamba:
            return self._impl(x)
        h = self._impl(x)
        out, _ = self._gru(h)
        return out


class BiDirectionalSpatialMamba(nn.Module):
    """Bi-directional Mamba applied along the spatial (patch) dimension."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.fwd = MambaLikeBlock(hidden_dim, d_state, d_conv, expand)
        self.bwd = MambaLikeBlock(hidden_dim, d_state, d_conv, expand)
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.use_ckpt = use_checkpointing

    def _spatial_pass(self, x_t: torch.Tensor) -> torch.Tensor:
        fwd_out = self.fwd(x_t)
        bwd_out = self.bwd(x_t.flip(1)).flip(1)
        cat = torch.cat([fwd_out, bwd_out], dim=-1)
        return self.drop(self.proj(cat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, H = x.shape
        residual = x
        out_list: List[torch.Tensor] = []
        for t in range(T):
            x_t = x[:, t]
            if self.use_ckpt and self.training:
                out_t = grad_ckpt.checkpoint(
                    self._spatial_pass, x_t, use_reentrant=False
                )
            else:
                out_t = self._spatial_pass(x_t)
            out_list.append(out_t)
        out = torch.stack(out_list, dim=1)
        return self.norm(out + residual)


class CausalTemporalMamba(nn.Module):
    """Causal (uni-directional) Mamba applied along the time dimension."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.mamba = MambaLikeBlock(hidden_dim, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.use_ckpt = use_checkpointing

    def _temporal_pass(self, x_p: torch.Tensor) -> torch.Tensor:
        return self.drop(self.mamba(x_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, H = x.shape
        residual = x
        x_p = x.permute(0, 2, 1, 3).reshape(B * P, T, H)
        if self.use_ckpt and self.training:
            out = grad_ckpt.checkpoint(
                self._temporal_pass, x_p, use_reentrant=False
            )
        else:
            out = self._temporal_pass(x_p)
        out = out.reshape(B, P, T, H).permute(0, 2, 1, 3)
        return self.norm(out + residual)


class STBlock(nn.Module):
    """One ST-Mamba block: BiDirectional Spatial → Causal Temporal."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.spatial = BiDirectionalSpatialMamba(
            hidden_dim, d_state, d_conv, expand, dropout, use_checkpointing
        )
        self.temporal = CausalTemporalMamba(
            hidden_dim, d_state, d_conv, expand, dropout, use_checkpointing
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        return x


class PatchEmbed(nn.Module):
    """Linear projection from (patch_size × n_channels) → hidden_dim."""

    def __init__(
        self,
        patch_size: int,
        n_channels: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(patch_size * n_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, S, C = x.shape
        x = x.reshape(B, T, P, S * C)
        x = self.drop(self.norm(self.proj(x)))
        return x


class PatchUnembed(nn.Module):
    """Project hidden_dim → (patch_size × 3)."""

    def __init__(
        self, patch_size: int, hidden_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, patch_size * 3)
        self.drop = nn.Dropout(dropout)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, H = x.shape
        x = self.drop(self.proj(x))
        return x.reshape(B, T, P, self.patch_size, 3)


class STMambaModel(nn.Module):
    """
    Full Bi-Directional Spatial-Temporal Mamba model.

    Input  : (B, T_in,  n_patches, patch_size, n_channels)
    Output : (B, T_out, n_patches, patch_size, 3)   — p, u, v
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = PatchEmbed(
            cfg.patch_size, cfg.n_channels, cfg.hidden_dim, cfg.dropout
        )
        self.blocks = nn.ModuleList(
            [
                STBlock(
                    cfg.hidden_dim,
                    cfg.mamba_d_state,
                    cfg.mamba_d_conv,
                    cfg.mamba_expand,
                    cfg.dropout,
                    cfg.use_checkpointing,
                )
                for _ in range(cfg.spatial_layers)
            ]
        )
        self.unembed = PatchUnembed(
            cfg.patch_size, cfg.hidden_dim, cfg.dropout
        )
        self.time_proj = nn.Linear(cfg.time_in, cfg.time_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = h.permute(0, 2, 3, 1)
        h = self.time_proj(h)
        h = h.permute(0, 3, 1, 2)
        out = self.unembed(h)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive Rollout
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def autoregressive_rollout(
    model: nn.Module,
    initial_window: torch.Tensor,
    rollout_steps: int,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Autoregressively predict *rollout_steps* future frames.

    At each step the model predicts the next single frame from the
    current input window.  That prediction is then appended to the
    window (replacing the oldest frame) and the process repeats.

    Parameters
    ----------
    model          : Trained STMambaModel (or DataParallel wrapper).
    initial_window : (1, T_in, n_patches, patch_size, n_channels)
    rollout_steps  : Number of future timesteps to predict.
    device         : Target device.
    use_amp        : Whether to use automatic mixed-precision.

    Returns
    -------
    predictions : (rollout_steps, n_patches, patch_size, 3)
    """
    model.eval()
    window = initial_window.to(device)
    all_preds: List[torch.Tensor] = []

    for step in range(rollout_steps):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(window)                      # (1, T_out, P, S, 3)

        pred_step = pred[:, 0:1]                      # (1, 1, P, S, 3)
        all_preds.append(pred_step[0, 0].cpu())       # (P, S, 3)

        # Slide window: drop oldest frame, append new prediction
        next_frame = window[:, -1:].clone()           # (1, 1, P, S, C)
        next_frame[..., :3] = pred_step               # overwrite p, u, v
        window = torch.cat([window[:, 1:], next_frame], dim=1)

    return torch.stack(all_preds, dim=0)              # (rollout_steps, P, S, 3)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Export
# ─────────────────────────────────────────────────────────────────────────────
def export_predictions_to_csv(
    predictions: np.ndarray,
    coords_orig: np.ndarray,
    normalizer: DataNormalizer,
    output_dir: str,
    prefix: str = "st_mamba_pred",
) -> List[str]:
    """
    Denormalise and write three CSV files (pressure, u_velocity, v_velocity).

    Parameters
    ----------
    predictions  : (rollout_steps, N_padded, 3) normalised predictions.
    coords_orig  : (N_real, 2) original mesh coordinates.
    normalizer   : Fitted DataNormalizer instance.
    output_dir   : Directory for output files.
    prefix       : Filename prefix.

    Returns
    -------
    paths : List of output CSV paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    N_real = len(coords_orig)
    R, N_pad, _ = predictions.shape
    N = min(N_real, N_pad)

    var_names = ["pressure", "u_velocity", "v_velocity"]
    file_stems = [
        f"{prefix}_pressure",
        f"{prefix}_u_velocity",
        f"{prefix}_v_velocity",
    ]
    paths: List[str] = []

    for v_idx, (v_name, stem) in enumerate(zip(var_names, file_stems)):
        pred_v = predictions[:, :N, v_idx]                           # (R, N)
        pred_v = normalizer.inverse_transform_var(pred_v.T, v_name)  # (N, R)

        df = pd.DataFrame(coords_orig[:N], columns=["x_coord", "y_coord"])
        for step in range(R):
            df[f"pred_t{step + 1}"] = pred_v[:, step]

        out_path = os.path.join(output_dir, f"{stem}.csv")
        df.to_csv(out_path, index=False)
        paths.append(out_path)
        print(f"  Exported: {out_path}  ({N:,} rows × {R} predicted timesteps)")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Seed Window Preparation
# ─────────────────────────────────────────────────────────────────────────────
def prepare_seed_window(
    data: np.ndarray,
    hilbert_idx: np.ndarray,
    patch_size: int,
    time_in: int,
    seed_start: int,
) -> Tuple[torch.Tensor, int]:
    """
    Build the initial input window for autoregressive rollout.

    Parameters
    ----------
    data        : (N, T, C) stacked normalised array (p, u, v, x_norm, y_norm).
    hilbert_idx : (N,) permutation from Hilbert sort.
    patch_size  : Spatial patch size.
    time_in     : Number of input timesteps the model expects.
    seed_start  : First timestep index for the seed window.

    Returns
    -------
    window    : (1, T_in, n_patches, patch_size, C) tensor.
    n_patches : Number of spatial patches (after padding).
    """
    data_sorted = data[hilbert_idx]
    N, T, C = data_sorted.shape

    n_patches = math.ceil(N / patch_size)
    pad_len = n_patches * patch_size - N
    if pad_len > 0:
        pad = np.zeros((pad_len, T, C), dtype=data_sorted.dtype)
        data_sorted = np.concatenate([data_sorted, pad], axis=0)

    # (n_patches, patch_size, T, C)
    patches = data_sorted.reshape(n_patches, patch_size, T, C)

    # Extract time window and transpose to model input format
    x = patches[:, :, seed_start : seed_start + time_in, :]  # (P, S, T_in, C)
    x = torch.from_numpy(x.transpose(2, 0, 1, 3)).float()   # (T_in, P, S, C)

    return x.unsqueeze(0), n_patches  # (1, T_in, P, S, C)


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    n_steps: int,
    checkpoint_path: Optional[str] = None,
    pressure_csv: Optional[str] = None,
    u_velocity_csv: Optional[str] = None,
    v_velocity_csv: Optional[str] = None,
    use_mock: bool = False,
    output_dir: str = "inference_outputs",
    seed_timestep: Optional[int] = None,
) -> Dict:
    """
    End-to-end inference pipeline for the ST-Mamba model.

    Parameters
    ----------
    n_steps        : Number of future timesteps to predict autoregressively.
    checkpoint_path: Path to a trained model checkpoint (.pth).
                     If *None*, a freshly initialised model is used (for testing).
    pressure_csv   : Path to the pressure CSV file.
    u_velocity_csv : Path to the u-velocity CSV file.
    v_velocity_csv : Path to the v-velocity CSV file.
    use_mock       : If True, generate synthetic data instead of loading CSVs.
    output_dir     : Directory where prediction CSVs are written.
    seed_timestep  : Starting timestep index for the seed window.  Defaults to
                     the first timestep of the test split.

    Returns
    -------
    results : dict with keys ``predictions_raw``, ``predictions_physical``,
              ``coords``, ``normalizer``, ``config``, ``csv_paths``.
    """
    cfg = Config(output_dir=output_dir, rollout_steps=n_steps)

    print("=" * 70)
    print("  ST-Mamba Inference Pipeline")
    print("=" * 70)
    print(f"  Prediction timesteps : {n_steps}")
    print(f"  Device               : {DEVICE}")
    print(f"  Backend              : {'mamba_ssm' if _MAMBA_AVAILABLE else 'GRU fallback'}")
    print(f"  Checkpoint           : {checkpoint_path or '(none — random weights)'}")
    print(f"  Output directory     : {output_dir}")
    print("=" * 70)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    if use_mock or not all(
        p and os.path.exists(p)
        for p in [pressure_csv, u_velocity_csv, v_velocity_csv]
    ):
        if not use_mock:
            print("  CSV files not found — falling back to mock data.")
        coords, pressure, u_vel, v_vel = make_mock_data()
    else:
        cfg.pressure_path = pressure_csv
        cfg.u_vel_path = u_velocity_csv
        cfg.v_vel_path = v_velocity_csv
        coords, pressure, u_vel, v_vel = load_csv_data(
            pressure_csv, u_velocity_csv, v_velocity_csv
        )

    N, T = pressure.shape
    print(f"  Nodes: {N:,}  |  Timesteps: {T}")

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    print("\n[2/5] Normalising data …")
    T_train = int(T * cfg.train_ratio)
    normalizer = DataNormalizer()
    p_n, u_n, v_n, c_n = normalizer.fit_transform(
        pressure, u_vel, v_vel, coords, T_train
    )

    # Stack into (N, T, 5): p, u, v, x_norm, y_norm
    x_rep = np.repeat(c_n[:, 0:1], T, axis=1)
    y_rep = np.repeat(c_n[:, 1:2], T, axis=1)
    data = np.stack([p_n, u_n, v_n, x_rep, y_rep], axis=2).astype(np.float32)

    # Hilbert sort
    print("  Computing Hilbert-curve ordering …")
    hilbert_idx = hilbert_sort_indices(coords)

    # ── 3. Build Model ────────────────────────────────────────────────────────
    print("\n[3/5] Building model …")
    model = STMambaModel(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print("  ✔ Checkpoint loaded successfully.")
    else:
        if checkpoint_path:
            print(f"  ⚠ Checkpoint not found at {checkpoint_path}")
        print("  Using randomly initialised weights (for testing only).")

    # ── 4. Prepare Seed Window ────────────────────────────────────────────────
    print("\n[4/5] Preparing seed window …")
    T_val = int(T * cfg.val_ratio)
    test_start = T_train + T_val

    if seed_timestep is not None:
        s_start = seed_timestep
    else:
        s_start = test_start

    # Clamp to valid range
    max_start = T - cfg.time_in
    if s_start > max_start:
        s_start = max(0, max_start)
        print(f"  Adjusted seed start to {s_start} (data has {T} timesteps).")

    seed_window, n_patches = prepare_seed_window(
        data, hilbert_idx, cfg.patch_size, cfg.time_in, s_start
    )
    print(
        f"  Seed window : timesteps [{s_start}, {s_start + cfg.time_in})"
    )
    print(f"  Tensor shape: {tuple(seed_window.shape)}")
    print(f"  Patches     : {n_patches}")

    # ── 5. Autoregressive Rollout ─────────────────────────────────────────────
    print(f"\n[5/5] Running autoregressive rollout for {n_steps} timestep(s) …")
    rollout = autoregressive_rollout(
        model, seed_window, n_steps, DEVICE, cfg.use_amp
    )
    # rollout: (n_steps, P, S, 3) → flatten patches → (n_steps, P*S, 3)
    R, P, S, Cv = rollout.shape
    pred_flat = rollout.reshape(R, P * S, Cv).float().numpy()
    print(f"  ✔ Rollout complete — output shape: ({R}, {P * S}, {Cv})")

    # ── 6. Denormalise & print summary ────────────────────────────────────────
    N_out = min(N, P * S)
    var_names = ["pressure", "u_velocity", "v_velocity"]

    print("\n── Prediction Summary (first 5 nodes, denormalised) ──")
    predictions_physical = np.zeros_like(pred_flat[:, :N_out, :])
    for v_idx, v_name in enumerate(var_names):
        pred_v = pred_flat[:, :N_out, v_idx]
        phys = normalizer.inverse_transform_var(pred_v.T, v_name)
        predictions_physical[:, :, v_idx] = phys.T

    header = "  {:>5s}".format("node")
    for step in range(min(n_steps, 5)):
        header += f"  {'t' + str(step + 1):>12s}"
    for v_idx, v_name in enumerate(var_names):
        print(f"\n  {v_name}:")
        print(header)
        for node in range(min(5, N_out)):
            row = f"  {node:5d}"
            for step in range(min(n_steps, 5)):
                row += f"  {predictions_physical[step, node, v_idx]:12.4f}"
            print(row)

    # ── 7. Export CSV ─────────────────────────────────────────────────────────
    print(f"\n── Exporting predictions to {output_dir} ──")
    csv_paths = export_predictions_to_csv(
        pred_flat, coords, normalizer, output_dir
    )

    print("\n✔ Inference complete.")
    return {
        "predictions_raw": pred_flat,
        "predictions_physical": predictions_physical,
        "coords": coords,
        "normalizer": normalizer,
        "config": cfg,
        "csv_paths": csv_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-Mamba autoregressive inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Predict 20 future timesteps using a trained checkpoint\n"
            "  python inference_st_mamba.py --n_steps 20 "
            "--checkpoint best_checkpoint.pth \\\n"
            "      --pressure_csv data/p.csv "
            "--u_velocity_csv data/u.csv "
            "--v_velocity_csv data/v.csv\n\n"
            "  # Quick smoke test with mock data\n"
            "  python inference_st_mamba.py --n_steps 5 --use_mock\n"
        ),
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        required=True,
        help="Number of future timesteps to predict (required).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained model checkpoint (.pth). "
        "If omitted, uses random weights (testing only).",
    )
    parser.add_argument(
        "--pressure_csv",
        type=str,
        default=None,
        help="Path to the pressure CSV file.",
    )
    parser.add_argument(
        "--u_velocity_csv",
        type=str,
        default=None,
        help="Path to the u-velocity CSV file.",
    )
    parser.add_argument(
        "--v_velocity_csv",
        type=str,
        default=None,
        help="Path to the v-velocity CSV file.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock/synthetic data for testing.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Directory for output CSV files (default: inference_outputs).",
    )
    parser.add_argument(
        "--seed_timestep",
        type=int,
        default=None,
        help="Starting timestep index for the input seed window. "
        "Defaults to the first timestep of the test split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        n_steps=args.n_steps,
        checkpoint_path=args.checkpoint,
        pressure_csv=args.pressure_csv,
        u_velocity_csv=args.u_velocity_csv,
        v_velocity_csv=args.v_velocity_csv,
        use_mock=args.use_mock,
        output_dir=args.output_dir,
        seed_timestep=args.seed_timestep,
    )
