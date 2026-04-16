"""
loss.py — AdaptiveWeightedLoss with component reweighting.

Change 8: Adaptive Loss Reweighting
  - Detect when weighted_mse plateaus, shift weight to amplitude/tke_amplitude
  - Prevents local minima where model learns the mean
  - Location: AdaptiveWeightedLoss class
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mode weight computation (TKE-based)
# ---------------------------------------------------------------------------
def compute_mode_weights(
    Z_train: np.ndarray,
    floor: float = 0.1,
) -> torch.Tensor:
    """Compute per-mode importance weights from training variance.

    Weights are proportional to the standard deviation of each mode,
    normalised so the mean weight = 1.

    Parameters
    ----------
    Z_train : (T_train, D) training latent sequences
    floor : minimum weight as fraction of mean weight

    Returns
    -------
    weights : (D,) float32 tensor of mode weights
    """
    std = Z_train.std(axis=0) + 1e-10  # (D,)
    weights = std / (std.mean() + 1e-10)
    weights = np.maximum(weights, floor)
    weights = weights / (weights.mean() + 1e-10)  # re-normalise after floor
    return torch.from_numpy(weights.astype(np.float32))


# ---------------------------------------------------------------------------
# Amplitude loss
# ---------------------------------------------------------------------------
def amplitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: int = 10,
) -> torch.Tensor:
    """Loss penalising amplitude (envelope) mismatch.

    Computes the mean squared error between local RMS amplitudes of
    prediction and target over sliding windows.

    Parameters
    ----------
    pred : (B, T, D) predicted latent sequence
    target : (B, T, D) ground-truth latent sequence
    window : window length for local RMS estimation

    Returns
    -------
    loss : scalar tensor
    """
    # Local RMS using sliding window
    if pred.shape[1] < window:
        window = pred.shape[1]

    # Unfold along time dimension: (B, n_win, D, window) — unfold places window dim last
    pred_unf = pred.unfold(1, window, 1)     # (B, n_win, D, window)
    tgt_unf = target.unfold(1, window, 1)    # (B, n_win, D, window)

    # RMS over window dimension
    pred_rms = pred_unf.pow(2).mean(dim=-1).sqrt()  # (B, n_win, D)
    tgt_rms = tgt_unf.pow(2).mean(dim=-1).sqrt()

    return F.mse_loss(pred_rms, tgt_rms)


# ---------------------------------------------------------------------------
# TKE-weighted amplitude loss
# ---------------------------------------------------------------------------
def tke_amplitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tke_mode_weights: torch.Tensor,
    window: int = 10,
) -> torch.Tensor:
    """Amplitude loss weighted by per-mode TKE importance.

    High-TKE modes (energetically dominant) contribute more to the loss.

    Parameters
    ----------
    pred : (B, T, D)
    target : (B, T, D)
    tke_mode_weights : (D,) per-mode weights
    window : window size for local RMS

    Returns
    -------
    loss : scalar tensor
    """
    if pred.shape[1] < window:
        window = pred.shape[1]

    pred_unf = pred.unfold(1, window, 1)
    tgt_unf = target.unfold(1, window, 1)

    pred_rms = pred_unf.pow(2).mean(dim=-1).sqrt()  # (B, n_win, D)
    tgt_rms = tgt_unf.pow(2).mean(dim=-1).sqrt()

    diff_sq = (pred_rms - tgt_rms).pow(2)  # (B, n_win, D)
    weights = tke_mode_weights.to(diff_sq.device).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    weighted_diff = diff_sq * weights
    return weighted_diff.mean()


# ---------------------------------------------------------------------------
# Weighted MSE
# ---------------------------------------------------------------------------
def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode_weights: torch.Tensor,
) -> torch.Tensor:
    """MSE weighted by per-mode importance.

    Parameters
    ----------
    pred : (B, T, D)
    target : (B, T, D)
    mode_weights : (D,) per-mode weights

    Returns
    -------
    loss : scalar tensor
    """
    diff_sq = (pred - target).pow(2)  # (B, T, D)
    weights = mode_weights.to(diff_sq.device).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    return (diff_sq * weights).mean()


# ---------------------------------------------------------------------------
# AdaptiveWeightedLoss (Change 8)
# ---------------------------------------------------------------------------
class AdaptiveWeightedLoss(nn.Module):
    """Loss with adaptive reweighting between MSE and amplitude terms.

    When weighted_mse plateaus (improvement < plateau_threshold for
    plateau_patience epochs), the amplitude and tke_amplitude weights
    are boosted to escape local minima where the model predicts the mean.

    Parameters
    ----------
    mode_weights : (D,) per-mode MSE weights
    tke_mode_weights : (D,) per-mode TKE amplitude weights
    wmse_weight : initial weight for weighted MSE term
    amp_weight : initial weight for amplitude loss
    tke_amp_weight : initial weight for TKE-amplitude loss
    amplitude_window : window size for amplitude computation
    adaptive_enabled : enable plateau detection and reweighting
    plateau_patience : epochs of plateau before boosting amplitude weight
    plateau_threshold : minimum relative improvement to avoid plateau
    amplitude_boost : multiplier for amplitude weight on plateau detection
    """

    def __init__(
        self,
        mode_weights: torch.Tensor,
        tke_mode_weights: torch.Tensor,
        wmse_weight: float = 1.0,
        amp_weight: float = 0.5,
        tke_amp_weight: float = 0.5,
        amplitude_window: int = 10,
        adaptive_enabled: bool = True,
        plateau_patience: int = 10,
        plateau_threshold: float = 0.01,
        amplitude_boost: float = 2.0,
    ):
        super().__init__()
        self.register_buffer("mode_weights", mode_weights)
        self.register_buffer("tke_mode_weights", tke_mode_weights)

        self.wmse_weight_base = wmse_weight
        self.amp_weight_base = amp_weight
        self.tke_amp_weight_base = tke_amp_weight
        self.amplitude_window = amplitude_window

        # Current (possibly boosted) weights
        self.wmse_weight = wmse_weight
        self.amp_weight = amp_weight
        self.tke_amp_weight = tke_amp_weight

        # Adaptive reweighting state
        self.adaptive_enabled = adaptive_enabled
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.amplitude_boost = amplitude_boost

        self._wmse_history: List[float] = []
        self._plateau_counter: int = 0
        self._boosted: bool = False

    def update_adaptive(self, wmse_val: float) -> bool:
        """Update plateau counter and adjust weights if plateau detected.

        Call once per epoch with the current training weighted_mse value.

        Returns
        -------
        boosted : True if weights were adjusted this call
        """
        if not self.adaptive_enabled:
            return False

        self._wmse_history.append(wmse_val)
        if len(self._wmse_history) < 2:
            return False

        prev = self._wmse_history[-2]
        curr = self._wmse_history[-1]
        rel_improvement = (prev - curr) / (abs(prev) + 1e-10)

        if rel_improvement < self.plateau_threshold:
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0
            # Reset to base weights if plateau resolved
            if self._boosted:
                self.amp_weight = self.amp_weight_base
                self.tke_amp_weight = self.tke_amp_weight_base
                self._boosted = False

        if self._plateau_counter >= self.plateau_patience and not self._boosted:
            self.amp_weight = self.amp_weight_base * self.amplitude_boost
            self.tke_amp_weight = self.tke_amp_weight_base * self.amplitude_boost
            self._boosted = True
            self._plateau_counter = 0
            return True

        return False

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute adaptive weighted loss.

        Parameters
        ----------
        pred : (B, T, D) predicted latent sequence
        target : (B, T, D) ground-truth latent sequence

        Returns
        -------
        total_loss : scalar tensor
        components : dict with 'wmse', 'amp', 'tke_amp', 'total'
        """
        wmse = weighted_mse(pred, target, self.mode_weights)
        amp = amplitude_loss(pred, target, window=self.amplitude_window)
        tke_amp = tke_amplitude_loss(
            pred, target, self.tke_mode_weights, window=self.amplitude_window
        )

        total = (
            self.wmse_weight * wmse
            + self.amp_weight * amp
            + self.tke_amp_weight * tke_amp
        )

        components = {
            "wmse": float(wmse.item()),
            "amp": float(amp.item()),
            "tke_amp": float(tke_amp.item()),
            "total": float(total.item()),
        }
        return total, components
