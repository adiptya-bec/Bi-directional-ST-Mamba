"""
preprocessing.py — Data preprocessing with solid masking DISABLED.

Change 3: Remove Solid Masking
  - Old: Velocity-based solid mask capturing 0.11% of cells
  - New: All cells treated as fluid; solid masking disabled via config flag
  - Rationale: No solid body exists in this all-fluid CFD domain
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Velocity distribution diagnostic (kept for reference)
# ---------------------------------------------------------------------------
def diagnose_velocity_distribution(
    velocity_data: np.ndarray,
    n_sample_steps: int = 50,
    percentiles: Tuple[float, ...] = (0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50),
) -> Dict[str, float]:
    """Compute velocity magnitude percentiles for threshold selection.

    Parameters
    ----------
    velocity_data : (N_cells, T, 2) or (N_cells, T) velocity snapshots
    n_sample_steps : number of steps to average over
    percentiles : percentile values to compute

    Returns
    -------
    diag : dict with percentile labels → values
    """
    if velocity_data.ndim == 3:
        vel_mag = np.sqrt(velocity_data[:, :, 0] ** 2 + velocity_data[:, :, 1] ** 2)
    else:
        vel_mag = np.abs(velocity_data)

    # Average over a sample of timesteps
    T = vel_mag.shape[1]
    step = max(1, T // n_sample_steps)
    sample = vel_mag[:, ::step].ravel()

    print("Velocity distribution diagnostic ...")
    print(f"\n  Velocity magnitude percentiles (averaged over {n_sample_steps} steps):")
    diag: Dict[str, float] = {}
    for p in percentiles:
        val = float(np.percentile(sample, p))
        label = f"p{p:5.2f}"
        diag[label] = val
        print(f"    {label} : {val:.4e} m/s")
    val_max = float(sample.max())
    diag["max"] = val_max
    print(f"    max   : {val_max:.4e} m/s")

    # Count cells below candidate thresholds
    print("\n  Cells below candidate thresholds:")
    for thresh in [1e-4, 1e-3, 1e-2, 1e-1, 1e0]:
        n = int((vel_mag.mean(axis=1) < thresh).sum())
        frac = n / vel_mag.shape[0] * 100
        print(f"    < {thresh:.0e} m/s : {n:10,}  ({frac:.2f}%)")

    return diag


# ---------------------------------------------------------------------------
# Solid masking (Change 3: DISABLED by default)
# ---------------------------------------------------------------------------
def build_solid_mask(
    velocity_data: np.ndarray,
    thresh: float = 0.1,
    enabled: bool = False,
) -> np.ndarray:
    """Build a boolean fluid mask (True = fluid, False = solid).

    Parameters
    ----------
    velocity_data : (N_cells, T) or (N_cells, T, 2) velocity snapshots
    thresh : velocity threshold in m/s for solid detection
    enabled : if False (default), returns all-True mask (all fluid)

    Returns
    -------
    fluid_mask : (N_cells,) boolean, True = fluid cell
    """
    if velocity_data.ndim == 3:
        vel_mean = np.sqrt(velocity_data[:, :, 0] ** 2 + velocity_data[:, :, 1] ** 2).mean(axis=1)
    else:
        vel_mean = np.abs(velocity_data).mean(axis=1)

    N_cells = vel_mean.shape[0]

    if not enabled:
        # Change 3: solid masking disabled — treat all cells as fluid
        print("Solid mask DISABLED — all cells treated as fluid.")
        print(f"  Fluid : {N_cells:,}")
        print(f"  Solid : 0  (0.00%)")
        return np.ones(N_cells, dtype=bool)

    # Legacy solid masking (kept for reference, not used)
    warnings.warn(
        "Solid masking is enabled. Verify that a solid body exists in the domain.",
        UserWarning,
        stacklevel=2,
    )
    solid_mask = vel_mean < thresh
    fluid_mask = ~solid_mask
    n_fluid = int(fluid_mask.sum())
    n_solid = int(solid_mask.sum())
    print(f"Building solid-body mask  thresh={thresh:.2e} m/s ...")
    print(f"  Fluid : {n_fluid:,}")
    print(f"  Solid : {n_solid:,}  ({n_solid / N_cells * 100:.2f}%)")
    return fluid_mask


# ---------------------------------------------------------------------------
# TKE weight computation
# ---------------------------------------------------------------------------
def compute_tke_weights(
    u_train: np.ndarray,
    v_train: np.ndarray,
    fluid_mask: Optional[np.ndarray] = None,
    floor_fraction: float = 0.01,
    print_interval: int = 50,
) -> np.ndarray:
    """Compute TKE-based spatial weights for POD.

    TKE_i = 0.5 * (Var_t(u_i) + Var_t(v_i))
    Weights are normalised so the mean over fluid cells = 1.

    Parameters
    ----------
    u_train : (N_cells, T_train) x-velocity training snapshots
    v_train : (N_cells, T_train) y-velocity training snapshots
    fluid_mask : (N_cells,) boolean fluid cell mask (None → all cells)
    floor_fraction : minimum weight as fraction of mean weight
    print_interval : print progress every N steps

    Returns
    -------
    tke_weights : (N_cells,) float32 spatial weights (0 for solid cells)
    """
    print("\n  Computing TKE weights from training velocity snapshots ...")
    N_cells, T = u_train.shape

    # Compute variance in chunks to save memory
    u_mean = u_train.mean(axis=1, keepdims=True)
    v_mean = v_train.mean(axis=1, keepdims=True)
    tke = 0.5 * (
        np.mean((u_train - u_mean) ** 2, axis=1)
        + np.mean((v_train - v_mean) ** 2, axis=1)
    )  # (N_cells,)

    tke_max = float(tke.max())
    if fluid_mask is None:
        fluid_mask = np.ones(N_cells, dtype=bool)

    n_fluid = int(fluid_mask.sum())
    n_above_floor = int((tke[fluid_mask] > floor_fraction * tke_max).sum())
    n_above_half = int((tke[fluid_mask] > 0.5 * tke_max).sum())
    print(
        f"    TKE: max={tke_max:.3e}"
        f"  cells>floor={n_above_floor:,}/{n_fluid:,}"
        f"  cells>0.5={n_above_half:,}/{n_fluid:,}"
    )

    # Apply fluid mask and normalise
    tke_fluid = tke.copy()
    tke_fluid[~fluid_mask] = 0.0

    # Apply floor to avoid zero weights in active cells
    tke_mean = float(tke_fluid[fluid_mask].mean()) if n_fluid > 0 else 1.0
    floor = floor_fraction * tke_max
    tke_fluid = np.where(fluid_mask, np.maximum(tke_fluid, floor), 0.0)

    # Normalise so mean over fluid = 1
    tke_mean_new = float(tke_fluid[fluid_mask].mean()) if n_fluid > 0 else 1.0
    weights = tke_fluid / (tke_mean_new + 1e-30)

    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Shedding frequency diagnostic
# ---------------------------------------------------------------------------
def shedding_frequency_diagnostic(
    pod_coeffs: np.ndarray,
    dt: float,
    var_label: str = "var",
    mode_idx: int = 1,
    pred_horizon: int = 50,
) -> Dict[str, float]:
    """Diagnose dominant shedding frequency in POD modal coefficients.

    Parameters
    ----------
    pod_coeffs : (T,) or (T, r) array of POD coefficients
    dt : time step in seconds
    var_label : variable name for printing
    mode_idx : 1-based mode index to analyse
    pred_horizon : current prediction horizon in steps

    Returns
    -------
    diag : dict with 'peak_freq_hz', 'shedding_period_steps', 'recommended_max_horizon'
    """
    if pod_coeffs.ndim == 2:
        coeffs_1d = pod_coeffs[:, mode_idx - 1]
    else:
        coeffs_1d = pod_coeffs

    T = len(coeffs_1d)
    freq = np.fft.rfftfreq(T, d=dt)  # Hz
    power = np.abs(np.fft.rfft(coeffs_1d - coeffs_1d.mean())) ** 2

    # Ignore DC component (index 0)
    peak_idx = int(np.argmax(power[1:]) + 1)
    peak_freq = float(freq[peak_idx])
    shedding_period_s = 1.0 / (peak_freq + 1e-12)
    shedding_period_steps = shedding_period_s / dt
    recommended_max = max(1, int(shedding_period_steps / 2) - 1)

    print(f"\nShedding frequency diagnostic  var={var_label}  mode={mode_idx} ...")
    print(f"  Peak frequency   : {peak_freq:.2f} Hz")
    print(f"  Shedding period  : {shedding_period_s:.4f} s  = {shedding_period_steps:.0f} steps")
    print(f"  Current pred_horizon : {pred_horizon} steps")
    print(f"  Recommended max  : {recommended_max} steps  (< half the shedding period)")
    if pred_horizon > recommended_max:
        print(
            f"  [WARN] pred_horizon={pred_horizon} exceeds recommended {recommended_max}. "
            "Consider reducing it in Config."
        )
    else:
        print(f"  [OK] pred_horizon={pred_horizon} satisfies Nyquist criterion.")

    return {
        "peak_freq_hz": peak_freq,
        "shedding_period_s": shedding_period_s,
        "shedding_period_steps": shedding_period_steps,
        "recommended_max_horizon": recommended_max,
    }
