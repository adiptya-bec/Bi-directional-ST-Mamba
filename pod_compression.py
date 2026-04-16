"""
pod_compression.py — POD compression with corrected rank truncation,
active-domain restriction, and pressure outlier diagnostics.

Changes applied (from technical assessment):
  Change 2: POD Rank Truncation — p:30, u:50, v:30 (was 150 each)
  Change 5: Active Domain Restriction — restrict SVD to TKE > 1% of max
  Change 6: Pressure Outlier Diagnostics — log before clipping
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Default rank configuration (Change 2)
# ---------------------------------------------------------------------------
DEFAULT_POD_RANKS: Dict[str, int] = {
    "p": 30,   # 99.9% energy at r=25; +5 margin
    "u": 50,   # 99.9% energy at r=43; +7 margin
    "v": 30,   # 99.9% energy at r=29; +1 margin
}


# ---------------------------------------------------------------------------
# Pressure outlier diagnostics (Change 6)
# ---------------------------------------------------------------------------
def diagnose_pressure_outliers(
    X: np.ndarray,
    percentile: float = 2.5,
    label: str = "p",
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagnose pressure outliers before clipping.

    Parameters
    ----------
    X : (N_cells, T) array of pressure snapshots
    percentile : clip percentile (e.g. 2.5 → clip beyond [2.5%, 97.5%])
    label : variable name for logging

    Returns
    -------
    lo, hi : lower and upper clip bounds (shape ())
    """
    lo = float(np.percentile(X, percentile))
    hi = float(np.percentile(X, 100.0 - percentile))
    n_below = int(np.sum(X < lo))
    n_above = int(np.sum(X > hi))
    total = X.size
    frac_below = n_below / total * 100
    frac_above = n_above / total * 100
    print(
        f"  [diag] '{label}' outliers: "
        f"{n_below} below {lo:.3e} ({frac_below:.3f}%), "
        f"{n_above} above {hi:.3e} ({frac_above:.3f}%)"
    )
    if frac_below + frac_above > 5.0:
        warnings.warn(
            f"Variable '{label}': {frac_below + frac_above:.1f}% of values "
            "are outside clip bounds — possible CFD divergence.",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.float32(lo), np.float32(hi)


# ---------------------------------------------------------------------------
# Active domain restriction (Change 5)
# ---------------------------------------------------------------------------
def compute_pod_active_domain(
    velocity_snapshots: np.ndarray,
    tke_fraction: float = 0.01,
) -> np.ndarray:
    """Return a boolean mask of cells with TKE > tke_fraction * max_TKE.

    Computing TKE as 0.5*(u'^2 + v'^2) per cell averaged over time.

    Parameters
    ----------
    velocity_snapshots : (N_cells, T, 2) array [u, v] or (N_cells, T) for one component
    tke_fraction : minimum TKE fraction of global max

    Returns
    -------
    active_mask : (N_cells,) boolean array
    """
    if velocity_snapshots.ndim == 3:
        u = velocity_snapshots[:, :, 0]
        v = velocity_snapshots[:, :, 1]
        tke = 0.5 * (np.var(u, axis=1) + np.var(v, axis=1))
    else:
        tke = np.var(velocity_snapshots, axis=1)

    tke_max = float(tke.max())
    threshold = tke_fraction * tke_max
    active_mask = tke >= threshold

    n_active = int(active_mask.sum())
    n_total = len(active_mask)
    print(
        f"  Active domain: {n_active:,} / {n_total:,} cells "
        f"({n_active / n_total * 100:.1f}%) with TKE ≥ {tke_fraction:.0%} of max"
    )
    return active_mask


# ---------------------------------------------------------------------------
# Core POD computation (method of snapshots, TKE-weighted)
# ---------------------------------------------------------------------------
def compute_pod(
    X: np.ndarray,
    r: int,
    tke_weights: Optional[np.ndarray] = None,
    active_mask: Optional[np.ndarray] = None,
    label: str = "var",
) -> Dict:
    """Compute POD basis via method of snapshots (TKE-weighted).

    Parameters
    ----------
    X : (N_cells, T_train) snapshot matrix
    r : number of POD modes to retain
    tke_weights : (N_cells,) spatial TKE weights (None → uniform)
    active_mask : (N_cells,) boolean active-domain mask (None → all cells)
    label : variable name for printing

    Returns
    -------
    dict with keys:
        phi   : (N_active, r) spatial modes (unit vectors in weighted space)
        sigma : (r,) singular values
        mean  : (N_cells,) temporal mean
        energy_fractions : (r,) cumulative energy fractions
        active_mask : (N_cells,) boolean mask used
        r_used : actual rank used
    """
    N_cells, T = X.shape
    t0 = time.time()

    # Mean-centre
    mean = X.mean(axis=1)  # (N_cells,)
    Xc = X - mean[:, None]  # (N_cells, T)

    # Apply active-domain mask
    if active_mask is None:
        active_mask = np.ones(N_cells, dtype=bool)
    Xc_active = Xc[active_mask]  # (N_active, T)
    N_active = Xc_active.shape[0]

    # TKE weights restricted to active cells
    if tke_weights is not None:
        w = tke_weights[active_mask].astype(np.float64)
        w = w / (w.sum() + 1e-12)  # normalise
        w_sqrt = np.sqrt(w)[:, None]  # (N_active, 1)
        Xw = Xc_active * w_sqrt      # weighted snapshot matrix
    else:
        Xw = Xc_active.copy()

    # Method of snapshots: C = Xw^T Xw  (T×T covariance)
    print(f"  POD '{label}' — method of snapshots  (N_train={T}, N_active={N_active:,}, r={r})")
    C = (Xw.T @ Xw) / (T - 1)  # (T, T)

    # SVD of the small T×T matrix
    t_svd = time.time()
    U, s, _ = np.linalg.svd(C, full_matrices=False)  # U:(T,T), s:(T,)
    print(f"    Snapshots SVD {time.time()-t_svd:.1f}s")

    # Energy fractions
    energy = s ** 2
    total_energy = energy.sum() + 1e-30
    cum_energy = np.cumsum(energy) / total_energy

    r90 = int(np.searchsorted(cum_energy, 0.90)) + 1
    r99 = int(np.searchsorted(cum_energy, 0.99)) + 1
    r999 = int(np.searchsorted(cum_energy, 0.999)) + 1
    print(
        f"    Energy:  90% at r={r90}  99% at r={r99}  99.9% at r={r999}  (using r={r})"
    )

    # Truncate to r modes
    r = min(r, T, N_active)
    U_r = U[:, :r]    # (T, r)
    s_r = s[:r]       # (r,)

    # Recover spatial modes: phi = Xw @ U_r / s_r  (normalised)
    # phi : (N_active, r)
    phi_w = Xw @ U_r / (s_r[None, :] + 1e-30)  # (N_active, r)

    # Un-weight to get physical modes
    if tke_weights is not None:
        phi = phi_w / (w_sqrt + 1e-30)  # (N_active, r)
    else:
        phi = phi_w  # (N_active, r)

    print(f"    POD '{label}' done in {time.time()-t0:.1f}s")

    return {
        "phi": phi.astype(np.float32),          # (N_active, r)
        "phi_w": phi_w.astype(np.float32),       # weighted modes
        "sigma": s_r.astype(np.float32),
        "mean": mean.astype(np.float32),
        "energy_fractions": cum_energy[:r].astype(np.float32),
        "active_mask": active_mask,
        "r_used": r,
        "r90": r90,
        "r99": r99,
        "r999": r999,
    }


# ---------------------------------------------------------------------------
# Encode / Decode helpers
# ---------------------------------------------------------------------------
def pod_encode(X: np.ndarray, pod: Dict) -> np.ndarray:
    """Project snapshots onto POD basis.

    Parameters
    ----------
    X : (N_cells, T) snapshot matrix
    pod : dict from compute_pod

    Returns
    -------
    A : (T, r) POD coefficients
    """
    active_mask = pod["active_mask"]
    mean = pod["mean"]
    phi_w = pod["phi_w"]   # (N_active, r)

    if "tke_weights" in pod:
        w = pod["tke_weights"][active_mask]
        w = w / (w.sum() + 1e-12)
        w_sqrt = np.sqrt(w)[:, None]
        Xc_active = (X[active_mask] - mean[active_mask, None]) * w_sqrt
    else:
        Xc_active = X[active_mask] - mean[active_mask, None]  # (N_active, T)

    A = phi_w.T @ Xc_active  # (r, T)
    return A.T.astype(np.float32)  # (T, r)


def pod_decode(A: np.ndarray, pod: Dict, N_cells: int) -> np.ndarray:
    """Reconstruct snapshots from POD coefficients.

    Parameters
    ----------
    A : (T, r) POD coefficients
    pod : dict from compute_pod
    N_cells : total number of cells

    Returns
    -------
    X_recon : (N_cells, T) reconstructed snapshots
    """
    active_mask = pod["active_mask"]
    mean = pod["mean"]
    phi = pod["phi"]   # (N_active, r)

    T = A.shape[0]
    X_recon = np.zeros((N_cells, T), dtype=np.float32)
    X_recon[active_mask] = (phi @ A.T).astype(np.float32)  # (N_active, T)
    X_recon += mean[:, None]
    return X_recon


# ---------------------------------------------------------------------------
# Full pipeline helper: compress all three variables
# ---------------------------------------------------------------------------
def compress_all_variables(
    data: Dict[str, np.ndarray],
    ranks: Optional[Dict[str, int]] = None,
    tke_weights: Optional[np.ndarray] = None,
    active_mask: Optional[np.ndarray] = None,
    pressure_outlier_percentile: float = 2.5,
) -> Tuple[Dict[str, Dict], np.ndarray]:
    """Compress p, u, v using POD with corrected rank truncation.

    Parameters
    ----------
    data : dict with keys 'p', 'u', 'v' → (N_cells, T_train) arrays
    ranks : dict with per-variable ranks (defaults to DEFAULT_POD_RANKS)
    tke_weights : (N_cells,) spatial TKE weights
    active_mask : (N_cells,) boolean active-domain mask
    pressure_outlier_percentile : percentile for pressure clipping

    Returns
    -------
    pods : dict[var] = pod_dict from compute_pod
    Z_train : (T_train, total_r) concatenated POD coefficients
    """
    if ranks is None:
        ranks = DEFAULT_POD_RANKS

    pods: Dict[str, Dict] = {}
    coeff_list = []

    for var in ("p", "u", "v"):
        X = data[var].astype(np.float64)  # (N_cells, T)
        N_cells, T = X.shape

        # Pressure outlier diagnostics (Change 6)
        if var == "p":
            lo, hi = diagnose_pressure_outliers(X, percentile=pressure_outlier_percentile, label=var)
            X = np.clip(X, lo, hi)

        r = ranks[var]
        pod = compute_pod(
            X,
            r=r,
            tke_weights=tke_weights,
            active_mask=active_mask,
            label=var,
        )
        pods[var] = pod

        # Encode all training snapshots
        A = pod_encode(X, pod)  # (T, r)
        coeff_list.append(A)

    Z_train = np.concatenate(coeff_list, axis=1)  # (T, r_p+r_u+r_v)
    print(
        f"\n  Latent space: Z shape = {Z_train.shape}  "
        f"(was (T, 450) with r=150 each)"
    )
    return pods, Z_train
