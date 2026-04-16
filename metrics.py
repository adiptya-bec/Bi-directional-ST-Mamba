"""
metrics.py — Comprehensive evaluation metrics for CFD forecasting.

Change 10: Phase-Aware & Spectral Metrics
  - Add: Phase error (via cross-correlation), amplitude error,
         PSD spectrum, Strouhal number
  - Old: RelErr only
  - Impact: Identify whether errors are phase (fixable) or amplitude (structural)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------
def rel_err(pred: np.ndarray, target: np.ndarray) -> float:
    """Relative error (RelErr = RMSE / RMS(target)).

    RelErr < 1.0 → better than the trivial mean predictor.
    RelErr > 1.0 → FAILURE: worse than predicting the mean.
    """
    mse_val = float(np.mean((pred - target) ** 2))
    rms_tgt = float(np.sqrt(np.mean(target ** 2))) + 1e-12
    return float(np.sqrt(mse_val)) / rms_tgt


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


# ---------------------------------------------------------------------------
# Phase error via cross-correlation
# ---------------------------------------------------------------------------
def phase_error_deg(
    pred: np.ndarray,
    target: np.ndarray,
    dt: float = 1.0,
    dominant_freq: Optional[float] = None,
) -> float:
    """Estimate phase error in degrees via cross-correlation.

    Parameters
    ----------
    pred : (T,) predicted 1D signal
    target : (T,) ground-truth 1D signal
    dt : time step in seconds
    dominant_freq : dominant frequency in Hz (estimated if None)

    Returns
    -------
    phase_error_degrees : mean phase error in degrees
    """
    T = len(pred)
    # Normalise signals
    p = pred - pred.mean()
    t = target - target.mean()

    p_std = p.std() + 1e-12
    t_std = t.std() + 1e-12
    p = p / p_std
    t = t / t_std

    # Cross-correlation
    corr = np.correlate(p, t, mode="full")
    lags = np.arange(-(T - 1), T)
    lag_peak = int(lags[np.argmax(corr)])

    # Convert lag to phase angle
    if dominant_freq is None:
        # Estimate dominant frequency from target
        freqs = np.fft.rfftfreq(T, d=dt)
        power = np.abs(np.fft.rfft(t)) ** 2
        if len(power) > 1:
            dominant_freq = float(freqs[1 + int(np.argmax(power[1:]))])
        else:
            dominant_freq = 1.0 / (T * dt)

    period_steps = 1.0 / (dominant_freq * dt + 1e-12)
    phase_error = (lag_peak / (period_steps + 1e-12)) * 360.0
    # Wrap to [-180, 180]
    phase_error = ((phase_error + 180) % 360) - 180
    return float(abs(phase_error))


# ---------------------------------------------------------------------------
# Amplitude preservation ratio
# ---------------------------------------------------------------------------
def amplitude_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """Ratio of predicted to ground-truth RMS amplitude.

    Ideal value = 1.0.
    < 1.0 → amplitude collapse (under-prediction).
    > 1.0 → over-prediction.
    """
    pred_rms = float(np.sqrt(np.mean(pred ** 2))) + 1e-12
    tgt_rms = float(np.sqrt(np.mean(target ** 2))) + 1e-12
    return pred_rms / tgt_rms


def amplitude_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Absolute amplitude error: |pred_RMS - target_RMS| / target_RMS."""
    pred_rms = float(np.sqrt(np.mean(pred ** 2)))
    tgt_rms = float(np.sqrt(np.mean(target ** 2))) + 1e-12
    return abs(pred_rms - tgt_rms) / tgt_rms


# ---------------------------------------------------------------------------
# Spectral metrics (PSD-based)
# ---------------------------------------------------------------------------
def compute_psd(
    signal: np.ndarray,
    dt: float = 1.0,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method.

    Parameters
    ----------
    signal : (T,) 1D time series
    dt : time step in seconds
    nperseg : segment length for Welch (None → T//4)

    Returns
    -------
    freqs : (N_freq,) frequency array in Hz
    psd : (N_freq,) PSD values
    """
    T = len(signal)
    if nperseg is None:
        nperseg = max(16, T // 4)
    fs = 1.0 / dt
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, T))
    return freqs, psd


def spectral_error(
    pred: np.ndarray,
    target: np.ndarray,
    dt: float = 1.0,
    n_top_modes: int = 5,
) -> Dict[str, float]:
    """Compare PSD of predicted and ground-truth signals.

    Parameters
    ----------
    pred : (T,) predicted signal
    target : (T,) ground-truth signal
    dt : time step in seconds
    n_top_modes : number of dominant frequency modes to compare

    Returns
    -------
    dict with spectral error metrics
    """
    f_pred, psd_pred = compute_psd(pred, dt)
    f_tgt, psd_tgt = compute_psd(target, dt)

    # Normalise PSDs
    psd_pred_norm = psd_pred / (psd_pred.sum() + 1e-12)
    psd_tgt_norm = psd_tgt / (psd_tgt.sum() + 1e-12)

    # Spectral L1 distance
    spectral_l1 = float(np.sum(np.abs(psd_pred_norm - psd_tgt_norm)))

    # Dominant frequency error
    dom_freq_pred = float(f_pred[1 + np.argmax(psd_pred[1:])])  # skip DC
    dom_freq_tgt = float(f_tgt[1 + np.argmax(psd_tgt[1:])])
    dom_freq_err = abs(dom_freq_pred - dom_freq_tgt) / (abs(dom_freq_tgt) + 1e-12)

    return {
        "spectral_l1": spectral_l1,
        "dominant_freq_pred_hz": dom_freq_pred,
        "dominant_freq_tgt_hz": dom_freq_tgt,
        "dominant_freq_err": dom_freq_err,
    }


# ---------------------------------------------------------------------------
# Strouhal number metric
# ---------------------------------------------------------------------------
def strouhal_error(
    pred_dominant_freq_hz: float,
    tgt_dominant_freq_hz: float,
    reference_freq_hz: float = 45.0,
) -> float:
    """Relative error in predicted Strouhal number.

    St = f * L / U  (proportional to frequency for fixed L, U)

    Parameters
    ----------
    pred_dominant_freq_hz : predicted dominant frequency
    tgt_dominant_freq_hz : ground-truth dominant frequency
    reference_freq_hz : reference shedding frequency in Hz

    Returns
    -------
    strouhal_rel_err : |St_pred - St_tgt| / |St_tgt|
    """
    # Normalise by reference to get dimensionless Strouhal ratio
    st_pred = pred_dominant_freq_hz / (reference_freq_hz + 1e-12)
    st_tgt = tgt_dominant_freq_hz / (reference_freq_hz + 1e-12)
    return float(abs(st_pred - st_tgt) / (abs(st_tgt) + 1e-12))


# ---------------------------------------------------------------------------
# ComprehensiveEvaluator (Change 10)
# ---------------------------------------------------------------------------
class ComprehensiveEvaluator:
    """Evaluates CFD forecasting with phase-aware and spectral metrics.

    Computes:
      - RelErr (relative error)
      - MSE, RMSE, MAE
      - Phase error (via cross-correlation)
      - Amplitude preservation ratio
      - Spectral L1 distance
      - Strouhal number error

    Parameters
    ----------
    dt : time step in seconds
    reference_freq_hz : reference shedding frequency for Strouhal
    compute_phase : enable phase error computation
    compute_amplitude : enable amplitude ratio computation
    compute_spectral : enable spectral metrics
    compute_strouhal : enable Strouhal number metric
    mode_subset : list of mode indices to average over (None → first 5)
    """

    def __init__(
        self,
        dt: float = 0.000222,
        reference_freq_hz: float = 45.0,
        compute_phase: bool = True,
        compute_amplitude: bool = True,
        compute_spectral: bool = True,
        compute_strouhal: bool = True,
        mode_subset: Optional[List[int]] = None,
    ):
        self.dt = dt
        self.reference_freq_hz = reference_freq_hz
        self.compute_phase = compute_phase
        self.compute_amplitude = compute_amplitude
        self.compute_spectral = compute_spectral
        self.compute_strouhal = compute_strouhal
        self.mode_subset = mode_subset

    def evaluate_latent(
        self,
        Z_pred: np.ndarray,
        Z_target: np.ndarray,
        var_labels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate predictions in latent (POD coefficient) space.

        Parameters
        ----------
        Z_pred : (T, D) predicted latent sequence
        Z_target : (T, D) ground-truth latent sequence
        var_labels : optional list of variable names for per-variable reporting

        Returns
        -------
        metrics : dict of scalar metric values
        """
        T, D = Z_pred.shape
        metrics: Dict[str, float] = {}

        # Global metrics
        metrics["mse"] = mse(Z_pred, Z_target)
        metrics["rmse"] = rmse(Z_pred, Z_target)
        metrics["mae"] = mae(Z_pred, Z_target)
        metrics["rel_err"] = rel_err(Z_pred, Z_target)

        # Per-mode metrics (averaged over a subset)
        modes = self.mode_subset if self.mode_subset is not None else list(range(min(5, D)))

        phase_errors = []
        amp_errors = []
        spectral_l1s = []
        dom_freq_errs = []

        for m in modes:
            p = Z_pred[:, m]
            t = Z_target[:, m]

            if self.compute_phase:
                phase_errors.append(phase_error_deg(p, t, dt=self.dt))

            if self.compute_amplitude:
                amp_errors.append(amplitude_error(p, t))

            if self.compute_spectral:
                spec = spectral_error(p, t, dt=self.dt)
                spectral_l1s.append(spec["spectral_l1"])
                dom_freq_errs.append(spec["dominant_freq_err"])

        if self.compute_phase and phase_errors:
            metrics["phase_error_deg"] = float(np.mean(phase_errors))
            metrics["phase_error_deg_max"] = float(np.max(phase_errors))

        if self.compute_amplitude and amp_errors:
            metrics["amplitude_error"] = float(np.mean(amp_errors))
            metrics["amplitude_ratio"] = float(np.mean([
                amplitude_ratio(Z_pred[:, m], Z_target[:, m]) for m in modes
            ]))

        if self.compute_spectral and spectral_l1s:
            metrics["spectral_l1"] = float(np.mean(spectral_l1s))
            metrics["dominant_freq_err"] = float(np.mean(dom_freq_errs))

        if self.compute_strouhal and dom_freq_errs:
            # Use mode 0 for Strouhal
            m0 = modes[0]
            spec0 = spectral_error(Z_pred[:, m0], Z_target[:, m0], dt=self.dt)
            metrics["strouhal_err"] = strouhal_error(
                spec0["dominant_freq_pred_hz"],
                spec0["dominant_freq_tgt_hz"],
                reference_freq_hz=self.reference_freq_hz,
            )

        return metrics

    def evaluate_physical(
        self,
        fields_pred: Dict[str, np.ndarray],
        fields_target: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate predictions in physical (reconstructed) space per variable.

        Parameters
        ----------
        fields_pred : dict[var] → (N_cells, T_pred)
        fields_target : dict[var] → (N_cells, T_pred)

        Returns
        -------
        per_var_metrics : dict[var] → dict of metric values
        """
        results: Dict[str, Dict[str, float]] = {}
        for var in fields_pred:
            pred = fields_pred[var]
            tgt = fields_target[var]
            results[var] = {
                "mse": mse(pred, tgt),
                "rmse": rmse(pred, tgt),
                "rel_err": rel_err(pred, tgt),
            }
        return results

    def print_report(self, metrics: Dict[str, float], model_name: str = "Model") -> None:
        """Print a formatted evaluation report."""
        print(f"\n{'='*60}")
        print(f"  Evaluation Report — {model_name}")
        print(f"{'='*60}")
        print(f"  RelErr        : {metrics.get('rel_err', float('nan')):.4f}  "
              f"({'PASS ✓' if metrics.get('rel_err', 2) < 1.0 else 'FAIL ✗ (> 1 = worse than mean)'})")
        print(f"  MSE           : {metrics.get('mse', float('nan')):.4e}")
        print(f"  RMSE          : {metrics.get('rmse', float('nan')):.4e}")

        if "phase_error_deg" in metrics:
            pe = metrics["phase_error_deg"]
            print(f"  Phase error   : {pe:.1f}°  ({'OK' if pe < 15 else 'HIGH'})")

        if "amplitude_error" in metrics:
            ae = metrics["amplitude_error"]
            ar = metrics.get("amplitude_ratio", float("nan"))
            print(f"  Amplitude err : {ae:.4f}  (ratio={ar:.3f})")

        if "spectral_l1" in metrics:
            print(f"  Spectral L1   : {metrics['spectral_l1']:.4f}")

        if "strouhal_err" in metrics:
            se = metrics["strouhal_err"]
            print(f"  Strouhal err  : {se:.4f}  ({'OK' if se < 0.05 else 'HIGH'})")
        print(f"{'='*60}\n")
