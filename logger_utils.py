"""
logger_utils.py — Comprehensive experiment logging.

Change 12: Comprehensive Experiment Logging
  - Track POD truncation, loss components, mode weights
  - Save metadata (config, truncation log, energy summary)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------
def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float, str, type(None))):
        return obj
    else:
        return str(obj)


# ---------------------------------------------------------------------------
# ExperimentLogger (Change 12)
# ---------------------------------------------------------------------------
class ExperimentLogger:
    """Logs experiment metadata, training history, and evaluation results.

    Creates the following files in `log_dir`:
      - metadata.json        : configuration and run info
      - truncation_log.json  : POD rank selection details
      - energy_summary.json  : POD energy capture per variable
      - training_history.json: per-epoch loss components
      - evaluation.json      : final evaluation metrics

    Parameters
    ----------
    log_dir : directory for log files
    experiment_name : name prefix for the experiment
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "cfd_forecasting",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start_time = time.time()

        # Internal state
        self._metadata: Dict[str, Any] = {}
        self._truncation_log: List[Dict] = []
        self._energy_summary: List[Dict] = []
        self._training_history: Dict[str, List] = {}
        self._evaluation: Dict[str, Any] = {}

        print(f"  ExperimentLogger: run_id={self.run_id}  log_dir={self.log_dir}")

    # ── Metadata (config, system info) ────────────────────────────────────────
    def log_metadata(self, config: Dict, extra: Optional[Dict] = None) -> None:
        """Log run configuration and metadata."""
        self._metadata = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": _to_json_safe(config),
        }
        if extra:
            self._metadata.update(_to_json_safe(extra))
        self._save("metadata.json", self._metadata)
        print(f"  Logged metadata → {self.log_dir / 'metadata.json'}")

    # ── POD truncation log (Change 12) ────────────────────────────────────────
    def log_pod_truncation(
        self,
        var: str,
        r_used: int,
        r90: int,
        r99: int,
        r999: int,
        cum_energy_at_r: float,
        n_active_cells: int,
        n_total_cells: int,
    ) -> None:
        """Log POD rank selection for one variable."""
        entry = {
            "variable": var,
            "r_used": int(r_used),
            "r_90pct": int(r90),
            "r_99pct": int(r99),
            "r_999pct": int(r999),
            "cum_energy_at_r": float(cum_energy_at_r),
            "n_active_cells": int(n_active_cells),
            "n_total_cells": int(n_total_cells),
            "active_fraction": float(n_active_cells / (n_total_cells + 1e-10)),
        }
        self._truncation_log.append(entry)

    def save_truncation_log(self) -> None:
        """Save accumulated POD truncation log."""
        self._save("truncation_log.json", self._truncation_log)
        print(f"  Saved truncation_log → {self.log_dir / 'truncation_log.json'}")

    # ── POD energy summary ────────────────────────────────────────────────────
    def log_energy_summary(
        self,
        var: str,
        sigma: np.ndarray,
        r_used: int,
        total_energy: float,
    ) -> None:
        """Log POD energy distribution for one variable."""
        energy = sigma ** 2
        total = float(energy.sum()) + 1e-30
        cum_energy = float(energy[:r_used].sum() / total)
        entry = {
            "variable": var,
            "r_used": int(r_used),
            "singular_values": sigma[:min(10, len(sigma))].tolist(),
            "total_energy": float(total_energy),
            "energy_captured_at_r": cum_energy,
            "energy_per_mode": (energy[:r_used] / total).tolist(),
        }
        self._energy_summary.append(entry)

    def save_energy_summary(self) -> None:
        """Save POD energy summary."""
        self._save("energy_summary.json", self._energy_summary)
        print(f"  Saved energy_summary → {self.log_dir / 'energy_summary.json'}")

    # ── Training history ──────────────────────────────────────────────────────
    def log_training_epoch(
        self,
        epoch: int,
        model_name: str,
        train_loss: float,
        val_loss: float,
        components: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
        amp_weight: float = 0.0,
    ) -> None:
        """Log one epoch of training history."""
        key = model_name
        if key not in self._training_history:
            self._training_history[key] = []
        entry: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr),
            "amp_weight": float(amp_weight),
        }
        if components:
            entry.update({f"train_{k}": float(v) for k, v in components.items()})
        self._training_history[key].append(entry)

    def save_training_history(self) -> None:
        """Save full training history."""
        self._save("training_history.json", self._training_history)
        print(f"  Saved training_history → {self.log_dir / 'training_history.json'}")

    # ── Evaluation results ────────────────────────────────────────────────────
    def log_evaluation(
        self,
        model_name: str,
        metrics: Dict[str, float],
        pred_horizon: int = 1,
        inference_mode: str = "iterative_rollout",
    ) -> None:
        """Log final evaluation metrics for one model."""
        self._evaluation[model_name] = {
            "metrics": _to_json_safe(metrics),
            "pred_horizon": pred_horizon,
            "inference_mode": inference_mode,
            "timestamp": datetime.now().isoformat(),
        }

    def save_evaluation(self) -> None:
        """Save evaluation results."""
        self._save("evaluation.json", self._evaluation)
        print(f"  Saved evaluation → {self.log_dir / 'evaluation.json'}")

    # ── Mode weights ──────────────────────────────────────────────────────────
    def log_mode_weights(
        self,
        mode_weights: np.ndarray,
        tke_mode_weights: np.ndarray,
    ) -> None:
        """Log per-mode loss weights."""
        data = {
            "mode_weights": {
                "mean": float(mode_weights.mean()),
                "min": float(mode_weights.min()),
                "max": float(mode_weights.max()),
                "values": mode_weights[:20].tolist(),  # save first 20
            },
            "tke_mode_weights": {
                "mean": float(tke_mode_weights.mean()),
                "min": float(tke_mode_weights.min()),
                "max": float(tke_mode_weights.max()),
                "values": tke_mode_weights[:20].tolist(),
            },
        }
        self._save("mode_weights.json", data)
        print(f"  Saved mode_weights → {self.log_dir / 'mode_weights.json'}")

    # ── Final summary ─────────────────────────────────────────────────────────
    def print_final_summary(self) -> None:
        """Print a final summary of all logged evaluation results."""
        elapsed = time.time() - self._start_time
        print()
        print("=" * 60)
        print(f"  Experiment: {self.experiment_name}  run_id={self.run_id}")
        print(f"  Elapsed: {elapsed:.0f}s")
        print("=" * 60)
        print(f"  {'Model':<25}  {'RelErr':>8}  {'Phase°':>8}  {'AmpErr':>8}")
        print("  " + "-" * 54)
        for model_name, data in self._evaluation.items():
            m = data.get("metrics", {})
            rel = m.get("rel_err", float("nan"))
            phase = m.get("phase_error_deg", float("nan"))
            amp = m.get("amplitude_error", float("nan"))
            status = "✓" if rel < 1.0 else "✗"
            print(f"  {model_name:<25}  {rel:>8.4f}{status}  {phase:>8.1f}  {amp:>8.4f}")
        print("=" * 60)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _save(self, filename: str, data: Any) -> None:
        """Save data as JSON to log_dir/filename."""
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(_to_json_safe(data), f, indent=2)


# ---------------------------------------------------------------------------
# Convenience function to create a logger from config dict
# ---------------------------------------------------------------------------
def create_logger(config: Optional[Dict] = None, log_dir: str = "logs") -> ExperimentLogger:
    """Create an ExperimentLogger from a config dict."""
    if config is None:
        config = {}
    experiment_name = config.get("pipeline", {}).get("case", "cfd_forecasting")
    if isinstance(experiment_name, int):
        experiment_name = f"case_{experiment_name}"
    logger = ExperimentLogger(log_dir=log_dir, experiment_name=str(experiment_name))
    logger.log_metadata(config)
    return logger
