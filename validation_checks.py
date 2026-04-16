"""
validation_checks.py — Pre-training configuration validation.

Change 11: Configuration Validation
  - Validates pred_horizon against Nyquist criterion
  - Checks POD energy capture at selected rank
  - Warns if parameter-to-sample ratio is too high (overfitting risk)
  - Validates model architecture settings
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class CheckResult:
    """Result of a single validation check."""

    def __init__(self, name: str, passed: bool, message: str, value: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.value = value

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_nyquist(
    pred_horizon: int,
    shedding_period_steps: float,
    check_name: str = "Nyquist criterion",
) -> CheckResult:
    """Verify pred_horizon satisfies the Nyquist sampling criterion.

    The prediction horizon must be less than half the shedding period
    to avoid aliasing / phase collapse.

    Nyquist: pred_horizon < shedding_period_steps / 2
    """
    recommended = max(1, int(shedding_period_steps / 2) - 1)
    passed = pred_horizon <= recommended

    if passed:
        msg = (
            f"pred_horizon={pred_horizon} ≤ recommended={recommended} "
            f"(shedding_period={shedding_period_steps:.1f} steps)"
        )
    else:
        msg = (
            f"VIOLATION: pred_horizon={pred_horizon} > recommended={recommended}  "
            f"(shedding_period={shedding_period_steps:.1f} steps). "
            f"Set pred_horizon={recommended} or lower."
        )
    return CheckResult(check_name, passed, msg, value=pred_horizon)


def check_pod_energy(
    energy_fractions: np.ndarray,
    r_used: int,
    threshold: float = 0.999,
    var_label: str = "var",
    check_name: Optional[str] = None,
) -> CheckResult:
    """Verify that selected POD rank captures enough energy."""
    if check_name is None:
        check_name = f"POD energy capture ({var_label})"

    cum_energy = float(energy_fractions[r_used - 1]) if r_used <= len(energy_fractions) else float(energy_fractions[-1])
    passed = cum_energy >= threshold
    msg = (
        f"r={r_used} captures {cum_energy * 100:.2f}% energy "
        f"({'≥' if passed else '<'} threshold {threshold * 100:.1f}%)"
    )
    return CheckResult(check_name, passed, msg, value=cum_energy)


def check_param_sample_ratio(
    n_params: int,
    n_train_samples: int,
    warn_threshold: float = 100.0,
    fail_threshold: float = 10000.0,
    check_name: str = "Parameter-to-sample ratio",
) -> CheckResult:
    """Check parameter-to-sample ratio for overfitting risk.

    Target: ratio < 100 (ideally < 10 for robust generalisation).
    """
    ratio = n_params / (n_train_samples + 1e-10)
    if ratio < warn_threshold:
        passed = True
        msg = f"ratio={ratio:.1f} (params={n_params:,} / samples={n_train_samples:,}) — OK"
    elif ratio < fail_threshold:
        passed = True  # warn but not fail
        msg = (
            f"ratio={ratio:.1f} (params={n_params:,} / samples={n_train_samples:,}) — "
            f"WARNING: high ratio (>{warn_threshold:.0f}), consider smaller model"
        )
    else:
        passed = False
        msg = (
            f"ratio={ratio:.1f} (params={n_params:,} / samples={n_train_samples:,}) — "
            f"CRITICAL: ratio exceeds {fail_threshold:.0f}, severe overfitting expected"
        )
    return CheckResult(check_name, passed, msg, value=ratio)


def check_model_architecture(
    d_model: int,
    n_layers: int,
    n_heads: int,
    latent_dim: int,
    check_name: str = "Model architecture",
) -> CheckResult:
    """Verify model architecture settings are valid."""
    issues = []

    if d_model % n_heads != 0:
        issues.append(f"d_model={d_model} not divisible by n_heads={n_heads}")
    if d_model > 512:
        issues.append(f"d_model={d_model} is large for a 300-step dataset; consider ≤256")
    if n_layers > 4:
        issues.append(f"n_layers={n_layers} may overfit; consider ≤4 for small datasets")
    if latent_dim > 200:
        issues.append(
            f"latent_dim={latent_dim} is high; consider POD rank reduction "
            f"(target ≤110 per technical assessment)"
        )

    passed = len(issues) == 0
    msg = "OK" if passed else "; ".join(issues)
    return CheckResult(check_name, passed, msg)


def check_solid_masking(
    solid_masking_enabled: bool,
    n_solid_cells: int = 0,
    n_total_cells: int = 1,
    check_name: str = "Solid masking",
) -> CheckResult:
    """Verify solid masking configuration.

    Warns if masking is enabled but < 1% cells are solid (likely no body).
    """
    if not solid_masking_enabled:
        msg = "DISABLED — all cells treated as fluid (recommended for all-fluid domains)"
        return CheckResult(check_name, True, msg)

    solid_frac = n_solid_cells / (n_total_cells + 1e-10)
    if solid_frac < 0.001:
        passed = False
        msg = (
            f"Solid masking ENABLED but only {solid_frac * 100:.2f}% cells are solid "
            f"({n_solid_cells:,} / {n_total_cells:,}). "
            "This domain likely has no solid body; disable solid masking."
        )
    else:
        passed = True
        msg = (
            f"Solid masking ENABLED with {solid_frac * 100:.2f}% solid cells "
            f"({n_solid_cells:,} / {n_total_cells:,})"
        )
    return CheckResult(check_name, passed, msg)


def check_latent_dim_reduction(
    old_latent_dim: int,
    new_latent_dim: int,
    check_name: str = "Latent dim reduction",
) -> CheckResult:
    """Check that POD rank reduction has been applied."""
    reduction_pct = (1 - new_latent_dim / (old_latent_dim + 1e-10)) * 100
    passed = new_latent_dim < old_latent_dim
    msg = (
        f"latent_dim: {old_latent_dim} → {new_latent_dim} "
        f"({reduction_pct:.0f}% reduction)"
        if passed
        else f"No reduction: latent_dim={new_latent_dim} (expected < {old_latent_dim})"
    )
    return CheckResult(check_name, passed, msg)


# ---------------------------------------------------------------------------
# Pre-training validation suite (Change 11)
# ---------------------------------------------------------------------------
class PreTrainingValidator:
    """Runs all pre-training configuration checks and prints a summary.

    Parameters
    ----------
    config : dict with pipeline configuration
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.results: List[CheckResult] = []

    def run_all(
        self,
        pred_horizon: int,
        shedding_period_steps: float,
        pod_energy_data: Optional[Dict] = None,
        n_model_params: Optional[int] = None,
        n_train_samples: Optional[int] = None,
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        latent_dim: Optional[int] = None,
        solid_masking_enabled: bool = False,
        n_solid_cells: int = 0,
        n_total_cells: int = 1,
    ) -> Tuple[int, int]:
        """Run all validation checks.

        Returns
        -------
        (n_passed, n_failed) : count of passed and failed checks
        """
        self.results = []

        # Check 1: Nyquist criterion
        self.results.append(
            check_nyquist(pred_horizon, shedding_period_steps)
        )

        # Check 2: POD energy (per variable if data provided)
        if pod_energy_data is not None:
            for var, data in pod_energy_data.items():
                self.results.append(
                    check_pod_energy(
                        data["energy_fractions"],
                        data["r_used"],
                        var_label=var,
                    )
                )

        # Check 3: Parameter-to-sample ratio
        if n_model_params is not None and n_train_samples is not None:
            self.results.append(
                check_param_sample_ratio(n_model_params, n_train_samples)
            )

        # Check 4: Model architecture
        if all(v is not None for v in [d_model, n_layers, n_heads, latent_dim]):
            self.results.append(
                check_model_architecture(d_model, n_layers, n_heads, latent_dim)
            )

        # Check 5: Solid masking
        self.results.append(
            check_solid_masking(solid_masking_enabled, n_solid_cells, n_total_cells)
        )

        # Check 6: Latent dim reduction vs legacy 450
        if latent_dim is not None:
            self.results.append(
                check_latent_dim_reduction(old_latent_dim=450, new_latent_dim=latent_dim)
            )

        return self.print_summary()

    def print_summary(self) -> Tuple[int, int]:
        """Print check results summary.

        Returns
        -------
        (n_passed, n_failed)
        """
        print()
        print("─" * 70)
        print("  Pre-Training Configuration Validation")
        print("─" * 70)

        n_passed = 0
        n_failed = 0
        for r in self.results:
            status = "PASS ✓" if r.passed else "FAIL ✗"
            print(f"  [{status}]  {r.name}")
            print(f"           {r.message}")
            if r.passed:
                n_passed += 1
            else:
                n_failed += 1

        print("─" * 70)
        print(f"  {n_passed} passed  {n_failed} failed")
        if n_failed == 0:
            print("  All configuration checks passed. ✓")
        else:
            print(f"  {n_failed} check(s) FAILED — fix before training!")
        print("─" * 70)
        print()

        return n_passed, n_failed
