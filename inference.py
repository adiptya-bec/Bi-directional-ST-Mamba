"""
inference.py — Iterative one-step autoregressive rollout.

Change 4: Iterative One-Step Inference
  - Old: 50-step direct horizon prediction (independent blocks)
  - New: One-step autoregressive rollout (phase-aware, error propagates)
  - Reveals true prediction quality; enables phase/amplitude metrics
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Iterative one-step rollout (Change 4)
# ---------------------------------------------------------------------------
@torch.no_grad()
def inference_iterative_rollout(
    model: nn.Module,
    Z_context: np.ndarray,
    n_steps: int,
    context_length: int,
    device: torch.device,
    use_amp: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Autoregressive one-step rollout for CFD forecasting.

    At each step:
      1. Take the last `context_length` latent states as input
      2. Predict the next 1 latent state
      3. Append prediction to history, advance window

    Parameters
    ----------
    model : trained temporal model (expects (B, T, D) → (B, 1, D) or (B, D))
    Z_context : (T_context, D) initial context window (normalised latent space)
    n_steps : number of future steps to predict
    context_length : input sequence length
    device : torch device
    use_amp : use automatic mixed precision
    verbose : print progress

    Returns
    -------
    Z_pred : (n_steps, D) predicted latent states
    """
    model.eval()
    D = Z_context.shape[1]

    # Build rolling context buffer
    # We need at least context_length steps to start
    T_ctx = Z_context.shape[0]
    if T_ctx < context_length:
        raise ValueError(
            f"Z_context has only {T_ctx} steps; need at least context_length={context_length}"
        )

    # Use the last context_length steps as initial window
    window = Z_context[-context_length:].copy()  # (context_length, D)
    predictions = []

    t0 = time.time()
    for step in range(n_steps):
        # Prepare input tensor: (1, context_length, D)
        x = torch.from_numpy(window[None]).float().to(device)  # (1, T, D)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = model(x)
        else:
            out = model(x)

        # Handle different output shapes
        if isinstance(out, (tuple, list)):
            out = out[0]
        out_np = out.squeeze(0).cpu().numpy()  # (1, D) or (D,)  — model outputs fp32
        if out_np.ndim == 2:
            z_next = out_np[0]   # take first (and only) predicted step
        else:
            z_next = out_np      # (D,)

        predictions.append(z_next)

        # Slide window: drop oldest, append new prediction
        window = np.roll(window, -1, axis=0)
        window[-1] = z_next

        if verbose and (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Rollout step {step+1:4d}/{n_steps}  ({elapsed:.1f}s)", end="\r")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Rollout complete: {n_steps} steps in {elapsed:.1f}s  ({elapsed/n_steps*1000:.1f} ms/step)")

    return np.array(predictions, dtype=np.float32)  # (n_steps, D)


# ---------------------------------------------------------------------------
# Legacy direct-horizon inference (kept for comparison)
# ---------------------------------------------------------------------------
@torch.no_grad()
def inference_direct(
    model: nn.Module,
    test_loader,
    device: torch.device,
    use_amp: bool = True,
) -> np.ndarray:
    """Direct (non-autoregressive) inference over a DataLoader.

    Parameters
    ----------
    model : trained temporal model
    test_loader : DataLoader yielding (x, y) batches
    device : torch device
    use_amp : use automatic mixed precision

    Returns
    -------
    preds : (N_windows, H, D) predicted latent states
    """
    model.eval()
    all_preds = []

    for x, _ in test_loader:
        x = x.to(device)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = model(x)
        else:
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        all_preds.append(out.cpu().float().numpy())

    return np.concatenate(all_preds, axis=0)  # (N_windows, H, D)


# ---------------------------------------------------------------------------
# Reconstruct full field from predicted latent states
# ---------------------------------------------------------------------------
def reconstruct_from_latent(
    Z_pred: np.ndarray,
    pods: dict,
    n_cells: int,
    latent_dim_per_var: dict,
) -> dict:
    """Reconstruct physical fields from predicted latent states.

    Parameters
    ----------
    Z_pred : (T_pred, total_r) predicted POD coefficients
    pods : dict[var] = pod_dict from pod_compression.compute_pod
    n_cells : total number of spatial cells
    latent_dim_per_var : dict[var] = r  (ranks used per variable)

    Returns
    -------
    fields : dict[var] → (N_cells, T_pred) reconstructed physical fields
    """
    from pod_compression import pod_decode  # local import to avoid circular deps

    fields = {}
    offset = 0
    for var, r in latent_dim_per_var.items():
        A = Z_pred[:, offset : offset + r]  # (T_pred, r)
        fields[var] = pod_decode(A, pods[var], n_cells)  # (N_cells, T_pred)
        offset += r

    return fields


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------
def run_inference_pipeline(
    model: nn.Module,
    Z_all: np.ndarray,
    train_steps: int,
    predict_steps: int,
    context_length: int,
    device: torch.device,
    use_amp: bool = True,
    mode: str = "iterative_rollout",
    test_loader=None,
) -> Tuple[np.ndarray, float]:
    """Run full inference pipeline.

    Parameters
    ----------
    model : trained model
    Z_all : (T_total, D) full latent sequence (train + predict)
    train_steps : number of training steps (context starts here)
    predict_steps : number of steps to forecast
    context_length : input window length
    device : torch device
    use_amp : mixed precision flag
    mode : "iterative_rollout" (default) or "direct"
    test_loader : DataLoader for direct mode

    Returns
    -------
    Z_pred : (predict_steps, D) predicted latent states
    elapsed : wall-clock time in seconds
    """
    t0 = time.time()

    if mode == "iterative_rollout":
        # Use training data as initial context window
        Z_context = Z_all[:train_steps]  # (T_train, D)
        Z_pred = inference_iterative_rollout(
            model,
            Z_context=Z_context,
            n_steps=predict_steps,
            context_length=context_length,
            device=device,
            use_amp=use_amp,
        )
    elif mode == "direct":
        if test_loader is None:
            raise ValueError("test_loader required for direct inference mode")
        preds = inference_direct(model, test_loader, device, use_amp)
        # Flatten windows into sequential prediction (take non-overlapping)
        Z_pred = preds.reshape(-1, preds.shape[-1])[:predict_steps]
    else:
        raise ValueError(f"Unknown inference mode: {mode!r}")

    elapsed = time.time() - t0
    return Z_pred, elapsed
