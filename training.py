"""
training.py — Extended training loop with early stopping on amplitude plateau.

Change 9: Extended Training + Early Stopping
  - Old: Fixed 100 epochs
  - New: 500 epochs + early stopping on amplitude loss plateau (patience=50)
  - Impact: Train until convergence; stop when amplitude learning saturates
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loss import AdaptiveWeightedLoss


# ---------------------------------------------------------------------------
# Dataset for POD latent sequences
# ---------------------------------------------------------------------------
class PODLatentDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over normalised POD latent sequences.

    Each sample is a pair (x, y) where:
      x : (context_length, D) input window
      y : (1, D) next-step target  (one-step prediction)

    Parameters
    ----------
    Z : (T, D) latent sequence
    context_length : input window length
    start : start index in Z (inclusive)
    end : end index in Z (exclusive)
    """

    def __init__(
        self,
        Z: np.ndarray,
        context_length: int,
        start: int = 0,
        end: Optional[int] = None,
    ):
        super().__init__()
        self.Z = Z
        self.context_length = context_length
        self.start = start
        self.end = end if end is not None else len(Z)
        # Valid window start indices
        self.indices = list(range(start, self.end - context_length))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.indices[i]
        x = torch.from_numpy(self.Z[s : s + self.context_length].astype(np.float32))
        y = torch.from_numpy(self.Z[s + self.context_length : s + self.context_length + 1].astype(np.float32))
        return x, y


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------
def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    n_epochs: int,
    T_0: int = 50,
    lr_min: float = 1e-6,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler.

    Parameters
    ----------
    optimizer : torch optimizer
    scheduler_type : 'cosine_with_restarts', 'cosine', 'step', or 'none'
    n_epochs : total training epochs
    T_0 : restart period for cosine with restarts
    lr_min : minimum learning rate

    Returns
    -------
    scheduler or None
    """
    if scheduler_type == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, eta_min=lr_min
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr_min
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=T_0, gamma=0.5)
    else:
        return None


# ---------------------------------------------------------------------------
# Early stopping (Change 9)
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Early stopping based on a monitored metric.

    Stops training when the monitored metric has not improved by at least
    min_delta for patience consecutive epochs.

    Parameters
    ----------
    patience : epochs to wait without improvement before stopping
    min_delta : minimum improvement to count as progress
    mode : 'min' (lower is better) or 'max'
    """

    def __init__(self, patience: int = 50, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._counter: int = 0
        self.best_epoch: int = 0
        self.stopped: bool = False

    def step(self, value: float, epoch: int) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        value : current metric value
        epoch : current epoch number (0-indexed)

        Returns
        -------
        stop : True if training should stop
        """
        if self._best is None:
            self._best = value
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = value < self._best - self.min_delta
        else:
            improved = value > self._best + self.min_delta

        if improved:
            self._best = value
            self.best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.stopped = True
                return True
        return False


# ---------------------------------------------------------------------------
# Main training loop (Change 9)
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    criterion: AdaptiveWeightedLoss,
    device: torch.device,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 3e-4,
    lr_min: float = 1e-6,
    lr_scheduler_type: str = "cosine_with_restarts",
    T_0: int = 50,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    context_length: int = 150,
    use_amp: bool = True,
    early_stopping_cfg: Optional[Dict] = None,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "model",
    print_every: int = 10,
) -> Dict:
    """Train a temporal model on POD latent sequences.

    Parameters
    ----------
    model : nn.Module to train
    Z_train : (T_train, D) training latent sequence
    Z_val : (T_val, D) validation latent sequence  (or slice of Z_all)
    criterion : AdaptiveWeightedLoss instance
    device : torch device
    epochs : maximum training epochs (default 500)
    batch_size : training batch size
    lr : initial learning rate
    lr_min : minimum learning rate for scheduler
    lr_scheduler_type : scheduler type
    T_0 : cosine restart period (epochs)
    weight_decay : AdamW weight decay
    grad_clip : gradient norm clipping (0 = disabled)
    context_length : input window length
    use_amp : mixed-precision training
    early_stopping_cfg : dict with 'patience', 'monitor', 'min_delta' (optional)
    checkpoint_dir : directory to save best model checkpoint
    model_name : name prefix for checkpoint file
    print_every : print progress every N epochs

    Returns
    -------
    history : dict with training history arrays
    """
    # Datasets and loaders
    train_ds = PODLatentDataset(Z_train, context_length)
    val_ds = PODLatentDataset(Z_val, context_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Scheduler
    scheduler = make_scheduler(
        optimizer, lr_scheduler_type, n_epochs=epochs, T_0=T_0, lr_min=lr_min
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # Early stopping (Change 9)
    if early_stopping_cfg is None:
        early_stopping_cfg = {"patience": 50, "monitor": "amplitude_loss", "min_delta": 0.001}
    es = EarlyStopping(
        patience=early_stopping_cfg.get("patience", 50),
        min_delta=early_stopping_cfg.get("min_delta", 0.001),
        mode="min",
    )
    monitor_key = early_stopping_cfg.get("monitor", "amplitude_loss")

    # Best model tracking
    best_val_loss = float("inf")
    best_state = None

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_wmse": [], "train_amp": [], "train_tke_amp": [],
        "val_wmse": [], "val_amp": [], "val_tke_amp": [],
        "lr": [], "epoch": [],
    }

    print(f"  Training {model_name}  params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    t_start = time.time()

    for epoch in range(epochs):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        train_comps: Dict[str, List[float]] = {k: [] for k in ["total", "wmse", "amp", "tke_amp"]}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # x:(B,T,D), y:(B,1,D)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                pred = model(x)              # (B, 1, D)
                loss, comps = criterion(pred, y)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            for k in train_comps:
                train_comps[k].append(comps.get(k, comps.get("total", 0.0)))

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_comps: Dict[str, List[float]] = {k: [] for k in ["total", "wmse", "amp", "tke_amp"]}

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                    pred = model(x)
                    _, comps = criterion(pred, y)
                for k in val_comps:
                    val_comps[k].append(comps.get(k, comps.get("total", 0.0)))

        # ── Bookkeeping ──────────────────────────────────────────────────────
        tr_loss = float(np.mean(train_comps["total"])) if train_comps["total"] else float("nan")
        vl_loss = float(np.mean(val_comps["total"])) if val_comps["total"] else tr_loss
        tr_wmse = float(np.mean(train_comps["wmse"])) if train_comps["wmse"] else float("nan")
        vl_amp = float(np.mean(val_comps["amp"])) if val_comps["amp"] else float("nan")
        vl_tke = float(np.mean(val_comps["tke_amp"])) if val_comps["tke_amp"] else float("nan")
        cur_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_wmse"].append(tr_wmse)
        history["train_amp"].append(float(np.mean(train_comps["amp"])))
        history["train_tke_amp"].append(float(np.mean(train_comps["tke_amp"])))
        history["val_wmse"].append(float(np.mean(val_comps["wmse"])))
        history["val_amp"].append(vl_amp)
        history["val_tke_amp"].append(vl_tke)
        history["lr"].append(cur_lr)
        history["epoch"].append(epoch + 1)

        # Save best model
        if not (vl_loss != vl_loss) and vl_loss < best_val_loss:  # not nan check
            best_val_loss = vl_loss
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_dir is not None:
                ckpt_path = Path(checkpoint_dir) / f"best_{model_name}.pt"
                torch.save(best_state, ckpt_path)

        # Adaptive loss update (Change 8)
        boosted = criterion.update_adaptive(tr_wmse)
        if boosted:
            print(
                f"  [adaptive] Plateau detected at epoch {epoch+1}: "
                f"boosting amplitude weight to {criterion.amp_weight:.2f}"
            )

        # LR scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + 1)
            else:
                scheduler.step()

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(
                f"  ep {epoch+1:4d}  "
                f"train={tr_loss:.3e}  val={vl_loss:.3e}  "
                f"[wmse={tr_wmse:.3e} amp={float(np.mean(train_comps['amp'])):.3e} "
                f"tke_amp={float(np.mean(train_comps['tke_amp'])):.3e}]  "
                f"lr={cur_lr:.1e}  {elapsed:.0f}s"
            )

        # Early stopping check (Change 9)
        monitor_val = vl_amp if monitor_key == "amplitude_loss" else vl_loss
        if monitor_val is not None and not (isinstance(monitor_val, float) and monitor_val != monitor_val):  # not nan
            if es.step(monitor_val, epoch):
                print(
                    f"  Early stopping at epoch {epoch+1} "
                    f"(best epoch={es.best_epoch+1}, patience={es.patience})"
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Reloaded best weights from epoch {es.best_epoch+1}")

    return history
