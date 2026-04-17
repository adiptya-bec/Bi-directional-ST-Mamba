"""
model.py — TransformerROM_Lite: lightweight Transformer for POD-ROM forecasting.

Change 7: Reduce Model Size
  - Old: Transformer 25.2M params, d_model=384, n_layers=6
  - New: TransformerROM_Lite ~120–200K params, d_model=128, n_layers=2
  - Rationale: 2-step periodic signal + 300-step training set requires minimal
    capacity; 100× parameter reduction prevents overfitting
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# TransformerROM_Lite (Change 7)
# ---------------------------------------------------------------------------
class TransformerROM_Lite(nn.Module):
    """Lightweight Transformer for POD-ROM coefficient forecasting.

    Architecture:
      1. Linear input projection: latent_dim → d_model
      2. Sinusoidal positional encoding
      3. N Transformer encoder layers (causal / full attention)
      4. Linear output projection: d_model → latent_dim (1-step ahead)

    Parameter count with defaults (d_model=128, n_layers=2, ff_mult=4):
      ≈ latent_dim * d_model * 2        (in/out projections)
      + n_layers * (4 * d_model^2 * 2)  (attention + FFN)
      ≈ 110*128*2 + 2*(4*128^2*2) = ~56K + ~262K ≈ ~320K params

    Parameters
    ----------
    latent_dim : total POD latent dimension (r_p + r_u + r_v = 110)
    d_model : transformer hidden dimension (default 128)
    n_layers : number of transformer encoder layers (default 2)
    n_heads : number of attention heads (default 8; d_model must be divisible)
    ff_mult : feed-forward multiplier (default 4)
    dropout : dropout probability (default 0.1)
    context_length : maximum input sequence length
    causal : if True, use causal (autoregressive) attention mask
    """

    def __init__(
        self,
        latent_dim: int = 110,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        context_length: int = 150,
        causal: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.context_length = context_length
        self.causal = causal

        # Input projection
        self.input_proj = nn.Linear(latent_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPE(d_model, max_len=max(context_length + 1, 512), dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection: predict 1 step ahead
        self.output_proj = nn.Linear(d_model, latent_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked)."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, latent_dim) input latent sequence

        Returns
        -------
        out : (B, 1, latent_dim) prediction for the next step
        """
        B, T, D = x.shape

        # Input projection + positional encoding
        h = self.input_proj(x)         # (B, T, d_model)
        h = self.pos_enc(h)            # (B, T, d_model)

        # Causal mask for autoregressive attention
        attn_mask = self._causal_mask(T, x.device) if self.causal else None

        # Transformer encoder
        h = self.transformer(h, mask=attn_mask)  # (B, T, d_model)

        # Predict next step from the last token representation
        z_next = self.output_proj(h[:, -1:, :])  # (B, 1, d_model) → (B, 1, latent_dim)

        return z_next  # (B, 1, latent_dim)


# ---------------------------------------------------------------------------
# LSTM baseline (lightweight version)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """Lightweight LSTM for POD-ROM coefficient forecasting.

    Parameters
    ----------
    latent_dim : total POD latent dimension
    hidden_dim : LSTM hidden size
    n_layers : number of LSTM layers
    dropout : dropout probability
    """

    def __init__(
        self,
        latent_dim: int = 110,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, latent_dim) → (B, 1, latent_dim)"""
        out, _ = self.lstm(x)            # (B, T, hidden_dim)
        z_next = self.output_proj(out[:, -1:, :])  # (B, 1, latent_dim)
        return z_next


# ---------------------------------------------------------------------------
# S4-Lite / Mamba fallback (lightweight SSM)
# ---------------------------------------------------------------------------
class S4LiteBlock(nn.Module):
    """Minimal S4-inspired block using diagonal state-space approximation."""

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input/output projections
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Diagonal state-space parameters (simplified)
        self.A = nn.Parameter(torch.randn(d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model) → (B, T, d_model)"""
        residual = x
        x = self.norm(x)

        # Gate mechanism (simplified)
        xz = self.in_proj(x)  # (B, T, 2*d_model)
        x_gate, z_gate = xz.chunk(2, dim=-1)
        x_gate = F.silu(x_gate)
        z_gate = torch.sigmoid(z_gate)

        out = self.out_proj(x_gate * z_gate)
        return residual + self.dropout(out)


class MambaLiteModel(nn.Module):
    """Lightweight Mamba-inspired model for POD-ROM forecasting."""

    def __init__(
        self,
        latent_dim: int = 110,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList(
            [S4LiteBlock(d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, latent_dim) → (B, 1, latent_dim)"""
        h = self.input_proj(x)  # (B, T, d_model)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        z_next = self.output_proj(h[:, -1:, :])  # (B, 1, latent_dim)
        return z_next


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def make_model(
    model_type: str,
    latent_dim: int = 110,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 8,
    ff_mult: int = 4,
    dropout: float = 0.1,
    context_length: int = 150,
    **kwargs,
) -> nn.Module:
    """Factory function to create a model by type name.

    Parameters
    ----------
    model_type : 'TransformerROM_Lite', 'LSTM', or 'MambaLite'
    latent_dim : POD latent dimension
    d_model : hidden dimension
    n_layers : number of layers
    n_heads : attention heads (Transformer only)
    ff_mult : feed-forward multiplier (Transformer only)
    dropout : dropout probability
    context_length : input sequence length

    Returns
    -------
    model : nn.Module
    """
    if model_type in ("TransformerROM_Lite", "Transformer"):
        return TransformerROM_Lite(
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            context_length=context_length,
        )
    elif model_type == "LSTM":
        return LSTMModel(
            latent_dim=latent_dim,
            hidden_dim=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
    elif model_type in ("MambaLite", "Mamba"):
        return MambaLiteModel(
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
