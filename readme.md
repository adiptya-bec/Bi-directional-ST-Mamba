# Bi-directional ST Mamba

## Notebooks

### `bi_directional_st_mamba_hdf5_pipeline.ipynb` — Single-Variable HDF5 Pipeline

Production-ready notebook for training LSTM, Transformer, and **Bi-directional ST-Mamba** models
on one variable extracted from a large multi-variable HDF5 file using regional POD compression.
Designed for **Kaggle 30 GB RAM / 2×T4** environments.

#### Architecture

The pipeline implements the full Bi-directional ST-Mamba spatial-temporal design:

| Component | Role |
|---|---|
| **CoordEmbedding** | Two-layer SiLU MLP mapping each region's normalised (x, y) centroid to a `d_model`-dim vector. Weights stored as `register_buffer` for auto-GPU migration. |
| **SpatialAttentionEncoder** | Pre-LN Transformer operating across the N_REGIONS spatial tokens at every timestep. Enables each region to attend to all others, learning upstream/correlated partner relationships. Preserves `(B, T, R_TOTAL)` shape contract. |
| **SpatialTemporalModel** | Thin wrapper: `spatial_encoder(x)` → `temporal_model(x)`. Works with LSTM, Transformer, and BiSTMamba temporal backends without modification. |

#### Expected Input Format

```
HDF5 file structure:
  step_0000  — shape (N_CELLS, 3)   [pressure, u_velocity, v_velocity]
  step_0001  — shape (N_CELLS, 3)
  ...
  step_NNNN  — shape (N_CELLS, 3)

coords.npy   — shape (N_CELLS, 2)   [x, y] cell centroids
```

The notebook loads **one variable at a time** (`HDF5_VAR` config) to keep peak host RAM within
limits. Loading is done step-by-step (streaming) without reading unused variable columns.

#### Kaggle Hardware Assumptions

- **RAM**: 30 GB. For a 15 GB source file with 3 variables, loading one variable ≈ 5 GB float32.
  In-place normalisation avoids a second full copy, keeping peak ≈ 5–7 GB for the raw field.
- **GPU**: 2×T4 (15 GB VRAM each). `DataParallel` is applied automatically if two GPUs are detected.
- **Recommended for very large data**: Reduce `N_REGIONS` (e.g. 10–20), `POD_MODES` (e.g. 20–30),
  `WIN_LEN` (e.g. 30–50), and `BATCH_SIZE` (e.g. 64). Disable `persistent_workers` or set
  `num_workers=0` if the Kaggle kernel OOM-kills worker processes.

#### Hyperparameter Reference

| Parameter | Default | Description |
|---|---|---|
| `N_REGIONS` | 20 | k-means spatial regions |
| `POD_MODES` | 30 | POD modes per region (R_TOTAL = N_REGIONS × POD_MODES) |
| `WIN_LEN` | 50 | Input window length |
| `HORIZON` | 20 | Prediction horizon |
| `D_MODEL` | 256 | Model hidden dimension |
| `N_SPATIAL_HEADS` | 4 | Attention heads in SpatialAttentionEncoder |
| `N_SPATIAL_LAYERS` | 2 | Transformer layers in SpatialAttentionEncoder |
| `EPOCHS` | 50 | Maximum training epochs |
| `PATIENCE` | 10 | Early stopping patience |

---

### Other Notebooks

| Notebook | Description |
|---|---|
| `st_mamba_production.ipynb` | Full production ST-Mamba with patch embedding, spatial refinement |
| `inference_st_mamba_v2.ipynb` | Inference only, with percentage-based input control |
| `airfoil_st_forecasting_kaggle.ipynb` | Airfoil CFD forecasting, CSV/memmap pipeline |
| `train_spatiotemporal_unified.ipynb` | Unified training script |
| `notebooks/spatial_pod_gcn_temporal_heads_full.ipynb` | GCN-based spatial encoder with POD |
