# SPROUT draw-a-digit explorer

A mobile-friendly web UI: draw a digit, the trained **Conv-SPROUT** model (14×14)
infers it via the **real** forward pass, and a vertical pipeline shows what
happened — the learned filters, which feature maps fired, and the full sparse head
network with the neurons that activated.

Design: [`docs/superpowers/specs/2026-06-19-sprout-draw-digit-explorer-design.md`](../docs/superpowers/specs/2026-06-19-sprout-draw-digit-explorer-design.md)

## Run it

```bash
# deps (beyond the project's numpy / scikit-learn / pillow):
.venv/bin/pip install fastapi "uvicorn[standard]" httpx

# launch (serves on 0.0.0.0:8000 by default)
.venv/bin/python -m webapp.server
```

Then open **http://localhost:8000** on the same machine, or
**http://<this-machine-ip>:8000** from your phone on the same LAN / Tailscale.

Override host/port with the `HOST` / `PORT` env vars.

## Retrain the model

The committed `webapp/model/conv_sprout.pkl` (≈93% MNIST-14 test accuracy,
self-sized to ~10 filters) is produced by:

```bash
.venv/bin/python -m webapp.train_export --steps 20000 --n-train 12000
```

This reproduces the promoted `conv-sprout` arm and bundles the model with the input
**scaler** (the per-pixel train mean/std a drawing must be standardized by) and meta.

## How it fits together

| Module | Role |
|---|---|
| `serialize.py` | save/load the `ConvModel` + `Scaler` + meta |
| `preprocess.py` | drawing canvas → standardized 14×14 (crop, center-of-mass, resize, 2×2 pool, z-score) — matches MNIST's own normalization |
| `infer.py` | run the real `ConvModel.forward`, package prediction + filters + feature maps + the full head graph (per-neuron activation, per-synapse weight/confidence) |
| `server.py` | FastAPI: `GET /` (frontend), `GET /meta`, `POST /infer` |
| `static/` | the pipeline UI (canvas, prediction, filters, feature maps, network graph) |
| `train_export.py` | train Conv-SPROUT on MNIST-14 and export the artifact |

The forward pass lives entirely in the gradient-checked `sprout` package — the web
layer only reads its state, so the visualization is faithful by construction.
