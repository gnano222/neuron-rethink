# SPROUT interactive explorer

An educational web UI for the SPROUT architecture. The main page is a **live
network trainer**: draw a 2-class dot dataset, hit Run, and watch a sparse
gradient-as-currency network wire itself to separate the classes — neurons,
synapses, confidences, prune/grow bursts and the decision boundary all update
live. Pause, step, restart, change the dataset or network size at any time.

Design: [`docs/superpowers/specs/2026-06-19-sprout-draw-digit-explorer-design.md`](../docs/superpowers/specs/2026-06-19-sprout-draw-digit-explorer-design.md)
(the original draw-a-digit explorer; the live trainer reuses the same server).

## Run it

```bash
.venv/bin/pip install fastapi "uvicorn[standard]" httpx     # one-time
.venv/bin/python -m webapp.server                          # serves 0.0.0.0:8000
```

Open **http://localhost:8000** (or `http://<machine-ip>:8000` from a phone on the
same LAN / Tailscale). Override host/port with `HOST` / `PORT`.

## The model

The live trainer uses the **promoted plain SPROUT architecture** — no convolution:
sparse self-rewiring `Network` with 2D confidence, phasic prune/grow at settledness
plateaus, the startle alarm, and the bounded grow scan (`phasic-startle-k4`, the
`evals.spec._sparse` config). Network size (Small/Medium/Large topology) is
selectable in the UI. Training runs **per request**: the browser POSTs `/train/step`
repeatedly and renders each snapshot, so the polling loop is the training cadence
(pause = stop polling).

## Endpoints

| Endpoint | Role |
|---|---|
| `GET /` | the live-trainer UI |
| `POST /train/start` | build a net on the posted dots+labels, return first snapshot |
| `POST /train/step` | advance N steps, return snapshot (graph + decision boundary + metrics + events) |
| `POST /train/restart` | re-wire the same dots with a fresh seed |
| `GET /train/snapshot` | current snapshot without stepping |

## Modules

| File | Role |
|---|---|
| `session.py` | the live `TrainingSession` (start/step/restart, thread-locked) |
| `snapshot.py` | serialize a `Network` graph + decision-boundary grid to JSON |
| `server.py` | FastAPI app (training + the retained conv endpoints) |
| `static/` | the trainer UI (dot canvas, controls, evolving SVG graph) |

## Retained: the convolutional digit explorer

The original Conv-SPROUT draw-a-digit work is **kept, not deleted** — the conv model
and its server endpoints remain available:

- `serialize.py`, `preprocess.py`, `infer.py`, `train_export.py`, `model/conv_sprout.pkl`
- `POST /infer`, `GET /graph`, `GET /meta` (the conv digit pipeline)
- `sprout/conv*.py` (the convolutional core)

Retrain the conv model with `python -m webapp.train_export`.
