# SPROUT draw-a-digit explorer — design

**Date:** 2026-06-19 · **Branch:** sprout-v1

## Goal

An interactive, mobile-friendly web UI: the user draws a digit on a canvas, the
trained Conv-SPROUT model infers which digit it is, and the page visualizes the
**full network with the neurons that fired highlighted** — so you can see what is
happening under the hood.

## Locked decisions

- **Model:** Conv-SPROUT at 14×14 (the promoted MNIST baseline) — 12-slot filter
  economy (3×3 kernels, self-sizes to ~7), ReLU + 2×2 maxpool → 6×6 feature map
  per filter → 432-dim head input → 32 hidden → 10 outputs.
- **Runtime:** local Python (FastAPI) server running the **real** `ConvModel.forward`
  (single source of truth — no JS reimplementation). Accessed from the phone over
  the LAN / Tailscale.
- **Layout:** vertical "pipeline" — scroll top-to-bottom to follow the signal.
- **Inference:** live (debounced ~150ms after each stroke) **and** an explicit
  Predict button.
- **Model artifact:** trained once and committed to git (~tens of KB) so the app
  runs with no setup; `train_export.py` regenerates it.
- **Dependencies:** add `fastapi` + `uvicorn` to the project `.venv`.
- **Frontend:** vanilla HTML / CSS / JS + `<canvas>` — no framework, no build step.

## Architecture — three pieces

1. **Model artifact** — a serialized `ConvModel` (filter economy `theta`/`active`
   + the sparse head `Network`) plus the **input scaler** (per-pixel train μ and σ
   over the 196 pixels) and small metadata (side=14, kh=kw=3, pool=2). Produced by
   `train_export.py`.
2. **Backend** (`webapp/server.py`) — FastAPI app. Loads the artifact once at
   startup. `GET /` serves the static frontend; `POST /infer` runs the real forward
   pass and returns one JSON payload with everything the visualization needs.
3. **Frontend** (`webapp/static/`) — one page implementing the pipeline layout and
   rendering the `/infer` JSON.

## The make-or-break detail: matching the training distribution

A drawn digit is only recognized if it enters the model exactly as MNIST training
data did. The model was trained on data from `get_dataset("mnist")`: 28×28 →
`_downsample_2x2` to 14×14 → `_standardize_on_train` (per-pixel `(x − μ)/(σ + ε)`).

The drawing → model chain in `preprocess.py`:

```
canvas grayscale (NxN, ink = bright on black)
  → crop to the ink bounding box
  → recenter by center-of-mass on a square field   (MNIST is pre-centered)
  → resize to 28×28
  → 2×2 mean-pool to 14×14  (reuse sprout.datasets._downsample_2x2)
  → standardize with the SAVED train μ/σ            (NOT recomputed per-drawing)
  → flatten to length-196, reshape to 14×14
```

Centering by center-of-mass is required: a digit drawn in a corner would otherwise
fall outside the distribution. The frontend captures ink as white-on-black to match
MNIST polarity (bright digit on dark background).

## Data contract — `POST /infer`

**Request:** `{ "pixels": [[...grayscale 0..1...]], "size": N }` — the raw square
canvas grayscale (e.g. 280×280), white digit on black.

**Response:**

```jsonc
{
  "prediction": 7,
  "probs": [0.01, 0.00, 0.02, ...],          // length 10, sums to 1
  "input_14x14": [[...]],                      // the standardized image fed in
  "filters": [                                 // one entry per filter slot
    { "slot": 0, "active": true, "kernel": [[...3x3...]] },
    { "slot": 5, "active": false, "kernel": null }
  ],
  "feature_maps": [                            // active filters only
    { "slot": 0, "map": [[...6x6 pooled activations...]] }
  ],
  "graph": {
    "neurons":  [ { "id": 0, "layer": 0, "x": 0.0, "y": -3.5, "act": 0.0 }, ... ],
    "synapses": [ { "pre": 0, "post": 440, "w": 0.21, "conf": 1.8 }, ... ]
  }
}
```

The `graph` describes the **full live head network** (every neuron and every live
synapse). `layer` distinguishes input (the 432 feature cells) / hidden (32) /
output (10). The frontend highlights neurons with `act > 0` and brightens the
edges on the active path; edge color encodes `conf` (blue = plastic → red = frozen).

`infer.py` builds this payload from a `ConvModel` and a standardized image. The
feature-map cells in `feature_maps` are exactly the head input neurons (layer 0 of
`graph`), so the visualization can reuse their positions.

## Frontend pipeline (the five bands of layout A)

1. **Canvas + controls** — square `<canvas>`, touch + mouse drawing, Clear /
   Predict. Debounced live inference after strokes.
2. **Prediction** — large predicted digit + 10 probability bars (the winning bar
   highlighted).
3. **① input 14×14** — the standardized image actually fed to the model.
4. **② learned filters** — the active 3×3 kernels as small diverging heatmaps
   (orange = positive tap, blue = negative); inactive/self-sized-away slots shown
   dimmed.
5. **③ feature maps** — per active filter, its 6×6 pooled map; lit cells = where
   that filter fired on this drawing.
6. **④ head network** — the sparse head graph: feature-map cells (input) → 32
   hidden → 10 digit outputs. The full network is drawn; fired neurons glow,
   active wires brighten, edge color = confidence. This is the payoff view.
7. **Legend** — brightness = activation, color = confidence.

Rendering note: the head input layer has 432 neurons. They are drawn as the
feature-map grids (band ③ doubles as the input column of band ④), not 432 loose
dots. Live synapses (a few hundred after pruning) are drawn as thin lines; the
active path is emphasized so the graph stays legible on a phone.

## File structure (new `webapp/` directory)

| File | Responsibility | Tested |
|---|---|---|
| `webapp/__init__.py` | package marker | — |
| `webapp/serialize.py` | `save_model(path, model, scaler, meta)` / `load_model(path)` — pickle the `ConvModel` + scaler + meta | yes (roundtrip) |
| `webapp/preprocess.py` | `to_model_input(pixels, size, scaler)` → standardized 14×14; center-of-mass, resize, pool, standardize | yes |
| `webapp/infer.py` | `run_inference(model, image14)` → the `/infer` payload dict | yes (parity) |
| `webapp/train_export.py` | train Conv-SPROUT on MNIST-14, compute scaler, write `webapp/model/conv_sprout.pkl` | — (script) |
| `webapp/server.py` | FastAPI app: load artifact, `GET /`, `POST /infer` | yes (smoke) |
| `webapp/static/index.html` | pipeline markup | — |
| `webapp/static/style.css` | mobile-first styling | — |
| `webapp/static/app.js` | canvas capture, fetch `/infer`, render the five bands + graph | — |
| `webapp/model/conv_sprout.pkl` | committed trained artifact | — |
| `tests/test_webapp_serialize.py` | save → load reproduces predictions | yes |
| `tests/test_webapp_preprocess.py` | centering + uses saved stats + output shape/range | yes |
| `tests/test_webapp_infer.py` | payload shape + prediction == `ConvModel.predict` | yes |
| `tests/test_webapp_server.py` | `TestClient`: `/` serves, `/infer` returns valid payload | yes |

## Testing approach (project TDD norm)

- **serialize:** save then load a tiny model; loaded model predicts identically to
  the original on a fixed input.
- **preprocess:** an off-center blob is recentered; output is 14×14; standardization
  uses the *saved* μ/σ (a constant-shift input changes the output, proving stats
  are not recomputed per-call).
- **infer (the key test):** on a standardized image, `run_inference` returns a
  payload whose `prediction`/`probs` equal `ConvModel.forward`/`predict` directly,
  and whose `graph` neuron/synapse counts match the head — proving the payload is a
  faithful mirror of the real forward pass.
- **server:** `TestClient` confirms `/` serves HTML and `/infer` accepts a blank
  canvas and returns a schema-valid payload.

No finite-difference gradient checks are needed: this is inference only and reuses
the already-gradient-checked `ConvModel.forward`.

## Out of scope (YAGNI)

- Live retraining / structural events in the browser (this is an inference + viz
  tool; the network is fixed at load time).
- Multiple model selection, datasets other than MNIST-14, or accounts/persistence.
- Public deployment / auth — it is a local tool reached over the LAN / Tailscale.

## Risks & mitigations

- **Drawn digits look out-of-distribution** → center-of-mass + bounding-box crop +
  white-on-black polarity; if accuracy is poor in practice, a thin Gaussian blur on
  the resized image (MNIST is anti-aliased) is the first knob to add.
- **432-input graph is cluttered on mobile** → render inputs as feature-map grids,
  draw only live synapses, emphasize the active path, keep zoom simple.
```
