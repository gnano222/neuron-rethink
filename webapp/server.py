"""FastAPI server for the SPROUT draw-a-digit explorer.

Loads the trained Conv-SPROUT artifact once, serves the static frontend at ``/``,
and runs the REAL forward pass on ``POST /infer``. The model path is the committed
``webapp/model/conv_sprout.pkl`` unless ``SPROUT_MODEL`` overrides it (used by
tests). Run it with::

    .venv/bin/python -m webapp.server            # http://0.0.0.0:8000
"""
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.infer import run_inference
from webapp.preprocess import to_model_input
from webapp.serialize import load_model

_HERE = os.path.dirname(__file__)
_STATIC = os.path.join(_HERE, "static")
_DEFAULT_MODEL = os.path.join(_HERE, "model", "conv_sprout.pkl")

app = FastAPI(title="SPROUT digit explorer")
_state: dict = {}


def _model_path() -> str:
    return os.environ.get("SPROUT_MODEL") or _DEFAULT_MODEL


def get_model():
    """Load (and cache) the model, scaler, meta. Re-reads if the path changed so a
    test can point SPROUT_MODEL at a fixture without process restart."""
    path = _model_path()
    if _state.get("path") != path:
        if not os.path.exists(path):
            raise HTTPException(503, f"model not found at {path}; run "
                                     "`python -m webapp.train_export` first")
        model, scaler, meta = load_model(path)
        _state.update(path=path, model=model, scaler=scaler, meta=meta)
    return _state["model"], _state["scaler"], _state["meta"]


class InferRequest(BaseModel):
    pixels: list[list[float]]      # square grayscale, ink bright (0..1 or 0..255)
    size: int | None = None


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/meta")
def meta():
    _, _, m = get_model()
    return m


@app.post("/infer")
def infer(req: InferRequest):
    model, scaler, m = get_model()
    pixels = np.asarray(req.pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[0] != pixels.shape[1]:
        raise HTTPException(400, "pixels must be a square 2D array")
    image14 = to_model_input(pixels, scaler, value_scale=m.get("value_scale", 255.0))
    payload = run_inference(model, image14)
    payload["model_meta"] = {"n_active_filters": int(model.conv.n_active),
                             "test_acc": m.get("test_acc")}
    return JSONResponse(payload)


app.mount("/static", StaticFiles(directory=_STATIC), name="static")


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
