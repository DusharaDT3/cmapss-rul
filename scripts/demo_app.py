"""Local demonstration app for T5b/T6: predict remaining useful life for one engine in a browser.

Usage:
  python scripts/demo_app.py            # then open http://localhost:8050
  python scripts/demo_app.py --port 9000

Pick an example engine or upload a CSV (same format as scripts/predict.py). The app runs the same validation
and scope checks as the command-line tool, then shows the random forest, window MLP and LSTM predictions
(mean of three seeds each), plus the exploratory v2 history MLP (scripts/predict_neural_history.py), next to
a chart of the engine's scaled sensor history. Architecture, provenance, saved metrics and
error replays support the coursework demo. The web server and UI add no dependencies
beyond the existing model environment; all inference runs locally.
"""
import argparse
import json
import sys
import tempfile
import time
import threading
from urllib.parse import urlsplit

import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import predict as cli  # noqa: E402
from src import data  # noqa: E402
from src.inference import load_runs  # noqa: E402
from src.neural_history import HistoryRun  # noqa: E402
from src.demo_evidence import build_evidence  # noqa: E402

MODELS = [("random_forest", "Random forest"), ("mlp", "Window MLP"), ("lstm", "LSTM")]
CFG = data.load_config(ROOT / "configs" / "data_fd001.json")
SCALER = data.load_config(ROOT / CFG["preprocess_file"])
FEATURES, WINDOW, CAP = SCALER["features"], CFG["window"], CFG["rul_cap"]
EXAMPLES = sorted(p.name for p in (ROOT / "examples").glob("*.csv"))
RUNS = {}
HISTORY_DIR = ROOT / "experiments" / "neural_history_v2"
HISTORY = []
EVIDENCE = {}
TEST_FRAME = None
PREDICT_LOCK = threading.Lock()
UI = ROOT / "scripts" / "demo_ui"
ASSETS = {"/": (UI / "index.html", "text/html"),
          "/style.css": (UI / "style.css", "text/css"),
          "/app.js": (UI / "app.js", "application/javascript"),
          "/figures/learning.png": (ROOT / "results/figures/fig09_learning_curves.png", "image/png"),
          "/figures/history.png": (HISTORY_DIR / "figures/learning_curves.png", "image/png")}


def history_prediction(engine):
    """Exploratory v2 history MLP: needs the whole observed history from cycle 1 (as predict_neural_history.py)."""
    card = {"model": "History MLP", "id": "history", "phase": "exploratory",
            "note": "Exploratory follow-up; the official test set had already been seen"}
    if not HISTORY:
        return {**card, "unavailable": "checkpoints not found in experiments/neural_history_v2"}
    if int(engine["cycle"].iloc[0]) != 1:
        return {**card, "unavailable": "needs the complete history starting at cycle 1"}
    sc = HISTORY[0].sensor_scaler
    scaled = data.apply_scaler(engine, sc)[sc["features"]].to_numpy()
    lo, hi = cli.SCALED_LIMITS
    if any(v.min() < lo or v.max() > hi for v in scaled.T):
        return {**card, "unavailable": "sensor history outside the FD001 training range"}
    t0 = time.perf_counter()
    p = np.array([r.predict_frame(engine)[0][0] for r in HISTORY])
    rul = float(p.mean())
    text, level = advice(rul)
    return {**card, "rul": round(rul, 1), "lo": round(float(p.min()), 1), "hi": round(float(p.max()), 1),
            "ms": round((time.perf_counter() - t0) * 1000, 1), "advice": text, "level": level,
            "seed_values": [{"seed": r.seed, "rul": float(v)} for r, v in zip(HISTORY, p)]}


def advice(rul):
    if rul < 30:
        return "Low predicted remaining life", "red"
    if rul < 60:
        return "Moderate predicted remaining life", "amber"
    return "Higher predicted remaining life", "green"


def analyse(path=None, frame=None):
    """Return a JSON-ready dict for one engine file, mirroring scripts/predict.py."""
    try:
        engine = cli.validate(frame if frame is not None else cli.read_engine(path), FEATURES, WINDOW)
    except cli.InputError as e:
        return {"ok": False, "kind": "Invalid input", "code": e.code, "message": str(e)}
    scaled_all = data.apply_scaler(engine, SCALER)[FEATURES].to_numpy(np.float32)
    chart = {"cycles": engine["cycle"].astype(int).tolist(),
             "sensors": {f: np.round(scaled_all[:, i], 4).tolist() for i, f in enumerate(FEATURES)},
             "window": WINDOW, "limits": cli.SCALED_LIMITS}
    window = scaled_all[-WINDOW:]
    lo, hi = cli.SCALED_LIMITS
    outside = [f for f, col in zip(FEATURES, window.T) if col.min() < lo or col.max() > hi]
    info = {"unit": int(engine["unit"].iloc[0]), "n_cycles": len(engine), "last_cycle": int(engine["cycle"].iloc[-1])}
    if outside:
        return {"ok": False, "kind": "Out of scope", "code": 3, **info, "chart": chart,
                "message": f"sensor values far outside the FD001 training range ({', '.join(outside)}); the model was "
                           "trained on one operating condition and one fault mode and cannot be trusted for this engine"}
    preds = []
    for name, label in MODELS:
        t0 = time.perf_counter()
        p = np.array([run.predict(window[None])[0] for run in RUNS[name]])
        rul = float(p.mean())
        text, level = advice(rul)
        preds.append({"id": name, "phase": "original", "model": label,
                      "seed_values": [{"seed": r.seed, "rul": float(v)} for r, v in zip(RUNS[name], p)], "rul": round(rul, 1), "lo": round(float(p.min()), 1),
                      "hi": round(float(p.max()), 1), "ms": round((time.perf_counter() - t0) * 1000, 1),
                      "advice": text, "level": level})
    preds.append(history_prediction(engine))
    return {"ok": True, "code": 0, **info, "chart": chart, "predictions": preds, "cap": CAP}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", status=200):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if not ctype.startswith("image/") else ""))
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route in ASSETS:
            path, mime = ASSETS[route]
            if path.exists():
                return self._send(path.read_bytes(), mime)
        elif route == "/api/examples":
            return self._send(json.dumps(EXAMPLES))
        elif route == "/api/evidence":
            return self._send(json.dumps(EVIDENCE, allow_nan=False))
        self._send(json.dumps({"error": "Not found"}), status=404)

    def do_POST(self):
        if self.path != "/api/predict":
            return self._send(json.dumps({"error": "Not found"}), status=404)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 5 * 1024 * 1024:
                return self._send(json.dumps({"error": "Upload must be between 1 byte and 5 MB"}), status=413)
            req = json.loads(self.rfile.read(size))
            if not isinstance(req, dict):
                raise ValueError("Expected a JSON object")
            with PREDICT_LOCK:
                if 'case_unit' in req:
                    unit = req['case_unit']
                    if type(unit) is not int or str(unit) not in EVIDENCE['truth']:
                        raise ValueError("Unknown saved test engine")
                    result = analyse(frame=TEST_FRAME[TEST_FRAME.unit == unit])
                    result.update(source=f"Official FD001 test engine {unit}", truth=EVIDENCE['truth'][str(unit)])
                elif 'example' in req:
                    if req['example'] not in EXAMPLES:
                        raise ValueError("Unknown example")
                    path = ROOT / "examples" / req['example']
                    result = analyse(path)
                    result['source'] = f"examples/{req['example']}"
                    result['message'] = result.get('message', '').replace(str(path), req['example'])
                    if req['example'] == 'engine_normal.csv' and result.get('ok'):
                        # Truth is attached only to this known fixture, never to an uploaded engine ID.
                        result['truth'] = EVIDENCE['truth'].get(str(result['unit']))
                elif isinstance(req.get('text'), str):
                    name = Path(str(req.get('name', 'engine.csv'))).name
                    with tempfile.NamedTemporaryFile('w', suffix='.csv') as tmp:
                        tmp.write(req['text'])
                        tmp.flush()
                        result = analyse(Path(tmp.name))
                        result['message'] = result.get('message', '').replace(tmp.name, name)
                    result['source'] = f"Uploaded file: {name}"
                else:
                    raise ValueError("Select an example or upload a CSV")
            self._send(json.dumps(result, allow_nan=False))
        except (ValueError, TypeError) as exc:
            self._send(json.dumps({"error": str(exc)}), status=400)
        except Exception:
            import traceback
            traceback.print_exc()
            self._send(json.dumps({"error": "Prediction failed. Check the server terminal for details."}), status=500)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    global EVIDENCE, TEST_FRAME
    torch.set_num_threads(2)
    print("Loading saved models (three seeds each)...", flush=True)
    for name, _ in MODELS:
        RUNS[name] = load_runs(name)
    sel = HISTORY_DIR / "selection.json"
    if sel.exists():
        HISTORY.extend(HistoryRun(HISTORY_DIR / f"selected_neural_seed{s}.pt") for s in data.load_config(sel)["seeds"])
    EVIDENCE = build_evidence(RUNS, HISTORY, CFG)
    TEST_FRAME = data.load_raw("FD001", "test")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Demo running at http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
