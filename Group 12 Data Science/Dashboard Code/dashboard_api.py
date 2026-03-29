"""
Local dashboard ML API (no Flask/FastAPI required).

- Trains 5 models from `train.csv` / `test.csv` in this folder
- Serves:
  - GET  /                      -> obesity_dashboard.html
  - GET  /<any file>            -> static files from this folder (js/css/html)
  - POST /api/predict           -> JSON prediction using selected model

Run:
  python dashboard_api.py
Then open:
  http://127.0.0.1:8000/obesity_dashboard.html
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
TARGET = "NObeyesdad_enc"

CLASS_NAMES = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight Level I",
    "Overweight Level II",
    "Obesity Type I",
    "Obesity Type II",
    "Obesity Type III",
]


def _load_xy() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing {TRAIN_CSV} or {TEST_CSV}")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Convert boolean to int (matches your `model.py`)
    bool_cols = train.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        train[bool_cols] = train[bool_cols].astype(int)
        test[bool_cols] = test[bool_cols].astype(int)

    if TARGET not in train.columns:
        raise ValueError(f"Target `{TARGET}` not found in train.csv")
    if TARGET not in test.columns:
        raise ValueError(f"Target `{TARGET}` not found in test.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].astype(int)
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].astype(int)
    return X_train, y_train, X_test, y_test


def train_models() -> Tuple[Dict[str, object], list]:
    X_train, y_train, X_test, y_test = _load_xy()
    feature_cols = list(X_train.columns)
    n_classes = int(max(y_train.max(), y_test.max()) + 1)

    # Train 5 models (based on your `model.py`, minus plotting).
    models: Dict[str, object] = {}

    models["Decision Tree"] = DecisionTreeClassifier(max_depth=8, random_state=42).fit(X_train, y_train)
    models["Random Forest"] = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
    models["KNN"] = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    # Enable probability so we can return probabilities to the dashboard
    models["SVM"] = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42).fit(X_train, y_train)
    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    ).fit(X_train, y_train)

    # Quick sanity metrics in console
    print("=" * 70)
    print("Dashboard API models trained from train.csv / test.csv")
    print(f"Features: {len(feature_cols)} | Classes: {n_classes} | Train: {len(X_train)} | Test: {len(X_test)}")
    for name, m in models.items():
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"- {name:20s} test acc: {acc:.4f}")
    print("=" * 70)

    return models, feature_cols


MODELS, FEATURE_COLS = train_models()


def predict(model_name: str, features: Dict) -> Dict:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    # Build 1-row dataframe in the exact column order used in train.csv
    row = {col: features.get(col, 0) for col in FEATURE_COLS}
    X = pd.DataFrame([row], columns=FEATURE_COLS)

    model = MODELS[model_name]
    pred_class = int(model.predict(X)[0])

    # Probabilities if available; else fallback to 1-hot
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0].astype(float).tolist()
        # Some sklearn models only return probs for seen classes; normalize to 7
        if len(probs) != 7 and hasattr(model, "classes_"):
            full = [0.0] * 7
            for p, cls in zip(probs, model.classes_):
                full[int(cls)] = float(p)
            probs = full
    else:
        probs = [0.0] * 7
        probs[pred_class] = 1.0

    return {
        "model": model_name,
        "classIdx": pred_class,
        "className": CLASS_NAMES[pred_class] if 0 <= pred_class < len(CLASS_NAMES) else str(pred_class),
        "probs": probs,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Allow file:// dashboard or other origins to call API
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, "text/plain; charset=utf-8", b"")

    def do_POST(self):
        if self.path != "/api/predict":
            self._send(404, "application/json; charset=utf-8", b'{"error":"not found"}')
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            model_name = payload.get("model") or "Random Forest"
            features = payload.get("features") or {}
            result = predict(model_name, features)
            self._send(200, "application/json; charset=utf-8", json.dumps(result).encode("utf-8"))
        except Exception as e:
            self._send(400, "application/json; charset=utf-8", json.dumps({"error": str(e)}).encode("utf-8"))

    def do_GET(self):
        # Serve static files from this folder
        rel = self.path.lstrip("/") or "obesity_dashboard.html"
        # Basic safety: disallow parent traversal
        if ".." in rel or rel.startswith(("/", "\\")):
            self._send(400, "text/plain; charset=utf-8", b"bad path")
            return

        file_path = (ROOT / rel).resolve()
        if ROOT not in file_path.parents and file_path != ROOT:
            self._send(400, "text/plain; charset=utf-8", b"bad path")
            return

        if not file_path.exists() or not file_path.is_file():
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return

        ext = file_path.suffix.lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml; charset=utf-8",
        }.get(ext, "application/octet-stream")

        self._send(200, ctype, file_path.read_bytes())


def main():
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("DASH_PORT", "8000"))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving dashboard + API on http://{host}:{port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

