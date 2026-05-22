"""
Train a Random Forest classifier to predict TA adjustment actions from observations.

Prerequisites:
    Place ``qry6114_2025f_autograding_final.xlsx`` in ``data/`` (see README).
    Run ``train_ppo_agent.py`` first (or ensure ``models/grading_meta.json`` exists).

Outputs:
    models/random_forest_model.pkl
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from grading_env import build_observation, load_grading_frame

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
XLSX = DATA_DIR / "qry6114_2025f_autograding_final.xlsx"
META_OUT = MODELS_DIR / "grading_meta.json"
MODEL_OUT = MODELS_DIR / "random_forest_model.pkl"


def _action_label(ta_minus_ai: float) -> int:
    """Map actual adjustment to discrete action 0–4 (flag if TA moved > 3)."""
    if ta_minus_ai > 3:
        return 4
    return int(np.clip(np.round(ta_minus_ai), 0, 3))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not XLSX.is_file():
        raise FileNotFoundError(
            f"Missing {XLSX}. Copy the course export into data/ (see README)."
        )
    if not META_OUT.is_file():
        raise FileNotFoundError(
            f"Missing {META_OUT.name}. Run train_ppo_agent.py first."
        )

    meta = json.loads(META_OUT.read_text(encoding="utf-8"))
    df = load_grading_frame(str(XLSX)).reset_index(drop=True)
    train_idx = np.array(meta["train_indices"], dtype=np.int64)
    max_resp = float(meta["max_resp_words"])
    max_fb = float(meta["max_fb_words"])

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for i in train_idx:
        row = df.iloc[int(i)]
        obs = build_observation(row, max_resp, max_fb)
        ai = float(pd.to_numeric(row["ai_score"], errors="coerce"))
        ta = float(pd.to_numeric(row["TA_Grade"], errors="coerce"))
        X_list.append(obs)
        y_list.append(_action_label(ta - ai))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=int(meta["random_state"]),
        class_weight="balanced",
    )
    clf.fit(X, y)

    with MODEL_OUT.open("wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
