"""
Train an A2C agent on GradingAdjustmentEnv (same setup as PPO).

Prerequisites:
    Place ``qry6114_2025f_autograding_final.xlsx`` in ``data/`` (see README).

Outputs:
    models/a2c_grading_agent.zip
    models/grading_meta.json  — written if missing; unchanged if already present
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from grading_env import GradingAdjustmentEnv, compute_norm_maxima, load_grading_frame
from train_ppo_agent import (
    RANDOM_STATE,
    TEST_SIZE,
    TOTAL_TIMESTEPS,
    _make_env,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
XLSX = DATA_DIR / "qry6114_2025f_autograding_final.xlsx"
MODEL_PATH = MODELS_DIR / "a2c_grading_agent"
META_OUT = MODELS_DIR / "grading_meta.json"


def _load_or_create_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    if META_OUT.is_file():
        meta = json.loads(META_OUT.read_text(encoding="utf-8"))
        train_idx = np.array(meta["train_indices"], dtype=np.int64)
        test_idx = np.array(meta["test_indices"], dtype=np.int64)
        max_resp = float(meta["max_resp_words"])
        max_fb = float(meta["max_fb_words"])
        return train_idx, test_idx, max_resp, max_fb, int(meta["n_rows_filtered"])

    n = len(df)
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_SIZE))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train_df = df.iloc[train_idx]
    max_resp, max_fb = compute_norm_maxima(train_df)
    return train_idx, test_idx, max_resp, max_fb, n


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not XLSX.is_file():
        raise FileNotFoundError(
            f"Missing {XLSX}. Copy the course export into data/ (see README)."
        )

    df = load_grading_frame(str(XLSX)).reset_index(drop=True)
    train_idx, test_idx, max_resp, max_fb, n = _load_or_create_split(df)

    train_env = GradingAdjustmentEnv(
        df, train_idx, max_resp, max_fb, rng=np.random.default_rng(RANDOM_STATE)
    )
    check_env(train_env, warn=True)

    venv = DummyVecEnv([_make_env(df, train_idx, max_resp, max_fb, RANDOM_STATE)])

    model = A2C(
        "MlpPolicy",
        venv,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        verbose=1,
        seed=RANDOM_STATE,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(str(MODEL_PATH))

    if not META_OUT.is_file():
        meta = {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "total_timesteps": TOTAL_TIMESTEPS,
            "n_rows_filtered": n,
            "max_resp_words": max_resp,
            "max_fb_words": max_fb,
            "train_indices": train_idx.astype(int).tolist(),
            "test_indices": test_idx.astype(int).tolist(),
            "model_file": "ppo_grading_agent.zip",
        }
        META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
