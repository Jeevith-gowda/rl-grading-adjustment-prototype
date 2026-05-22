"""
Train a PPO agent on GradingAdjustmentEnv to mimic TA score adjustments.

Prerequisites:
    Place ``qry6114_2025f_autograding_final.xlsx`` in ``data/`` (see README).

Outputs:
    models/ppo_grading_agent.zip  — saved PPO policy (Stable-Baselines3)
    models/grading_meta.json      — train/test indices + word-count caps for evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from grading_env import GradingAdjustmentEnv, compute_norm_maxima, load_grading_frame

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
XLSX = DATA_DIR / "qry6114_2025f_autograding_final.xlsx"
MODEL_PATH = MODELS_DIR / "ppo_grading_agent"
META_OUT = MODELS_DIR / "grading_meta.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOTAL_TIMESTEPS = 50_000


def _make_env(df: pd.DataFrame, idx: np.ndarray, max_r: float, max_f: float, seed: int):
    """Factory for DummyVecEnv (SB3 expects a callable returning a fresh env)."""

    def _init():
        rng = np.random.default_rng(seed)
        return GradingAdjustmentEnv(df, idx, max_r, max_f, rng=rng)

    return _init


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not XLSX.is_file():
        raise FileNotFoundError(
            f"Missing {XLSX}. Copy the course export into data/ (see README)."
        )

    df = load_grading_frame(str(XLSX))
    df = df.reset_index(drop=True)
    n = len(df)
    if n < 50:
        raise RuntimeError(f"Too few rows after filtering: {n}")

    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_SIZE))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_df = df.iloc[train_idx]
    max_resp, max_fb = compute_norm_maxima(train_df)

    train_env = GradingAdjustmentEnv(
        df, train_idx, max_resp, max_fb, rng=np.random.default_rng(RANDOM_STATE)
    )
    check_env(train_env, warn=True)

    venv = DummyVecEnv([_make_env(df, train_idx, max_resp, max_fb, RANDOM_STATE)])

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        seed=RANDOM_STATE,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(str(MODEL_PATH))

    meta = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_rows_filtered": int(n),
        "max_resp_words": max_resp,
        "max_fb_words": max_fb,
        "train_indices": train_idx.astype(int).tolist(),
        "test_indices": test_idx.astype(int).tolist(),
        "model_file": MODEL_PATH.name + ".zip",
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}.zip")
    print(f"Saved metadata to {META_OUT}")


if __name__ == "__main__":
    main()
