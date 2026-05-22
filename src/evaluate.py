"""
Evaluate the trained PPO agent vs simple baselines on the held-out test split.

Reads:
    models/grading_meta.json   — must match the train run (indices + normalization)
    models/ppo_grading_agent.zip
    data/qry6114_2025f_autograding_final.xlsx

Writes:
    results/results.md

Windows note: import ``stable_baselines3`` (hence PyTorch) *before* ``sklearn`` so
OpenMP/MKL DLLs do not break ``torch`` initialization (WinError 1114).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO  # import before sklearn on Windows (DLL order)
from sklearn.linear_model import LinearRegression

from grading_env import (
    build_observation,
    compute_adjustment_reward,
    load_grading_frame,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
XLSX = DATA_DIR / "qry6114_2025f_autograding_final.xlsx"
META = MODELS_DIR / "grading_meta.json"
MODEL_ZIP = MODELS_DIR / "ppo_grading_agent.zip"
RESULTS = RESULTS_DIR / "results.md"


def reward_continuous_prediction(pred: float, actual: float) -> float:
    """
    Same tiered reward as numeric actions, for a continuous regression prediction.
    No flag semantics — large TA moves are judged purely by numeric error.
    """
    err = abs(float(pred) - float(actual))
    if err <= 0.5:
        return 10.0
    if err <= 1.0:
        return 5.0
    if err > 2.0:
        return -5.0
    return 0.0


def actual_adjustment_bucket(a: float) -> str:
    """Coarse bins for confusion-style tables."""
    if a <= 0:
        return "<=0"
    if a <= 1:
        return "(0,1]"
    if a <= 2:
        return "(1,2]"
    if a <= 3:
        return "(2,3]"
    return ">3"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not XLSX.is_file():
        raise FileNotFoundError(
            f"Missing {XLSX}. Copy the course export into data/ (see README)."
        )
    if not META.is_file():
        raise FileNotFoundError(f"Missing {META}. Run src/train_ppo_agent.py first.")

    meta = json.loads(META.read_text(encoding="utf-8"))
    df = load_grading_frame(str(XLSX)).reset_index(drop=True)
    test_idx = np.array(meta["test_indices"], dtype=np.int64)
    train_idx = np.array(meta["train_indices"], dtype=np.int64)
    max_resp = float(meta["max_resp_words"])
    max_fb = float(meta["max_fb_words"])

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    ai_train = pd.to_numeric(train_df["ai_score"], errors="coerce").to_numpy().reshape(-1, 1)
    y_train = (
        pd.to_numeric(train_df["TA_Grade"], errors="coerce")
        - pd.to_numeric(train_df["ai_score"], errors="coerce")
    ).to_numpy()

    reg = LinearRegression()
    reg.fit(ai_train, y_train)

    model = PPO.load(str(MODEL_ZIP))

    rewards_ppo: list[float] = []
    rewards_trust: list[float] = []
    rewards_plus2: list[float] = []
    rewards_reg: list[float] = []

    conf_ppo: dict[tuple[str, str], int] = defaultdict(int)
    per_q_ppo: dict[int, list[float]] = defaultdict(list)

    for _, row in test_df.iterrows():
        ai = float(pd.to_numeric(row["ai_score"], errors="coerce"))
        ta = float(pd.to_numeric(row["TA_Grade"], errors="coerce"))
        actual = ta - ai
        qid = int(pd.to_numeric(row["questions_id"], errors="coerce"))

        obs = build_observation(row, max_resp, max_fb)
        act_arr, _ = model.predict(obs[np.newaxis, :], deterministic=True)
        action = int(np.asarray(act_arr).reshape(-1)[0])

        r_ppo = compute_adjustment_reward(action, actual)
        r_trust = compute_adjustment_reward(0, actual)
        r_p2 = compute_adjustment_reward(2, actual)
        pred_reg = float(reg.predict(np.array([[ai]]))[0])
        r_reg = reward_continuous_prediction(pred_reg, actual)

        rewards_ppo.append(r_ppo)
        rewards_trust.append(r_trust)
        rewards_plus2.append(r_p2)
        rewards_reg.append(r_reg)

        pred_label = str(action) if action < 4 else "flag"
        conf_ppo[(pred_label, actual_adjustment_bucket(actual))] += 1
        per_q_ppo[qid].append(r_ppo)

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    lines: list[str] = []
    lines.append("# RL grading prototype — evaluation results")
    lines.append("")
    lines.append("**Setup:** PPO (`stable-baselines3`) on `GradingAdjustmentEnv`, 50k timesteps, ")
    lines.append(f"train/test = {1 - meta['test_size']:.0%}/{meta['test_size']:.0%} split (`random_state={meta['random_state']}`). ")
    lines.append(
        f"Filtered rows (excluding `questions_id == 9`): **{meta['n_rows_filtered']}**; **test rows:** **{len(test_df)}**. "
    )
    lines.append("Rows: `data/qry6114_2025f_autograding_final.xlsx`, **`questions_id != 9`**. ")
    lines.append("Reward tiers match `grading_env.compute_adjustment_reward`.")
    lines.append("")
    lines.append("## Mean reward on test set")
    lines.append("")
    lines.append("| Method | Mean reward |")
    lines.append("|--------|------------:|")
    lines.append(f"| PPO (trained) | {mean(rewards_ppo):.4f} |")
    lines.append(f"| Baseline: always trust AI (action 0) | {mean(rewards_trust):.4f} |")
    lines.append(f"| Baseline: always +2 (action 2) | {mean(rewards_plus2):.4f} |")
    lines.append(
        f"| Baseline: OLS regression (`TA−AI` ~ `ai_score` only) | {mean(rewards_reg):.4f} |"
    )
    lines.append("")

    lines.append("## Confusion-style counts (PPO predicted action vs actual adjustment bucket)")
    lines.append("")
    lines.append("Rows: **predicted action** (0–3 = point bump, `flag` = action 4). ")
    lines.append("Cols: **bucket of actual** `TA_Grade - ai_score`.")
    lines.append("")

    col_order = ["<=0", "(0,1]", "(1,2]", "(2,3]", ">3"]
    row_order = ["0", "1", "2", "3", "flag"]

    lines.append("| pred \\ actual | " + " | ".join(col_order) + " | **row sum** |")
    lines.append("|---|:---|:---|:---|:---|:---|---:|")
    for pr in row_order:
        cells = [str(conf_ppo.get((pr, c), 0)) for c in col_order]
        rsum = sum(conf_ppo.get((pr, c), 0) for c in col_order)
        lines.append(f"| **{pr}** | " + " | ".join(cells) + f" | {rsum} |")

    col_sums = [sum(conf_ppo.get((pr, c), 0) for pr in row_order) for c in col_order]
    lines.append(
        "| **col sum** | " + " | ".join(str(s) for s in col_sums) + f" | {sum(col_sums)} |"
    )
    lines.append("")

    lines.append("## Per-question mean reward (PPO on test rows)")
    lines.append("")
    lines.append("| questions_id | n (test) | mean reward |")
    lines.append("|-------------:|---------:|------------:|")
    for q in sorted(per_q_ppo.keys()):
        xs = per_q_ppo[q]
        lines.append(f"| {q} | {len(xs)} | {mean(xs):.4f} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Auto-generated by `src/evaluate.py`.*")

    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
