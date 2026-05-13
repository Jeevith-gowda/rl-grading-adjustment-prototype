# RL Grading Adjustment Prototype

This repository is a **research prototype** that uses reinforcement learning (PPO) to predict **how a teaching assistant (TA) would adjust** an AI-assigned score on student submissions. The agent observes lightweight features of each submission and chooses a discrete **adjustment policy**: trust the AI (0), add +1 / +2 / +3 points, or **flag** the case for human review when the TA’s eventual correction is large.

It is **not** production grading software; it demonstrates how policy-gradient RL can align with human correction patterns on a small, tabular dataset.

---

## Data

- **Source:** course export `qry6114_2025f_autograding_final.xlsx` (place this file in the project root — it is **gitignored** because it may contain student text and identifiers).
- **Rows used:** **1,317** submissions after:
  - dropping rows with missing `ai_score` or `TA_Grade`, and  
  - **excluding `questions_id == 9`**, where the TA scale (0–20) does not match the AI scale (0–10).  
- **Target:** `TA_Grade - ai_score` (TA adjustment on the same 0–10 band as the AI for remaining questions).

---

## Approach

| Component | Description |
|-----------|-------------|
| **Environment** | `GradingAdjustmentEnv` ([`grading_env.py`](grading_env.py)) — one-step episodes; observation includes normalized `ai_score`, one-hot `questions_id`, and normalized word counts for `response_txt` and `ai_feedback`. |
| **Actions** | `Discrete(5)`: 0 = no change, 1–3 = +1…+3, 4 = **flag** (rewarded when the true adjustment is **> 3**). |
| **Reward** | Shaped tiers: +10 within 0.5 of true adjustment, +5 within 1.0, −5 if error exceeds 2, +13 for a correct flag when the true adjustment exceeds 3 points. |
| **Algorithm** | **PPO** from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (`MlpPolicy`, ~50k timesteps, 80/20 train/test split, `random_state=42`). |

Training writes **`rl_grading_agent.zip`** and **`grading_meta.json`** (split + normalization constants). The committed zip/meta match the evaluation snapshot below.

---

## Results (test set, n = 263)

| Method | Mean reward |
|--------|------------:|
| **PPO (trained)** | **8.18** |
| Baseline: always trust AI (0 adjustment) | 1.44 |
| Baseline: always +2 | 2.70 |
| Baseline: OLS — predict adjustment from `ai_score` only | 7.15 |

Full tables (confusion-style counts, per-`questions_id` breakdown) are in [`results.md`](results.md).

---

## Key finding

**The AI is systematically harsh relative to the TA on harder, more open-ended algorithmic sketch questions** (e.g. asymptotic ordering, time-complexity analyses, heaps). That shows up as:

- **Low reward** for the “always trust AI” baseline — the TA often **raises** scores, so predicting zero adjustment is wrong much of the time.  
- The naive **“always +2”** baseline still underperforms PPO, but beats “trust AI,” consistent with a **positive mean** TA adjustment.  
- **Per-question PPO mean reward** is lowest on challenging items (e.g. questions **3, 7, 8** in the held-out split), where nuance dominates and the feature set is weakest.

Interpretation: the TA is correcting **upward** more than downward on this slice of the rubric; a fixed bump is crude, but a learned policy that uses question ID and text length signals helps.

---

## Per-question performance (summary)

From the snapshot in `results.md` (PPO mean reward on test rows only):

| questions_id | n (test) | mean reward |
|-------------:|---------:|------------:|
| 3 | 29 | 5.34 |
| 4 | 27 | 6.52 |
| 5 | 4 | 10.00 |
| 6 | 3 | 8.33 |
| 7 | 23 | 6.04 |
| 8 | 41 | 5.49 |
| 10 | 51 | 10.16 |
| 11 | 23 | 12.09 |
| 12 | 27 | 9.96 |
| 13 | 35 | 9.34 |

---

## How to run

```bash
cd rl_autograding_prototype   # or clone and enter repo root
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
# PyTorch installs with stable-baselines3; for CPU-only wheels see https://pytorch.org/get-started/locally/

# 1) Add the course Excel file next to these scripts:
#    qry6114_2025f_autograding_final.xlsx

python train_grading_agent.py   # retrains; overwrites rl_grading_agent.zip + grading_meta.json
python evaluate.py              # refreshes results.md (requires the same .xlsx + new meta)
```

**Windows:** `evaluate.py` imports Stable-Baselines3 **before** scikit-learn to avoid a known **DLL load order** issue (`WinError 1114` with PyTorch).

---

## Limitations

- **Small data** (~1.3k rows) — high variance; test set has only **263** examples.  
- **Simplified features** — no rubric sub-scores, no embeddings of student text, no graph of peer similarity.  
- **Discrete coarse actions** — real TA deltas are fractional; +3 vs +4 is not modeled except via the **flag** action.  
- **Single course / term** — no claim of generalization to other domains without retraining.

---

## Future work

- Add **rubric-level features** (if exported) and **dense text embeddings** (e.g. sentence-transformers) for `response_txt` / `ai_feedback`.  
- **Richer action space** (e.g. discrete bins for negative adjustments, finer positive steps) or direct regression with distributional RL.  
- **Calibration** per `questions_id` and per assignment; multi-task or meta-learning across courses.  
- **Offline RL / imitation** (BC / CQL) as a stronger baseline than PPO on static logs.

---

## Repository layout

```
rl_autograding_prototype/
├── README.md
├── requirements.txt
├── grading_env.py
├── train_grading_agent.py
├── evaluate.py
├── results.md
├── rl_grading_agent.zip      # trained PPO (committed)
├── grading_meta.json         # split + norms (committed; must match zip)
└── .gitignore
```

`qry6114_2025f_autograding_final.xlsx` is intentionally **not** tracked.

---

## Publishing to GitHub

This repo was initialized with `git` locally. If `gh` CLI is unavailable, create an empty repository named **`rl-grading-adjustment-prototype`** on GitHub, then:

```bash
cd rl_autograding_prototype
git remote add origin https://github.com/<YOUR_USER>/rl-grading-adjustment-prototype.git
git push -u origin main
```

Use SSH remotes if you prefer (`git@github.com:...`).

---

## License

Prototype / educational use. Add a license file if you redistribute beyond research.
