# RL grading prototype — evaluation results

**Setup:** PPO (`stable-baselines3`) on `GradingAdjustmentEnv`, 50k timesteps, 
train/test = 80%/20% split (`random_state=42`). 
Filtered rows (excluding `questions_id == 9`): **1317**; **test rows:** **263**. 
Rows: `qry6114_2025f_autograding_final.xlsx`, **`questions_id != 9`**. 
Reward tiers match `grading_env.compute_adjustment_reward`.

## Mean reward on test set

| Method | Mean reward |
|--------|------------:|
| PPO (trained) | 8.1825 |
| Baseline: always trust AI (action 0) | 1.4449 |
| Baseline: always +2 (action 2) | 2.6996 |
| Baseline: OLS regression (`TA−AI` ~ `ai_score` only) | 7.1483 |

## Confusion-style counts (PPO predicted action vs actual adjustment bucket)

Rows: **predicted action** (0–3 = point bump, `flag` = action 4). 
Cols: **bucket of actual** `TA_Grade - ai_score`.

| pred \ actual | <=0 | (0,1] | (1,2] | (2,3] | >3 | **row sum** |
|---|:---|:---|:---|:---|:---|---:|
| **0** | 69 | 11 | 3 | 0 | 0 | 83 |
| **1** | 4 | 12 | 0 | 0 | 0 | 16 |
| **2** | 7 | 10 | 71 | 16 | 3 | 107 |
| **3** | 0 | 0 | 0 | 0 | 0 | 0 |
| **flag** | 1 | 1 | 0 | 1 | 54 | 57 |
| **col sum** | 81 | 34 | 74 | 17 | 57 | 263 |

## Per-question mean reward (PPO on test rows)

| questions_id | n (test) | mean reward |
|-------------:|---------:|------------:|
| 3 | 29 | 5.3448 |
| 4 | 27 | 6.5185 |
| 5 | 4 | 10.0000 |
| 6 | 3 | 8.3333 |
| 7 | 23 | 6.0435 |
| 8 | 41 | 5.4878 |
| 10 | 51 | 10.1569 |
| 11 | 23 | 12.0870 |
| 12 | 27 | 9.9630 |
| 13 | 35 | 9.3429 |

---

*Snapshot from `evaluate.py`; re-run evaluation after training to refresh.*
