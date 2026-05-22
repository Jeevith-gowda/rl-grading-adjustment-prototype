# RL Grading Adjustment Prototype

A small experiment: can reinforcement learning learn to predict how a TA would adjust an AI-graded student submission?

Built on Professor Cheng's autograding dataset.

## The idea

The AI grader gives scores to student answers. The TA reviews those and adjusts them. Goal: train an RL agent that predicts the TA's adjustment, so we can flag only the cases that actually need human review.

## The data

- 1,317 student submissions, each graded by both the AI and the TA
- One question (question 9) was excluded because it uses a different scoring scale

## What the agent does

For each submission, the agent picks one of 5 actions:
- Trust the AI (no change)
- Add +1 point
- Add +2 points
- Add +3 points
- Flag for human review

It learns from how the TA actually adjusted similar submissions in the past.

## Results

Tested on 263 held-out submissions:

| Method | Mean reward |
|--------|------:|
| PPO agent (this prototype) | **8.18** |
| Regression baseline | 7.15 |
| Always +2 | 2.70 |
| Always trust AI | 1.44 |

The agent gets the right adjustment bucket ~75% of the time and correctly flags the high-disagreement cases 95% of the time.

## What I noticed

The AI is systematically harsh on harder algorithmic questions — Knapsack, BFS/DFS, MST/Shortest Paths. The TA consistently adds 2-3 points back. The agent picked up on this pattern.

## Repository layout

```
rl-grading-adjustment-prototype/
├── README.md
├── MANAGER_MANUAL.md
├── requirements.txt
├── .gitignore
│
├── src/                          ← all Python source files
│   ├── grading_env.py
│   ├── train_ppo_agent.py
│   ├── train_a2c_agent.py
│   ├── train_random_forest.py
│   └── evaluate.py
│
├── models/                       ← all trained model artifacts
│   ├── ppo_grading_agent.zip
│   ├── a2c_grading_agent.zip
│   ├── random_forest_model.pkl
│   └── grading_meta.json
│
├── results/                      ← all evaluation outputs
│   └── results.md
│
├── presentation/                 ← website
│   └── index.html
│
└── data/                         ← data files (gitignored)
    └── (qry6114_2025f_autograding_final.xlsx lives here but not committed)
```

## How to run

```bash
cd rl-grading-adjustment-prototype
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Place qry6114_2025f_autograding_final.xlsx in data/
python src/train_ppo_agent.py
python src/train_a2c_agent.py
python src/train_random_forest.py
python src/evaluate.py
```

## What's missing

This is a prototype, not production:
- Small dataset (1,317 examples)
- Simple features (no rubric details, no response embeddings)
- Coarse actions (no fractional adjustments)
- Single course

Next steps would be adding richer features (rubric scores, text embeddings) and finer-grained actions.
