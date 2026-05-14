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

## Files

- `grading_env.py` — the RL environment
- `train_grading_agent.py` — training script
- `evaluate.py` — comparison vs baselines
- `results.md` — full results
- `rl_grading_agent.zip` — the trained model

## How to run

```bash
pip install -r requirements.txt
# add qry6114_2025f_autograding_final.xlsx to the folder
python train_grading_agent.py
python evaluate.py
```

## What's missing

This is a prototype, not production:
- Small dataset (1,317 examples)
- Simple features (no rubric details, no response embeddings)
- Coarse actions (no fractional adjustments)
- Single course

Next steps would be adding richer features (rubric scores, text embeddings) and finer-grained actions.
