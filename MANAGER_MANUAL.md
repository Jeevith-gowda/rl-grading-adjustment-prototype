# Manager’s Manual — RL Grading Adjustment Prototype

*Read this the night before you talk to Professor Cheng. Everything is in plain English.*

---

## 1. What the professor asked for

**The dataset she gave**  
She shared an autograding export: student submissions with an **AI score**, **TA final grade**, question text, student response, AI feedback, and metadata like which question each row belongs to. It’s real course data from her autograding pipeline—not a toy CSV from a textbook.

**Her question (in spirit)**  
Something like: *“How can we practice reinforcement learning on real data so it might eventually help autograding?”* She wasn’t asking for a finished product; she was asking for a **credible direction** that uses **her** data and sounds like modern ML.

**What was ambiguous—and how we interpreted it**  
- She didn’t specify “train a classifier” vs “train an RL agent” vs “fine-tune an LLM.”  
- She didn’t say whether the goal was to **replace** the AI grader, **assist** the TA, or **prioritize** human review.

**Our interpretation:**  
Treat the TA as the “ground truth policy” for how scores should be **adjusted** after the AI runs. Build a small RL prototype that **predicts the TA’s adjustment** (how many points to add, or whether to flag for a human) from cheap features. That’s easy to explain in a meeting, uses RL honestly enough for a first pass, and doesn’t pretend we solved grading.

---

## 2. How we framed the problem

**Why “predict TA adjustment” instead of “replace the AI grader”?**  
- Replacing the AI would mean building a whole new scoring model and defending it academically—that’s a thesis, not a weekend prototype.  
- **Adjustment prediction** only asks: “Given what the AI already said, what would the TA do next?” That’s smaller, measurable, and grounded in **pairs** of (AI score, TA score) she already has.

**Why that maps to RL**  
We defined **states** (features about a submission), **actions** (discrete adjustments + flag), and **rewards** (how close we were to the TA’s real adjustment). The agent improves a **policy**—a rule that picks actions from states—by trial and error over many examples. That’s the RL story.

We did **not** claim the TA is an “environment” simulating the world; the environment is a **programmed stand-in** that shows one submission at a time and scores your guess against her historical decision. That’s enough for a prototype narrative.

---

## 3. What RL actually does here (plain language)

**Step by step**  
1. The system shows the agent **one submission** (really: a vector of numbers summarizing that row).  
2. The agent picks **one of five actions**: trust the AI (0), add 1, 2, or 3 points, or **flag** for human review.  
3. We compare that choice to what **actually happened**: TA grade minus AI score on that row.  
4. We assign a **reward**: big reward if we were very close, medium if close enough, penalty if we were way off, extra if we correctly flagged a “big” TA change.  
5. Over **thousands of training steps**, PPO nudges the policy toward choices that earn more reward on average.

**Why call it RL and not “just classification”?**  
- A classifier would usually output a class and train with cross-entropy. Here we framed **actions** and **scalar rewards** with asymmetric tiers (great / ok / bad / flag bonus). That’s **reward-shaped policy learning**, which is RL-flavored.  
- **Important honesty:** each “episode” is **one step** (see → act → reward → done). That’s very close to a **contextual bandit** or even supervised learning with a weird loss. We still used an off-the-shelf **PPO** trainer because it’s robust and quick; we’re not claiming sample-efficient deep RL on a sparse MDP.

**Crucial point for Cheng:** The value is the **pipeline story** (data → env → policy → evaluation), not that PPO is the only algorithm that could ever work here.

---

## 4. Walk-through of the code

### `grading_env.py` — the environment (high level)

- **`load_grading_frame`:** Reads the Excel file, drops rows with missing scores, and **removes question 9** (AI 0–10 vs TA 0–20 on that item—apples to oranges).  
- **`_word_count`:** Counts words in text fields for simple “how long was the answer / feedback” features.  
- **`build_observation`:** Builds a fixed-length vector: normalized AI score, **one-hot** for question ID (1–13), normalized response length, normalized feedback length. Max word counts come from the **training split only** so we don’t leak test info into scaling.  
- **`GradingAdjustmentEnv`:** A Gymnasium env. **`reset`** picks a random **training** row and returns its observation. **`step`** takes the agent’s action, computes `TA_Grade - AI_score`, runs **`compute_adjustment_reward`**, returns reward and `terminated=True` (one step per episode).  
- **`compute_adjustment_reward`:** Encodes the tier rules (+10 / +5 / 0 / −5) and the **flag** path (+13 when TA moved more than 3 points and we flagged).

### `train_grading_agent.py` — training pipeline

- Loads the filtered dataframe, **shuffles** indices with a fixed seed, splits **80% train / 20% test**.  
- Computes normalization maxima from **train** rows only.  
- Wraps the env in Stable-Baselines3’s **`DummyVecEnv`** (one parallel env).  
- Builds **PPO** with an MLP policy, trains for **50,000** timesteps, saves **`rl_grading_agent.zip`**.  
- Writes **`grading_meta.json`**: train/test index lists and the word-count caps so **`evaluate.py`** can reproduce the same split and features without retraining.

### `evaluate.py` — baselines and comparison

- Reloads data + meta, rebuilds train/test slices **the same way**.  
- **Baselines:**  
  - Always **trust AI** (action 0).  
  - Always **+2** (action 2)—a nod to “AI is harsh.”  
  - **Linear regression** predicting `TA - AI` from **AI score alone**; reward uses the same distance tiers as numeric actions (no flag semantics for regression).  
- Loads the **saved PPO**, runs **deterministic** predictions on each **test** row, computes the same reward function for everyone.  
- Writes **`results.md`**: mean rewards, confusion-style counts (predicted discrete action vs bucketed actual adjustment), per-question mean PPO reward on test.

### Why those reward tiers?

They’re a **teaching-friendly** compromise:  
- Big bonus for being **very** close to the TA’s adjustment (within 0.5).  
- Smaller bonus within 1.0.  
- Penalty if we’re **more than 2** points off—forces the policy not to ignore large errors.  
- **Flag** action gets a jackpot when the TA moved **more than 3** points—so the model has a way to say “this needs a human” without pretending a +3 cap fits every case.

---

## 5. The results — what they mean

**Why 8.18 vs 7.15 (regression) matters**  
Same reward function for everyone. PPO uses **more features** than “AI score only,” and a **nonlinear policy** (MLP). Beating linear regression means the extra signal (question ID, lengths) and the policy class **carry real information**—not that PPO is magic on 1,317 rows.

**What the confusion table tells us**  
Rows are **what the agent predicted**; columns are **buckets of the true TA adjustment**. Big mass on **“+2”** predictions when actual adjustments fall in the **(1,2]** bucket shows the agent learned “often bump by about two.” The **flag** row lining up with the **>3** column shows it learned when human review is appropriate for large TA swings.

**Why per-question performance varies**  
Some questions have **narrower** TA adjustment patterns (easier to guess); others are open-ended and noisy (harder). Lower mean reward on a question can mean **harder prediction**, not that the TA was “wrong.”

**Honest strength / weakness**  
- **Strong:** Clear story, reproducible code, beats sensible baselines on the same metric, flagging large disagreements.  
- **Weak:** Small *n*, coarse actions, no text semantics, one term of one course—**do not** over-claim generalization.

---

## 6. Things Professor Cheng might ask (cheat sheet)

| Question | One-line answer |
|----------|-----------------|
| Why **PPO** instead of **DQN**? | Actions are discrete but tiny (5); PPO is stable and default in SB3; DQN shines more on larger discrete spaces and needs more tuning here. |
| Why **this action space**? | Matches how we talk about TA fixes (+1/+2) and a catch-all **flag** for huge moves; keeps the story simple. |
| Why **exclude question 9**? | TA scores there go up to **20** while AI stays **0–10**—same number doesn’t mean same scale. |
| Why **these features** and not embeddings? | Prototype speed + interpretability; embeddings are listed as **next step**. |
| Is this **really RL** or classification? | It’s **policy optimization with rewards**; episodes are one-step so it’s close to a **bandit**—we’re not hiding that. |
| How does this **scale** to more questions / courses? | Retrain per course or multi-task with question/course IDs; need more data and probably richer features. |
| **Next step** with a week? | Wire in **sentence embeddings** + rubric fields if available; add **calibration** plots per question. |
| Why no **graph neural network**? | No graph was built over students/questions in this slice—would need a definition of nodes/edges first. |
| How long does **training** take? | On CPU, on the order of **~1–2 minutes** for 50k steps with this tiny env. |
| What **breaks** if data triples? | Probably nothing technically—training gets slower; you’d want **validation**, maybe **early stopping**, and to revisit normalization. |

---

## 7. What to say if she asks about Decima

**Reminder:** Decima was a research / narrative anchor around autograding agents (the “autograding lab” direction you explored earlier).

**Why this pivoted to her data:**  
Decima-style demos are great for imagination, but Cheng’s question was grounded in **her export** and **her classroom**. This prototype is deliberately **small, concrete, and tied to her spreadsheet** so the conversation stays about **what her TA actually did** on real submissions—not a fictional stack.

---

*End of manual. Good luck tomorrow.*
