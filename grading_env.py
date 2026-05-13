"""
GradingAdjustmentEnv — a minimal Gymnasium environment for TA adjustment prediction.

Each episode corresponds to a single submission row. The agent observes engineered
features and selects a discrete adjustment (or a "human review" flag). Reward is
shaped to match how close the choice is to the realized TA adjustment
(TA_Grade - ai_score) on the training distribution.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

NUM_QUESTIONS = 13
OBS_DIM = 1 + NUM_QUESTIONS + 2  # 16


def _word_count(text: Any) -> int:
    """Simple whitespace token count for normalization features."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    s = str(text).strip()
    if not s:
        return 0
    return len(s.split())


def compute_adjustment_reward(action: int, actual_adjustment: float) -> float:
    """
    Reward mapping used for training and evaluation.

    Actions:
        0–3 : predict a numeric adjustment of that many points (added to AI score).
        4   : flag for human review (intended when the TA moves the score by a lot).

    Args:
        action: int in [0, 4]
        actual_adjustment: TA_Grade - ai_score (may be fractional).

    Returns:
        Scalar reward.
    """
    if action == 4:
        if actual_adjustment > 3:
            return 10.0 + 3.0  # strong success + flag bonus
        return -5.0  # unnecessary flag

    pred = float(action)
    err = abs(pred - float(actual_adjustment))
    if err <= 0.5:
        return 10.0
    if err <= 1.0:
        return 5.0
    if err > 2.0:
        return -5.0
    return 0.0  # strictly between 1 and 2 points error


def build_observation(
    row: pd.Series,
    max_resp_words: float,
    max_fb_words: float,
    num_questions: int = NUM_QUESTIONS,
) -> np.ndarray:
    """Build the same observation vector used inside GradingAdjustmentEnv."""
    max_resp = float(max_resp_words) if max_resp_words > 0 else 1.0
    max_fb = float(max_fb_words) if max_fb_words > 0 else 1.0
    dim = 1 + num_questions + 2
    ai = float(pd.to_numeric(row["ai_score"], errors="coerce"))
    qid = int(pd.to_numeric(row["questions_id"], errors="coerce"))
    resp_w = _word_count(row.get("response_txt", ""))
    fb_w = _word_count(row.get("ai_feedback", ""))
    obs = np.zeros(dim, dtype=np.float32)
    obs[0] = np.clip(ai / 10.0, 0.0, 1.0)
    if 1 <= qid <= num_questions:
        obs[1 + (qid - 1)] = 1.0
    obs[1 + num_questions] = min(resp_w / max_resp, 1.0)
    obs[2 + num_questions] = min(fb_w / max_fb, 1.0)
    return obs


class GradingAdjustmentEnv(gym.Env):
    """
    One-step contextual bandit: each reset samples one row; one action ends the episode.

    Observation (float32 vector, length 16):
        [0]       ai_score / 10.0
        [1:14]    one-hot for questions_id in 1..13 (index i = 1 if id == i+1)
        [14]      response word count / max_resp_words (train-derived)
        [15]      ai_feedback word count / max_fb_words (train-derived)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        row_indices: np.ndarray,
        max_resp_words: float,
        max_fb_words: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()
        self._df = df.reset_index(drop=True)
        self._indices = np.asarray(row_indices, dtype=np.int64)
        self._max_resp = float(max_resp_words) if max_resp_words > 0 else 1.0
        self._max_fb = float(max_fb_words) if max_fb_words > 0 else 1.0
        self._rng = rng if rng is not None else np.random.default_rng()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        self._cursor: Optional[int] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        pick = int(self._rng.integers(0, len(self._indices)))
        self._cursor = int(self._indices[pick])
        row = self._df.iloc[self._cursor]
        obs = build_observation(row, self._max_resp, self._max_fb, NUM_QUESTIONS)
        return obs, {}

    def step(self, action: int):
        if self._cursor is None:
            raise RuntimeError("Environment must be reset before step().")

        row = self._df.iloc[self._cursor]
        ai = float(pd.to_numeric(row["ai_score"], errors="coerce"))
        ta = float(pd.to_numeric(row["TA_Grade"], errors="coerce"))
        actual_adj = ta - ai

        reward = float(compute_adjustment_reward(int(action), actual_adj))
        obs = np.zeros(OBS_DIM, dtype=np.float32)  # terminal observation (unused)
        terminated = True
        truncated = False
        info = {
            "actual_adjustment": actual_adj,
            "ai_score": ai,
            "TA_Grade": ta,
            "questions_id": int(pd.to_numeric(row["questions_id"], errors="coerce")),
            "action": int(action),
        }
        self._cursor = None
        return obs, reward, terminated, truncated, info


def load_grading_frame(xlsx_path: str) -> pd.DataFrame:
    """
    Load the course workbook and keep comparable-scale rows.

    Excludes ``questions_id == 9`` (TA rubric on 0–20 there while AI stays 0–10).
    Drops rows with missing scores.
    """
    df = pd.read_excel(xlsx_path)
    q = pd.to_numeric(df["questions_id"], errors="coerce")
    ai = pd.to_numeric(df["ai_score"], errors="coerce")
    ta = pd.to_numeric(df["TA_Grade"], errors="coerce")
    mask = q.notna() & ai.notna() & ta.notna() & (q != 9)
    return df.loc[mask].copy()


def compute_norm_maxima(train_df: pd.DataFrame) -> tuple[float, float]:
    """Max word counts on the training split for normalization (leak-free for test)."""
    resp = train_df["response_txt"].map(_word_count)
    fb = train_df["ai_feedback"].map(_word_count)
    return float(resp.max() or 1.0), float(fb.max() or 1.0)
