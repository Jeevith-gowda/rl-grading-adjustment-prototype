"""
Exploratory data analysis for qry6114_2025f_autograding_final.xlsx.

Outputs (under analysis/):
  - 6 PNG visualizations
  - data_analysis_report.md
  - sample_submissions.md
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from grading_env import _word_count, load_grading_frame

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"
XLSX = DATA_DIR / "qry6114_2025f_autograding_final.xlsx"

# Light theme, readable fonts
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)

QUESTION_SHORT = {
    3: "Q3 Order notation",
    4: "Q4 Non-recursive analysis",
    5: "Q5 Substitution",
    6: "Q6 Master theorem",
    7: "Q7 Quicksort",
    8: "Q8 Heap",
    10: "Q10 0-1 Knapsack",
    11: "Q11 Fractional Knapsack",
    12: "Q12 BFS/DFS",
    13: "Q13 MST / shortest paths",
}


def _trunc(text: str, max_len: int = 500) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _assignment_title(row: pd.Series) -> str:
    qt = str(row.get("question_txt", "") or "").strip()
    if not qt:
        return f"Question {int(row['questions_id'])}"
    first = qt.split("\n")[0].strip()
    return _trunc(first, 120)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(XLSX)
    filtered = load_grading_frame(str(XLSX)).reset_index(drop=True)
    filtered["score_diff"] = filtered["ai_score"] - filtered["TA_Grade"]
    filtered["ta_adjustment"] = filtered["TA_Grade"] - filtered["ai_score"]
    filtered["response_words"] = filtered["response_txt"].map(_word_count)
    filtered["feedback_words"] = filtered["ai_feedback"].map(_word_count)
    return raw, filtered


def pct_underscored(df: pd.DataFrame) -> float:
    return 100.0 * (df["score_diff"] < 0).mean()


def plot_01_scatter(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    q_ids = sorted(df["questions_id"].unique())
    palette = sns.color_palette("tab20", n_colors=len(q_ids))
    for qid, color in zip(q_ids, palette):
        sub = df[df["questions_id"] == qid]
        ax.scatter(
            sub["ai_score"],
            sub["TA_Grade"],
            label=f"Q{int(qid)}",
            alpha=0.65,
            s=36,
            color=color,
            edgecolors="white",
            linewidths=0.3,
        )
    lim_lo, lim_hi = 0, 10.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="#888888", lw=2, label="y = x")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("AI score")
    ax.set_ylabel("TA grade")
    ax.set_title("AI score vs TA grade — points below diagonal = AI underscored")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, title="Question")
    note = "Most points lie below the diagonal (TA grade > AI score)."
    fig.text(0.5, 0.02, note, ha="center", fontsize=12, color="#444444", style="italic")
    fig.tight_layout(rect=[0, 0.04, 0.82, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_02_histogram(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    diffs = df["score_diff"]
    bins = np.arange(-10, 7, 1)
    counts, _, patches = ax.hist(
        diffs, bins=bins, color="#3b82f6", edgecolor="white", alpha=0.85
    )
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=2, label="Exact agreement")

    ymax = float(max(counts)) if len(counts) else 1.0
    ax.set_ylim(0, ymax * 1.22)

    for patch, count in zip(patches, counts):
        height = float(count)
        if height < 5:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        ax.text(
            x,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#333333",
            fontweight="bold",
        )

    n_total = len(diffs)
    n_under = int((diffs < 0).sum())
    n_at_or_above = int((diffs >= 0).sum())
    pct_under = 100.0 * n_under / n_total if n_total else 0.0
    pct_above = 100.0 - pct_under
    label_y = ymax * 1.08

    ax.annotate(
        f"{pct_under:.0f}% (n={n_under}) underscored",
        xy=(-5.0, label_y * 0.55),
        xytext=(-7.5, label_y),
        fontsize=11,
        color="#333333",
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="#4338ca", lw=1.2),
    )
    ax.annotate(
        f"{pct_above:.0f}% (n={n_at_or_above}) at or above",
        xy=(2.5, label_y * 0.55),
        xytext=(4.5, label_y),
        fontsize=11,
        color="#333333",
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="#4338ca", lw=1.2),
    )

    ax.set_xlabel("AI score − TA grade")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of AI − TA differences (n={n_total})")
    ax.legend(loc="upper right")
    caption = (
        f"{pct_under:.1f}% of submissions show AI underscoring (TA grade > AI score); "
        f"{pct_above:.1f}% at or above the diagonal."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=12, color="#444444", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_03_boxplot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = df.copy()
    plot_df["question"] = plot_df["questions_id"].astype(int).astype(str)
    order = [str(q) for q in sorted(plot_df["questions_id"].unique())]
    sns.boxplot(
        data=plot_df,
        x="question",
        y="score_diff",
        hue="question",
        order=order,
        palette="Blues",
        legend=False,
        ax=ax,
        fliersize=3,
    )
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Question ID")
    ax.set_ylabel("AI score − TA grade")
    ax.set_title("AI−TA difference by question")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_04_bar_counts(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    counts = df.groupby("questions_id").size().sort_index()
    bars = ax.bar(counts.index.astype(int), counts.values, color="#0d9488", edgecolor="white")
    ax.bar_label(bars, fontsize=11)
    ax.set_xlabel("Question ID")
    ax.set_ylabel("Number of submissions")
    ax.set_title("Submission volume per question")
    ax.set_xticks(counts.index.astype(int))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_05_response_length(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    words = df["response_words"]
    ax.hist(words, bins=40, color="#6366f1", edgecolor="white", alpha=0.85)
    med = float(words.median())
    mean = float(words.mean())
    ax.axvline(med, color="#dc2626", linestyle="--", linewidth=2, label=f"Median = {med:.0f}")
    ax.axvline(mean, color="#f59e0b", linestyle="-.", linewidth=2, label=f"Mean = {mean:.0f}")
    if words.max() > words.min() * 10:
        ax.set_yscale("log")
    ax.set_xlabel("Response length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of student response lengths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_06_score_distributions(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        df["ai_score"],
        bins=np.arange(-0.5, 11.5, 1),
        alpha=0.6,
        color="#2563eb",
        label="AI score",
        edgecolor="white",
    )
    ax.hist(
        df["TA_Grade"],
        bins=np.arange(-0.5, 11.5, 1),
        alpha=0.6,
        color="#ea580c",
        label="TA grade",
        edgecolor="white",
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("AI score vs TA grade distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pick_samples(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Return (category_label, row) for 10 strategic samples."""
    used: set[int] = set()
    picks: list[tuple[str, pd.Series]] = []

    def take(pool: pd.DataFrame, label: str, n: int = 1) -> None:
        for _, row in pool.iterrows():
            if len(picks) >= 10:
                return
            sid = int(row["submission_id"])
            if sid in used:
                continue
            used.add(sid)
            picks.append((label, row))
            if sum(1 for p in picks if p[0] == label) >= n:
                return

    exact = df[df["score_diff"] == 0]
    take(exact, "Exact agreement", 2)

    mid_under = df[(df["ta_adjustment"] >= 2) & (df["ta_adjustment"] <= 3)]
    take(mid_under.sort_values("ta_adjustment"), "AI underscored by 2–3 points", 3)

    extreme = df[df["ta_adjustment"] >= 5]
    take(extreme.sort_values("ta_adjustment", ascending=False), "AI underscored by 5+ points", 2)

    over = df[df["score_diff"] > 0].sort_values("score_diff", ascending=False)
    take(over, "AI overscored (TA lowered grade)", 1)

    per_q = df.groupby("questions_id")["ta_adjustment"].mean()
    easy_q = int(per_q.idxmin())
    hard_q = int(per_q.idxmax())
    take(df[df["questions_id"] == easy_q].head(20), f"Easier question (Q{easy_q})", 1)
    take(
        df[df["questions_id"] == hard_q].sort_values("ta_adjustment", ascending=False).head(20),
        f"Harder question (Q{hard_q})",
        1,
    )

    # Fill to 10 if needed
    if len(picks) < 10:
        for _, row in df.iterrows():
            sid = int(row["submission_id"])
            if sid not in used:
                used.add(sid)
                picks.append(("Additional example", row))
            if len(picks) >= 10:
                break

    return picks[:10]


def _interpret_sample(row: pd.Series) -> str:
    diff = float(row["score_diff"])
    adj = float(row["ta_adjustment"])
    qid = int(row["questions_id"])
    qname = QUESTION_SHORT.get(qid, f"Question {qid}")

    if diff == 0:
        return "AI and TA agreed on the score — no adjustment needed."
    if diff > 0:
        return (
            f"AI overgraded by {diff:.1f} points — TA lowered the score; "
            f"may reflect rubric nuance on {qname}."
        )
    if adj >= 5:
        return (
            f"AI undergraded by {adj:.1f} points — large gap on {qname}, "
            "often seen on algorithm-heavy items where the TA restores partial credit."
        )
    if adj >= 2:
        return (
            f"AI undergraded by {adj:.1f} points — typical pattern on {qname} "
            "where the TA bumps scores for reasonable but imperfect answers."
        )
    return (
        f"AI undergraded by {adj:.1f} points on {qname} — "
        "TA applied a modest upward adjustment."
    )


def write_sample_submissions(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Sample submissions",
        "",
        "Ten representative rows from the filtered dataset (question 9 excluded). ",
        "Use these to sanity-check how AI scores, TA grades, and feedback align in practice.",
        "",
    ]
    for i, (label, row) in enumerate(_pick_samples(df), start=1):
        adj = float(row["ta_adjustment"])
        lines.extend(
            [
                f"## Sample {i}: {label}",
                "",
                f"- **submission_id:** {int(row['submission_id'])}",
                f"- **questions_id:** {int(row['questions_id'])}",
                f"- **Assignment:** {_assignment_title(row)}",
                f"- **ai_score:** {float(row['ai_score']):.1f}",
                f"- **TA_Grade:** {float(row['TA_Grade']):.1f}",
                f"- **Adjustment (TA − AI):** {adj:+.1f}",
                "",
                "### Student response",
                "",
                "```",
                _trunc(str(row.get("response_txt", "")), 500),
                "```",
                "",
                "### AI feedback",
                "",
                "```",
                _trunc(str(row.get("ai_feedback", "")), 500),
                "```",
                "",
                f"**Interpretation:** {_interpret_sample(row)}",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def _stats_block(series: pd.Series) -> str:
    return (
        f"| Mean | {series.mean():.3f} |\n"
        f"| Median | {series.median():.3f} |\n"
        f"| Std dev | {series.std():.3f} |\n"
        f"| Min | {series.min():.3f} |\n"
        f"| Max | {series.max():.3f} |"
    )


def write_report(raw: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    n_raw = len(raw)
    n = len(df)
    pct_under = pct_underscored(df)
    pct_any_adj = 100.0 * (df["ta_adjustment"] != 0).mean()
    mean_diff = df["score_diff"].mean()

    per_q = (
        df.groupby("questions_id")
        .agg(
            n=("submission_id", "count"),
            mean_ai=("ai_score", "mean"),
            mean_ta=("TA_Grade", "mean"),
            mean_diff=("score_diff", "mean"),
        )
        .round(3)
        .sort_index()
    )
    table_lines = [
        "| Question | n | Mean AI | Mean TA | Mean (AI−TA) |",
        "|---------:|--:|--------:|--------:|-------------:|",
    ]
    for qid, row in per_q.iterrows():
        table_lines.append(
            f"| {int(qid)} | {int(row['n'])} | {row['mean_ai']:.2f} | "
            f"{row['mean_ta']:.2f} | {row['mean_diff']:.2f} |"
        )

    biggest_gap = per_q["mean_diff"].idxmin()
    smallest_gap = per_q["mean_diff"].idxmax()
    sub_per_student = df.groupby("submitter_id").size()

    lines = [
        "# Data Analysis Report",
        "",
        "## 1. Dataset overview",
        "",
        "- **Source file:** `data/qry6114_2025f_autograding_final.xlsx`",
        "- **Course context:** ITCS 6114 autograding export (Professor Cheng), peer + AI grading with TA final grades",
        f"- **Time range:** {raw['timestamp'].min()} to {raw['timestamp'].max()}",
        f"- **Raw rows:** {n_raw} submissions × {len(raw.columns)} columns",
        "",
        "**Columns (raw):**",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| `submission_id` | Unique submission identifier |",
        "| `submitter_id` | Student identifier |",
        "| `questions_id` | Question / rubric item ID |",
        "| `question_txt` | Question prompt text |",
        "| `response_txt` | Student answer text |",
        "| `timestamp` | Submission time |",
        "| `ai_score` | Automated AI score (typically 0–10) |",
        "| `peer_score` | Peer review score (often missing) |",
        "| `ai_feedback` | AI-generated feedback |",
        "| `peer_feedback` | Peer feedback (often missing) |",
        "| `TA_Grade` | TA final grade after review |",
        "",
        f"**Filter applied:** {n} rows retained after dropping missing scores and **excluding `questions_id == 9`**. "
        "Question 9 uses a 0–20 TA scale while the AI stays on 0–10, so including it would mix incompatible scales.",
        "",
        "## 2. Score distributions",
        "",
        "**AI score**",
        "",
        _stats_block(df["ai_score"]),
        "",
        "**TA grade**",
        "",
        _stats_block(df["TA_Grade"]),
        "",
        "![AI vs TA score distributions](06_ta_score_distribution.png)",
        "",
        "## 3. The central finding: AI is systematically harsh",
        "",
        f"- **Mean (AI − TA):** {mean_diff:.3f} — negative values mean the TA typically grades *higher* than the AI.",
        f"- **Submissions where TA > AI (AI underscored):** {pct_under:.1f}%",
        f"- **Submissions with any TA adjustment (TA ≠ AI):** {pct_any_adj:.1f}%",
        "",
        "![AI vs TA scatter](01_ai_vs_ta_scatter.png)",
        "",
        "![AI − TA difference histogram](02_diff_histogram.png)",
        "",
        "## 4. Per-question breakdown",
        "",
        *table_lines,
        "",
        f"- **Largest average AI harshness (most negative mean diff):** question **{int(biggest_gap)}** "
        f"({QUESTION_SHORT.get(int(biggest_gap), '')}) at {per_q.loc[biggest_gap, 'mean_diff']:.2f}.",
        f"- **Closest average agreement:** question **{int(smallest_gap)}** "
        f"({QUESTION_SHORT.get(int(smallest_gap), '')}) at {per_q.loc[smallest_gap, 'mean_diff']:.2f}.",
        "- Algorithm-heavy items (Knapsack, BFS/DFS, MST/shortest paths) tend to show larger upward TA adjustments.",
        "",
        "![AI−TA difference by question](03_per_question_boxplot.png)",
        "",
        "## 5. Data volume and balance",
        "",
        f"- **Questions represented:** {df['questions_id'].nunique()} (IDs 1–13 except 9)",
        f"- **Unique students (`submitter_id`):** {df['submitter_id'].nunique()}",
        f"- **Submissions per student:** mean {sub_per_student.mean():.1f}, "
        f"median {sub_per_student.median():.0f}, max {sub_per_student.max()}",
        "- Volume is **uneven across questions** — some items dominate the dataset.",
        "",
        "![Submission volume per question](04_submissions_per_question.png)",
        "",
        "## 6. Response characteristics",
        "",
        f"- **Response length (words):** mean {df['response_words'].mean():.0f}, "
        f"median {df['response_words'].median():.0f}, max {df['response_words'].max()}",
        f"- **AI feedback length (words):** mean {df['feedback_words'].mean():.0f}, "
        f"median {df['feedback_words'].median():.0f}, max {df['feedback_words'].max()}",
        "",
        "![Response length distribution](05_response_length_distribution.png)",
        "",
        "## 7. Key insights for modeling",
        "",
        "- **The systematic AI bias is large enough to be learnable** — most submissions need a nonnegative TA bump, especially on harder algorithm questions.",
        "- **Per-question variance is substantial** — `questions_id` (one-hot) should remain a core feature; a single global adjustment rule will miss item-specific behavior.",
        "- **Response and feedback lengths vary by orders of magnitude** — normalize word counts (as in `grading_env.py`) rather than using raw lengths.",
        "",
        "---",
        "",
        "*Auto-generated by `src/data_analysis.py`*",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    if not XLSX.is_file():
        raise FileNotFoundError(f"Missing {XLSX}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    raw, df = load_datasets()

    plots = [
        ("01_ai_vs_ta_scatter.png", plot_01_scatter),
        ("02_diff_histogram.png", plot_02_histogram),
        ("03_per_question_boxplot.png", plot_03_boxplot),
        ("04_submissions_per_question.png", plot_04_bar_counts),
        ("05_response_length_distribution.png", plot_05_response_length),
        ("06_ta_score_distribution.png", plot_06_score_distributions),
    ]
    for name, fn in plots:
        out = ANALYSIS_DIR / name
        fn(df, out)
        print(f"Wrote {out}")

    write_report(raw, df, ANALYSIS_DIR / "data_analysis_report.md")
    write_sample_submissions(df, ANALYSIS_DIR / "sample_submissions.md")
    print("Done.")


if __name__ == "__main__":
    main()
