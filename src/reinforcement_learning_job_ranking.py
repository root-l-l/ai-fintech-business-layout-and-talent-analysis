"""Run a clearly labelled epsilon-greedy bandit demonstration for job ranking.

The candidates are real cleaned job records.  No behavioural click, application
or offer data are available, so the reward is a deterministic researcher-defined
skill-fit score for a hypothetical financial-data-and-AI learner.  This is a
reproducible reinforcement-learning demonstration, not a real user-personalized
recommendation experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_directories, repo_path, write_json


TARGET_PROFILE_NAME = "金融数据与AI应用学习者（模拟画像）"
TARGET_SKILLS = ("Python", "SQL", "金融", "风控", "数据分析", "统计", "机器学习", "大模型")
REWARD_DEFINITION = "每个候选岗位在职位名称+技能要求中命中的模拟画像技能数 / 8；不使用真实点击、投递或录用反馈。"


def skill_matches(text: str) -> list[str]:
    lowered = text.casefold()
    return [skill for skill in TARGET_SKILLS if skill.casefold() in lowered]


def train_bandit(rewards: np.ndarray, episodes: int, seed: int, alpha: float, epsilon_start: float, epsilon_end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.zeros(len(rewards), dtype=float)
    counts = np.zeros(len(rewards), dtype=int)
    history = np.zeros(episodes, dtype=float)
    for step in range(episodes):
        fraction = step / max(episodes - 1, 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * fraction
        if rng.random() < epsilon:
            action = int(rng.integers(len(rewards)))
        else:
            best = np.flatnonzero(values == values.max())
            action = int(rng.choice(best))
        reward = rewards[action]
        values[action] += alpha * (reward - values[action])
        counts[action] += 1
        history[step] = reward
    return values, counts, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an epsilon-greedy bandit demonstration for observed job records.")
    parser.add_argument("--input", type=Path, default=repo_path("data", "processed", "jobs_clean.csv"))
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=repo_path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Required recruitment input is missing: {args.input}")
    if args.episodes < 1 or not 0 < args.alpha <= 1:
        raise ValueError("episodes must be positive and alpha must be in (0, 1]")
    jobs = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    required = {"job_id", "company_name", "company_code", "job_title", "skills_raw", "source_sheet", "source_row"}
    missing = required - set(jobs.columns)
    if missing:
        raise ValueError(f"jobs input missing columns: {sorted(missing)}")
    ranking = jobs.loc[:, ["job_id", "company_name", "company_code", "job_title", "skills_raw", "source_sheet", "source_row"]].copy()
    text = (ranking["job_title"].str.strip() + " " + ranking["skills_raw"].str.strip()).str.strip()
    ranking["matched_profile_skills"] = text.map(skill_matches).map("、".join)
    ranking["simulated_skill_fit_reward"] = ranking["matched_profile_skills"].map(lambda value: 0 if not value else len(value.split("、")) / len(TARGET_SKILLS))
    rewards = ranking["simulated_skill_fit_reward"].to_numpy(dtype=float)
    values, counts, history = train_bandit(rewards, args.episodes, args.seed, args.alpha, args.epsilon_start, args.epsilon_end)
    ranking["bandit_q_value"] = values.round(6)
    ranking["selection_count"] = counts
    ranking["target_profile"] = TARGET_PROFILE_NAME
    ranking["reward_definition"] = REWARD_DEFINITION
    ranking["method_boundary"] = "epsilon-greedy 多臂老虎机模拟；奖励是确定性技能匹配，非真实用户反馈。"
    ranking = ranking.sort_values(["bandit_q_value", "simulated_skill_fit_reward", "job_id"], ascending=[False, False, True]).reset_index(drop=True)
    ranking.insert(0, "recommendation_rank", np.arange(1, len(ranking) + 1))
    rolling = pd.Series(history).rolling(200, min_periods=1).mean()
    curve = pd.DataFrame({"episode": np.arange(1, args.episodes + 1), "simulated_reward": history, "rolling_mean_reward_200": rolling})
    summary = pd.DataFrame([{
        "candidate_job_records": len(ranking), "episodes": args.episodes, "seed": args.seed, "alpha": args.alpha,
        "epsilon_start": args.epsilon_start, "epsilon_end": args.epsilon_end,
        "random_policy_expected_reward": round(float(rewards.mean()), 4),
        "last_500_episode_mean_reward": round(float(history[-min(500, len(history)):].mean()), 4),
        "best_candidate_simulated_reward": round(float(rewards.max()), 4),
        "target_profile": TARGET_PROFILE_NAME, "reward_definition": REWARD_DEFINITION,
        "method_boundary": "没有真实用户行为数据；本表仅展示强化学习方法如何在明确的模拟反馈下学习排序策略。",
    }])
    tables, figures = args.output_dir / "tables", args.output_dir / "figures"
    ensure_directories([tables, figures])
    ranking.drop(columns=["skills_raw"]).to_csv(tables / "rl_job_recommendation_ranking.csv", index=False, encoding="utf-8")
    curve.to_csv(tables / "rl_job_recommendation_learning_curve.csv", index=False, encoding="utf-8")
    summary.to_csv(tables / "rl_job_recommendation_summary.csv", index=False, encoding="utf-8")
    write_json(tables / "rl_job_recommendation_config.json", {"target_profile": TARGET_PROFILE_NAME, "target_skills": TARGET_SKILLS, "reward_definition": REWARD_DEFINITION, **summary.iloc[0].to_dict()})
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(curve["episode"], curve["rolling_mean_reward_200"], color="#2B7A78", linewidth=2, label="200轮滚动平均奖励")
    axis.axhline(rewards.mean(), color="#C84C3A", linestyle="--", label="随机策略期望奖励")
    axis.set(title="Epsilon-greedy 岗位排序模拟的学习曲线", xlabel="交互轮次（模拟）", ylabel="技能匹配奖励（0-1）", ylim=(0, 1))
    axis.legend(frameon=False)
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    figure.savefig(figures / "rl_job_recommendation_learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Ranked {len(ranking)} observed job record(s) in a {args.episodes}-episode simulated bandit run.")
    print(f"Wrote ranking and learning curve to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
