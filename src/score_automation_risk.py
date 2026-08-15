"""Assess AI task-automation potential from observed recruitment text.

The resulting score is a transparent, researcher-defined proxy for task
automation potential.  It is not an observed probability that a person or a
whole occupation will be replaced.  Each posting is scored from its recorded
job title and skills text only, and every lexical hit is retained for review.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_directories, repo_path


@dataclass(frozen=True)
class Dimension:
    """One explicitly specified component of the seven-dimension rubric."""

    key: str
    label: str
    raises_risk: bool
    terms: tuple[str, ...]


# The script applies one fixed dictionary to every posting. It operationalizes
# task cues appearing in recruitment text; absence of a word is not evidence
# that a task is absent from the actual job.
DIMENSIONS = (
    Dimension(
        "repeatability",
        "任务重复性",
        True,
        ("录入", "数据处理", "数据清洗", "对账", "清算", "核算", "测试", "测试用例", "脚本", "监控", "报表", "文档", "批处理", "维护", "部署", "发布", "运维", "审核", "验收"),
    ),
    Dimension(
        "rule_clarity",
        "规则明确性",
        True,
        ("规则", "流程", "规范", "标准", "测试", "测试用例", "核算", "清算", "对账", "审批", "审查", "报送", "报表", "SQL", "脚本", "数据治理", "指标"),
    ),
    Dimension(
        "data_structuredness",
        "数据结构化程度",
        True,
        ("SQL", "数据库", "数据仓库", "数据治理", "数据清洗", "数据处理", "ETL", "报表", "指标", "交易", "清算", "核算", "表格", "Excel", "结构化", "统计", "数据建模", "数据平台"),
    ),
    Dimension(
        "complex_communication",
        "复杂沟通需求",
        False,
        ("沟通", "协调", "客户", "商务", "谈判", "咨询", "访谈", "演讲", "宣讲", "售前", "销售", "需求调研", "产品演示", "跨部门", "跨团队", "团队协作", "对接", "驻场", "交付"),
    ),
    Dimension(
        "responsibility_judgment",
        "责任判断需求",
        False,
        ("负责人", "主管", "总监", "架构", "方案", "设计", "规划", "策略", "创新", "研究", "算法", "模型验证", "复杂问题", "问题解决", "业务理解", "需求分析", "用户体验", "风险管理", "领导", "管理", "决策", "评估"),
    ),
    Dimension(
        "compliance_trust",
        "合规或信任责任",
        False,
        ("合规", "监管", "审计", "消保", "反洗钱", "征信", "信贷审批", "授信", "贷后", "客户信任", "隐私", "数据安全", "保密", "投顾", "投资建议", "财务核算", "支付", "证券交易"),
    ),
    Dimension(
        "cross_function_coordination",
        "跨部门协调需求",
        False,
        ("跨部门", "跨团队", "团队协作", "协同", "对接", "项目管理", "项目推进", "客户", "驻场", "交付", "协调", "沟通", "需求调研", "访谈", "商务", "销售", "产品"),
    ),
)

RUBRIC = (
    "七维等权研究设计：任务重复性、规则明确性、数据结构化程度为正向项；"
    "复杂沟通、责任判断、合规或信任责任、跨部门协调为反向项。每项按命中词数映射为1/3/4/5分"
    "（0/1/2-3/4个及以上）；替代风险评分=100*[三项正向得分之和+四项(6-反向得分)之和]/35。"
    "评分仅反映本研究词典与岗位文本下的任务自动化潜力，不是实际替代概率。"
)


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def matched_terms(text: str, terms: Iterable[str]) -> list[str]:
    """Return distinct pre-specified terms found in the observed text."""

    lowered = text.casefold()
    return [term for term in terms if term.casefold() in lowered]


def ordinal_score(hit_count: int) -> int:
    """Map a lexical cue count to the pre-specified 1--5 ordinal scale."""

    if hit_count == 0:
        return 1
    if hit_count == 1:
        return 3
    if hit_count <= 3:
        return 4
    return 5


def risk_band(score: float) -> str:
    if score >= 67:
        return "高"
    if score >= 34:
        return "中"
    return "低"


def score_postings(jobs: pd.DataFrame) -> pd.DataFrame:
    required = {"job_id", "company_name", "company_code", "job_title", "skills_raw", "source_sheet", "source_row"}
    missing = required - set(jobs.columns)
    if missing:
        raise ValueError(f"jobs file missing required columns: {sorted(missing)}")

    scored = jobs.loc[:, sorted(required)].copy()
    scored["task_text"] = (
        jobs["job_title"].map(clean_text) + " " + jobs["skills_raw"].map(clean_text)
    ).str.strip()

    positive_columns: list[str] = []
    negative_columns: list[str] = []
    for dimension in DIMENSIONS:
        evidence_column = f"evidence_{dimension.key}"
        score_column = f"score_{dimension.key}"
        evidence = scored["task_text"].map(lambda text: matched_terms(text, dimension.terms))
        scored[evidence_column] = evidence.map("、".join)
        scored[score_column] = evidence.map(len).map(ordinal_score)
        (positive_columns if dimension.raises_risk else negative_columns).append(score_column)

    positive_total = scored[positive_columns].sum(axis=1)
    reversed_negative_total = sum(6 - scored[column] for column in negative_columns)
    scored["risk_score"] = ((positive_total + reversed_negative_total) / 35 * 100).round(1)
    scored["risk_band"] = scored["risk_score"].map(risk_band)
    scored["analysis_unit"] = "公开招聘记录（职位名称+技能要求）"
    scored["rubric"] = RUBRIC
    return scored.drop(columns=["task_text"])


def unique_join(values: pd.Series) -> str:
    return "、".join(sorted({value for value in values.astype(str) if value}))


def summarize_roles(scored: pd.DataFrame) -> pd.DataFrame:
    score_columns = [f"score_{dimension.key}" for dimension in DIMENSIONS]
    evidence_columns = [f"evidence_{dimension.key}" for dimension in DIMENSIONS]
    aggregations: dict[str, object] = {
        "job_id": ["count", unique_join],
        "source_sheet": unique_join,
        "source_row": unique_join,
        "risk_score": "mean",
        **{column: "mean" for column in score_columns},
        **{column: unique_join for column in evidence_columns},
    }
    roles = (
        scored.groupby(["company_code", "company_name", "job_title"], dropna=False)
        .agg(aggregations)
        .reset_index()
    )
    roles.columns = [
        "company_code",
        "company_name",
        "job_title",
        "posting_count",
        "job_ids",
        "source_sheets",
        "source_rows",
        "risk_score",
        *score_columns,
        *evidence_columns,
    ]
    roles["risk_score"] = roles["risk_score"].round(1)
    for column in score_columns:
        roles[column] = roles[column].round(1)
    roles["risk_band"] = roles["risk_score"].map(risk_band)
    roles["rubric"] = RUBRIC
    return roles.sort_values(["risk_score", "posting_count", "company_name", "job_title"], ascending=[False, False, True, True])


def make_summary(scored: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    dimensions = {dimension.key: dimension.label for dimension in DIMENSIONS}
    rows: list[dict[str, object]] = [
        {"metric": "招聘记录数", "value": len(scored), "note": "去重后的公开招聘记录。"},
        {"metric": "公司-岗位组合数", "value": len(roles), "note": "按公司、岗位名称汇总；岗位名称保留招聘原文。"},
        {"metric": "平均任务自动化潜力评分", "value": round(float(scored["risk_score"].mean()), 1), "note": "七维等权研究设计结果。"},
    ]
    for band in ("高", "中", "低"):
        count = int((scored["risk_band"] == band).sum())
        rows.append({"metric": f"{band}风险招聘记录数", "value": count, "note": f"占比 {count / len(scored):.1%}"})
    for key, label in dimensions.items():
        column = f"score_{key}"
        rows.append({"metric": f"{label}平均分", "value": round(float(scored[column].mean()), 2), "note": "1至5分；分数由固定词典命中映射。"})
    return pd.DataFrame(rows)


def make_figure(roles: pd.DataFrame, output: Path) -> None:
    """Draw the ten highest and lowest scoring company-role combinations."""

    ensure_directories([output.parent])
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    display = roles.copy()
    display["label"] = display["company_name"] + " | " + display["job_title"]
    top = display.nlargest(10, "risk_score").sort_values("risk_score")
    bottom = display.nsmallest(10, "risk_score").sort_values("risk_score", ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for axis, frame, title, color in (
        (axes[0], top, "任务自动化潜力最高的公司-岗位组合", "#C84C3A"),
        (axes[1], bottom, "任务自动化潜力最低的公司-岗位组合", "#277A76"),
    ):
        axis.barh(frame["label"], frame["risk_score"], color=color)
        axis.set_xlim(0, 100)
        axis.set_xlabel("研究设计评分（0-100）")
        axis.set_title(title)
        axis.grid(axis="x", linestyle=":", alpha=0.5)
        for y, value in enumerate(frame["risk_score"]):
            axis.text(float(value) + 1, y, f"{value:.1f}", va="center", fontsize=8)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score task-automation potential from observed job postings.")
    parser.add_argument("--input", type=Path, default=repo_path("data", "processed", "jobs_clean.csv"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Required jobs input is missing: {args.input}")
    jobs = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    scored = score_postings(jobs)
    roles = summarize_roles(scored)
    summary = make_summary(scored, roles)
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    ensure_directories([tables, figures])
    scored.to_csv(tables / "ai_substitution_risk_job_level.csv", index=False, encoding="utf-8")
    roles.to_csv(tables / "ai_substitution_risk.csv", index=False, encoding="utf-8")
    summary.to_csv(tables / "ai_substitution_risk_summary.csv", index=False, encoding="utf-8")
    ranking = pd.concat(
        [roles.nlargest(10, "risk_score").assign(ranking_group="最高10"), roles.nsmallest(10, "risk_score").assign(ranking_group="最低10")],
        ignore_index=True,
    )
    ranking.to_csv(tables / "ai_substitution_risk_top_bottom.csv", index=False, encoding="utf-8")
    make_figure(roles, figures / "ai_substitution_risk.png")
    print(f"Scored {len(scored)} job record(s) and {len(roles)} company-role combination(s).")
    print(f"Wrote tables to {tables} and figure to {figures / 'ai_substitution_risk.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
