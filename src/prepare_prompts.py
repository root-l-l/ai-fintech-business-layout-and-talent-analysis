"""Generate three version-controlled prompts for job-posting extraction."""

from __future__ import annotations

from common import ensure_directories, repo_path


SCHEMA = '''{
  "job_title": null,
  "salary_range": null,
  "work_location": null,
  "hard_skills": [],
  "soft_skills": [],
  "education": null,
  "experience": null,
  "ai_tech_stack": [],
  "evidence": {
    "job_title": null,
    "salary_range": null,
    "work_location": null,
    "hard_skills": null,
    "soft_skills": null,
    "education": null,
    "experience": null,
    "ai_tech_stack": null
  }
}'''


PROMPTS = {
    "prompt_v1.md": f"""# P1 / v1.0: Zero-shot baseline

你是一名招聘信息抽取助手。阅读以下职位原文，提取职位名称、薪资范围、工作地点、硬技能、软技能、学历、经验和 AI 相关技术栈。只输出一个合法 JSON 对象，不要解释。

输出结构：
```json
{SCHEMA}
```

职位原文：
```text
{{job_description_raw}}
```
""",
    "prompt_v2.md": f"""# P2 / v2.0: Schema and field constraints

你是一名审慎的招聘信息结构化提取助手。仅依据职位原文抽取信息；任何未明确出现的信息均不得推测。单值字段未提及返回 `null`，列表字段未提及返回 `[]`。只输出一个合法 JSON 对象，不要输出 Markdown 或解释。

字段规则：
1. `salary_range` 保留原文薪资表达，例如 `15-25K`、`20-30K·14薪`、`面议`；不得自行换算或补充。
2. `hard_skills` 仅填写语言、工具、框架、数据库、模型、行业技术或可操作方法；`soft_skills` 仅填写沟通、协作、学习、责任心、抗压等行为能力。
3. `ai_tech_stack` 仅填写原文明确提到的 AI 模型、算法、框架、智能体、RAG、NLP 或 AI 工具；不能因公司或岗位属于金融科技而添加。
4. `job_title`、`work_location`、`education` 和 `experience` 保留原文含义，不要擅自归类或缩写。
5. 每个非空字段须在 `evidence` 中给出简短原文依据；没有可引用证据则该字段置空。

输出结构：
```json
{SCHEMA}
```

职位原文：
```text
{{job_description_raw}}
```
""",
    "prompt_v3.md": f"""# P3 / v3.0: One-shot, anti-hallucination, and audit

你是一名审慎的招聘信息结构化提取助手。仅依据职位原文抽取信息。最终只输出一个合法 JSON 对象，不输出 Markdown、解释或推理过程。

硬性规则：
1. 原文没有明确写出的信息，绝对不得根据常识、公司名称或岗位名称补充。单值字段用 `null`，列表字段用 `[]`。
2. `salary_range` 保留原文表达，不换算为年薪或数值；`面议`、`未公示`是原文薪资表达，不得改写为 `null`。
3. 不把“优先”“加分项”“了解”“熟悉”改写成必备条件。可提取该技术或能力，但其限定语须保留在 `evidence`。
4. `hard_skills` 只包括语言、工具、框架、数据库、模型、行业技术或可操作方法；`soft_skills` 只包括行为能力；AI 技术栈只包括明确出现的 AI 技术。
5. 输出前静默检查每个非空字段是否有原文依据；没有依据则置空。列表去重并尽量保留原文术语。

字段定义与输出结构：
```json
{SCHEMA}
```

以下为格式示例。它是虚构示例，只展示标注方法，不是待处理职位的一部分：

示例职位原文：
```text
岗位：金融数据工程师。工作地点：北京。薪资：20-30K·14薪。技能要求：硬技能：Python、SQL、数据仓库。软技能：沟通能力、团队协作。学历要求：本科及以上。经验要求：3-5年。AI相关技术栈要求：无明确要求。
```

示例输出：
```json
{{
  "job_title": "金融数据工程师",
  "salary_range": "20-30K·14薪",
  "work_location": "北京",
  "hard_skills": ["Python", "SQL", "数据仓库"],
  "soft_skills": ["沟通能力", "团队协作"],
  "education": "本科及以上",
  "experience": "3-5年",
  "ai_tech_stack": [],
  "evidence": {{
    "job_title": "岗位：金融数据工程师",
    "salary_range": "薪资：20-30K·14薪",
    "work_location": "工作地点：北京",
    "hard_skills": "硬技能：Python、SQL、数据仓库",
    "soft_skills": "软技能：沟通能力、团队协作",
    "education": "学历要求：本科及以上",
    "experience": "经验要求：3-5年",
    "ai_tech_stack": "AI相关技术栈要求：无明确要求"
  }}
}}
```

待提取职位原文：
```text
{{job_description_raw}}
```
""",
}


def main() -> int:
    directory = repo_path("docs", "prompts")
    ensure_directories([directory])
    for filename, content in PROMPTS.items():
        (directory / filename).write_text(content, encoding="utf-8")
    print(f"Wrote {len(PROMPTS)} prompt versions to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
