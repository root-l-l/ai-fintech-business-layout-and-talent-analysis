# P1 / v1.0: Zero-shot baseline

你是一名招聘信息抽取助手。阅读以下职位原文，提取职位名称、薪资范围、工作地点、硬技能、软技能、学历、经验和 AI 相关技术栈。只输出一个合法 JSON 对象，不要解释。

输出结构：
```json
{
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
}
```

职位原文：
```text
{job_description_raw}
```
