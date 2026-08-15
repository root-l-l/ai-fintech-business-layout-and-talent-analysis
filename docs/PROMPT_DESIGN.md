# Prompt 设计文档

## 1. 任务与边界

本实验使用大语言模型从公开招聘信息中抽取八个字段：职位名称、薪资范围、工作地点、硬技能、软技能、学历要求、经验要求与 AI 相关技术栈。输入是同一条岗位的职位名称、原始薪资、工作地点和职位描述拼接后的完整文本；人工金标准不进入模型输入。

所有薪资保留原文表达。除原文明确为日薪、面议、未公示或未提及外，`K` 按千元人民币月薪理解；`14薪`、`15薪` 是发薪月数，不转换为月薪或年薪。模型不能根据公司名称、职位名称或行业常识补全未出现的字段。

## 2. 输出契约

三版 Prompt 共享同一个 JSON Schema，因此版本差异只来自提示约束，不来自评价字段变化。

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

单值字段未提及返回 `null`；列表字段未提及返回 `[]`。`evidence` 仅保存简短原文依据，不作为人工金标准。

## 3. 三个版本

| 版本 | 设计目的 | 新增机制 | 完整模板 |
|---|---|---|---|
| P1 / v1.0 | 建立 zero-shot 基线 | 固定 JSON 输出与字段列表 | [prompt_v1.md](prompts/prompt_v1.md) |
| P2 / v2.0 | 降低格式与缺失字段推断 | 字段定义、`null`/`[]` 规则、逐字段原文证据 | [prompt_v2.md](prompts/prompt_v2.md) |
| P3 / v3.0 | 进一步抑制无证据补全 | 虚构 one-shot 格式示例、反幻觉规则、静默自检、保留“优先/加分项”限定 | [prompt_v3.md](prompts/prompt_v3.md) |

P1 只要求完成结构化抽取。P2 要求每个非空字段存在简短原文证据，并明确区分硬技能、软技能与 AI 技术栈。P3 不改变 Schema；它增加一个与研究样本无关的虚构示例，要求输出前静默核对证据，并禁止将“优先”“了解”“熟悉”等限定语改写为硬性必备条件。

## 4. 执行配置

正式实验模型为 `deepseek-v4-flash`，固定 `temperature=0`、JSON 输出模式、最大 1,200 token 和三并发。`FULL_R1` 对 43 条按公司分层的岗位样本分别运行 P1--P3，共 129 次调用；`FULL_R2` 对每家公司固定 1 条岗位的 15 条子样本复跑，共 45 次调用。174 个输出均通过 JSON 结构校验。

```bash
python3 src/prepare_prompts.py
python3 src/build_prompt_run_inputs.py

# 需要自行设置兼容 API 的密钥；密钥不得写入仓库。
export LLM_API_KEY='...'
python3 src/run_prompt_batch.py \
  --base-url '<provider-base-url>' \
  --api-key-env LLM_API_KEY \
  --model deepseek-v4-flash \
  --run-id FULL_R1 \
  --versions P1 P2 P3

# 稳定性复跑：先固定 15 家公司各 1 条岗位，再以新的运行编号调用。
python3 src/prepare_stability_sample.py
python3 src/run_prompt_batch.py \
  --base-url '<provider-base-url>' \
  --api-key-env LLM_API_KEY \
  --model deepseek-v4-flash \
  --run-id FULL_R2 \
  --versions P1 P2 P3 \
  --job-ids-file data/interim/prompt_stability_sample.csv

python3 src/validate_llm_outputs.py \
  --model-id deepseek-v4-flash --run-ids FULL_R1 FULL_R2
python3 src/evaluate_llm.py --model-id deepseek-v4-flash --run-id FULL_R1
python3 src/evaluate_prompt_stability.py \
  --model-id deepseek-v4-flash --run-ids FULL_R1 FULL_R2
```

真实调用日志保存在 `data/reference/llm_outputs.csv`，每条记录包含岗位 ID、Prompt 版本、运行轮次、模型 ID、起止时间、原始模型输出及解析结果路径。该文件中已有的正式实验结果可以复评；重新调用会受模型版本、服务端状态和网络条件影响。

## 5. 评价口径

字段准确率使用严格比较：职位名称、薪资、地点、学历和经验在归一化后必须完全一致；硬技能、软技能和 AI 技术栈按无序集合完全一致。版本级准确率是八个字段准确率的宏平均。操作性幻觉率定义为“金标准为空而模型输出非空”的字段比例；它不覆盖金标准非空但内容错误的情形。稳定性为同模型、同 Prompt、同岗位在 `FULL_R1` 与 `FULL_R2` 间的逐字段两两一致率。

这种严格评价会低估“人工金标准使用复合短语、模型输出原子技能”时的语义相近结果。因此，结果应同时保留字段级指标、原始输出和证据字段，而不只报告单一宏平均数。

## 6. 实际结果与选择

| Prompt | 严格宏平均准确率 | 操作性幻觉率 | 平均运行时间（秒） | 宏平均跨轮一致性 |
|---|---:|---:|---:|---:|
| P1 | 0.6279 | 0.0465 | 3.119 | 0.9250 |
| P2 | 0.6482 | 0.0494 | 4.846 | 0.9583 |
| P3 | 0.6744 | 0.0494 | 2.986 | 0.9333 |

在本实验样本和该模型下，P3 的严格宏平均准确率最高，P2 的跨轮一致性最高。该结论不外推到其他模型、温度设置、岗位样本或日期。完整结果见 [Part 3 结果报告](report/06_part3_results.md) 与 `outputs/tables/prompt_evaluation.csv`、`outputs/tables/prompt_stability.csv`。
