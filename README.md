# AI金融科技上市公司业务布局与人才需求分析

本仓库对应《数字金融模拟实训》课程作业。研究以可复现、可核验为原则：代码不生成或补造公司、财务、年报、招聘、模型评估数据；所有外部材料必须在数据台账中记录来源、获取日期与本地路径。

## 当前进度

- 已建立项目结构、数据字典、候选样本复核表、来源台账和完整分析脚本。
- 已按第一阶段材料建立 15 家公司样本池，并导入相应年报、产品页链接与招聘信息。样本范围是本次课程研究的研究对象，不表述为“所有 A 股公司”。
- 已导入并清洗 242 条公开岗位记录；用于 Prompt 实验的 43 条分层样本，其人工金标准直接采用结构化招聘表，并保留来源行号。三版 Prompt 的真实模型输出已完成导入、校验和评价，正式结果见 `docs/report/06_part3_results.md`；不使用早期不含职位名称、薪资和地点字段的试运行结果。

## 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 可复现流程

以下命令均从仓库根目录运行。先运行离线校验和词库构建：

```bash
python3 src/validate_sources.py
python3 src/build_keyword_lexicon.py
python3 src/clean_jobs.py
python3 src/extract_report_sections.py
python3 src/semantic_analysis.py --method sentence-transformer --input-path data/interim/annual_report_sections_final15.csv
python3 src/build_knowledge_graph.py
python3 src/prepare_prompts.py
python3 src/build_gold_annotation_draft.py
python3 src/finalize_gold_annotations.py
python3 src/normalize_salaries.py
python3 src/evaluate_llm.py
python3 src/score_automation_risk.py
```

`score_automation_risk.py` reads the 242 cleaned job records and writes a
reviewable seven-dimension task-automation-potential rubric, company-role
rankings and a figure to `outputs/`. It is a researcher-defined scoring design,
not an observed probability of replacing a worker.

## 加分项复现

```bash
MPLBACKEND=Agg python3 src/reinforcement_learning_job_ranking.py
python3 src/fine_tune_domain_encoder.py
MPLBACKEND=Agg python3 src/create_bonus_visualizations.py
```

强化学习排序使用真实岗位作为候选项，但反馈是明确披露的模拟技能匹配奖励；它不包含真实点击或投递数据。微调脚本实际训练本机缓存的中文预训练文本编码器，并保留留出集、随机种子、权重和训练日志；该模型不是生成式大语言模型，详见 `docs/report/08_bonus_items.md`。

## 官网产品快照导入

官网快照用于补强产品线和应用场景的来源证据，不混入年报语义分析语料。导入DOCX文字快照时显式传入文件路径：

```bash
python3 src/import_product_snapshots.py --inputs \
  '/path/to/1-4 产品文字版.docx' \
  '/path/to/5-8产品全部.docx' \
  '/path/to/13-15产品.docx'
```

脚本会保存原始快照、生成公司级文本证据表、SHA-256 台账和覆盖表。原始 JPEG 截图可用以下命令另行归档；该步骤记录图像文件、页序、像素尺寸和哈希，不会从截图自动推断产品事实：

```bash
python3 src/import_product_screenshots.py \
  --input-dir '/path/to/9-12'
```

有关 DOCX 和 JPEG 快照如何用于知识图谱逐条核验，见 `docs/report/09_product_page_snapshot_evidence.md`。

第二部分的产品线证据和财务差异比较可按以下顺序重建：

```bash
python3 src/verify_product_line_claims.py
python3 src/build_product_line_evidence.py
python3 src/extract_financial_comparison.py
python3 src/build_financial_ai_comparison.py
python3 src/build_knowledge_graph.py
MPLBACKEND=Agg python3 src/create_financial_ai_figure.py
```

财务提取优先读取年报主要会计数据表，并保存页码、原文片段和提取方式。跨页表格的人工来源行只保存在 `data/reference/financial_metrics_2025_manual_corrections.csv`，不得用估算数值替代。

下载报告仅使用台账中的官方链接：

```bash
python3 src/download_reports.py
python3 src/extract_report_sections.py
```

最终报告的嵌入语义分析使用预训练模型：

```bash
python3 src/semantic_analysis.py --method sentence-transformer \
  --model-name BAAI/bge-small-zh-v1.5 \
  --input-path data/interim/annual_report_sections_final15.csv
```

该命令首次执行会下载公开预训练模型；请在 `outputs/tables/semantic_run_metadata.json` 记录模型版本、运行日期和环境。若下载失败，不得把 TF-IDF 结果写成“预训练词嵌入结果”。

## 数据规则

1. 年报：优先巨潮资讯网公告原件；下载后保留 PDF、URL、下载日期与 SHA-256。
2. 官网：保存 URL、页面标题、抓取日期和原始文本；遵守网站条款与 robots 规则。
3. 招聘：仅录入公开可访问、允许人工查看的信息；保留原始职位文本和链接。不要在代码中保存 Cookie、账号或 API Key。
4. 最低 50 条招聘信息是作业要求。若公开样本不足，保留实际数量并在报告中披露，不能补造。
5. Prompt 指标必须基于 `data/reference/gold_job_annotations.csv` 的人工来源标注计算。本项目的金标准来自用户确认的结构化招聘表；模型输出必须单独真实运行、保留时间戳与原始输出。

## 目录说明

- `data/raw/`：原始年报、官网文本、招聘记录，原则上不修改。
- `data/interim/`：抽取和清洗中间结果。
- `data/processed/`：分析输入数据。
- `data/reference/`：样本标准、来源台账、字段字典和人工标注。
- `outputs/`：表格、图、知识图谱和运行元数据。
- `docs/prompts/`：三套 Prompt 文档。
- `docs/PROMPT_DESIGN.md`：三版 Prompt 的设计逻辑、调用配置、评价口径和真实结果。
- `docs/REPOSITORY_SUBMISSION.md`：GitHub/Gitee 提交材料清单、忽略规则与复现实操步骤。
- `docs/report/FINAL_REPORT.md`：四个部分及加分项的最终文字报告。
- `src/`：全部可执行 Python 脚本。

## 关键限制

年报章节采用自动定位后的窗口提取，仍须在正式引用时核验页码。不得将 15 家研究样本的发现外推为全市场结论；不得在无真实模型输出时填写 Prompt 准确率、幻觉率或稳定性指标。
