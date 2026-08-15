# GitHub/Gitee 代码仓库提交说明

## 1. 建议仓库信息

- 仓库名：`ai-fintech-business-layout-and-talent-analysis`
- 可见性：课程提交建议设为私有；如教师要求公开，先确认用户提供的招聘表、官网截图和年报是否允许公开再调整。
- 默认分支：`main`
- 项目入口：[README.md](../README.md)
- Prompt 文档：[PROMPT_DESIGN.md](PROMPT_DESIGN.md)
- 最终报告：[FINAL_REPORT.md](report/FINAL_REPORT.md)

当前本地 Git 仓库没有远程地址，因此本项目尚未发布到 GitHub 或 Gitee。以下材料已整理为可提交状态，但不会在未获得账户授权与目标仓库地址时自行推送。

## 2. 建议提交内容

应提交：

- `src/`：完整可执行 Python 脚本。
- `docs/`：研究报告、Prompt 设计、数据边界与复现说明。
- `data/reference/`、`data/interim/`、`data/processed/`：必要的结构化台账、中间表、人工金标准与验证结果。
- `outputs/tables/`、`outputs/knowledge_graph/`、`outputs/figures/`：由脚本生成的结果，便于教师复核。
- `requirements.txt`、`.gitignore`、`README.md`。

默认不提交：

- 用户提供的原始年报 PDF、原始招聘 CSV、官网 DOCX 快照及 JPEG 截图。这些文件体积较大，且可能受到版权、平台条款或课程数据管理约束；本项目保留其文件清单、哈希、页码、路径规则和导入脚本。
- `.venv/`、缓存、`.env`、API Key 和其他密钥。

若教师要求提交原始材料，应先确认许可与上传容量，再单独建立受限访问的课程附件或私有仓库；不要把 API Key、Cookie 或账号信息加入版本控制。

## 3. 克隆后的复现步骤

```bash
git clone <GitHub-or-Gitee-repository-url>
cd ai-fintech-business-layout-and-talent-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

原始材料必须按 `data/reference/annual_report_manifest.csv` 与 `data/raw/README.md` 的说明放置。然后从仓库根目录运行：

```bash
python3 src/validate_sources.py
python3 src/extract_report_sections.py
python3 src/semantic_analysis.py --method sentence-transformer \
  --input-path data/interim/annual_report_sections_final15.csv
python3 src/summarize_ai_directions.py
python3 src/verify_product_line_claims.py
python3 src/build_product_line_evidence.py
python3 src/extract_financial_comparison.py
python3 src/build_financial_ai_comparison.py
python3 src/build_knowledge_graph.py
MPLBACKEND=Agg python3 src/create_financial_ai_figure.py
```

Prompt 实验的离线复评必须显式筛选正式模型与运行轮次：`src/validate_llm_outputs.py --model-id deepseek-v4-flash --run-ids FULL_R1 FULL_R2`、`src/evaluate_llm.py --model-id deepseek-v4-flash --run-id FULL_R1` 和 `src/evaluate_prompt_stability.py --model-id deepseek-v4-flash --run-ids FULL_R1 FULL_R2`。重新调用模型需要自行配置兼容 API 的访问地址与 API Key；历史输出不会被伪造或由脚本自动补全。

## 4. 上传命令

在 GitHub 或 Gitee 网页新建一个空仓库后，二选一执行：

```bash
# GitHub
git branch -M main
git add .
git commit -m "Add reproducible course project"
git remote add origin https://github.com/<username>/ai-fintech-business-layout-and-talent-analysis.git
git push -u origin main
```

```bash
# Gitee
git branch -M main
git add .
git commit -m "Add reproducible course project"
git remote add origin https://gitee.com/<username>/ai-fintech-business-layout-and-talent-analysis.git
git push -u origin main
```

提交前执行 `git status --short`，确认原始 PDF、截图、`.env` 和虚拟环境未被暂存。若仓库已有远程地址，只需使用已有地址，不要重复添加 `origin`。

## 5. 可复现性与学术诚信说明

本仓库的年报语义结果来自 `BAAI/bge-small-zh-v1.5` 的实际嵌入运行；Prompt 指标来自 `deepseek-v4-flash` 的实际输出；财务数值保留年报页码、原文片段与跨页人工选值记录。产品线验证区分全部实体文本验证、部分验证与当前材料未命中，未命中不被改写为产品不存在。强化学习奖励为模拟技能匹配奖励，领域微调使用的是文本编码器而非生成式 LLM。任何复现者均不应删除这些边界或将其表述为未观察到的业务效果。
