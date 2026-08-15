# 8 加分项：强化学习、实际微调与可视化

本节对应作业要求中的三项加分内容，合计最多 10 分。所有候选岗位、标签、实验指标和图表均来自本仓库已有的真实数据或实际运行日志；其中的模拟部分单独标明，不能写成真实用户行为或真实业务部署。

## 8.1 强化学习岗位推荐排序（对应加分项 +2）

### 方法与数据边界

使用 `src/reinforcement_learning_job_ranking.py` 对 `jobs_clean.csv` 中 242 条真实、去重后的岗位记录运行 epsilon-greedy 多臂老虎机。候选项保留其真实 `job_id`、公司、岗位名称和来源行号。

本项目没有用户点击、投递、面试或录用反馈，因此不能伪造“真实推荐效果”。为展示强化学习的状态--行动--奖励--更新过程，定义了一个**模拟画像**“金融数据与 AI 应用学习者”，画像技能为 Python、SQL、金融、风控、数据分析、统计、机器学习和大模型。一个岗位的确定性模拟奖励等于其职位名称和技能要求中命中画像技能的个数除以 8。智能体以 epsilon-greedy 策略选择岗位，并以 $Q(a)\leftarrow Q(a)+\alpha[r-Q(a)]$ 更新动作价值。

### 实际运行结果

固定随机种子 `20260815`，运行 5,000 个模拟交互轮次，学习率 $\alpha=0.2$，探索率从 0.35 线性降至 0.02。随机策略的期望模拟奖励为 0.1291，最后 500 轮的平均模拟奖励为 0.6115，候选集合中最高的模拟技能匹配奖励为 0.6250。完整排序在 `outputs/tables/rl_job_recommendation_ranking.csv`，配置与边界说明在 `outputs/tables/rl_job_recommendation_config.json`。

![图8-1：Epsilon-greedy 岗位排序模拟学习曲线](/Users/emi/Documents/保险/outputs/figures/rl_job_recommendation_learning_curve.png)

**可写入报告的结论：** 在预先公开的模拟技能匹配奖励下，bandit 策略能够逐步优先选择高匹配候选岗位。**不能写：** 模型提高了真实用户投递率、就业率或薪资满意度；因为本研究没有此类行为数据。

## 8.2 小规模领域预训练模型的实际微调（对应加分项 +2）

### 训练设计

使用本机缓存的 `BAAI/bge-small-zh-v1.5`（BERT 架构，23,953,920 个参数）进行实际监督微调。为适配金融科技招聘语料，任务定义为判断岗位是否在用户提供的“AI 相关技术栈”字段中**明确出现** AI、人工智能、大模型、模型、智能体、Agent、机器学习、深度学习、NLP、自然语言、知识图谱、计算机视觉、算法或风控等词。

输入只使用“职位名称+技能要求”，有意不输入作为标签来源的 AI 技术栈字段，避免直接的标签泄漏。242 条记录按标签分层、随机种子 `20260815` 划分为 193 条训练记录与 49 条独立留出记录。训练时只解冻预训练编码器的最后一层与分类头，在 CPU 上训练 3 个 epoch；实际训练权重、随机种子、配置和逐轮日志均已保留。

### 留出集结果

| 评价集 | 样本量 | 准确率 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 独立留出集 | 49 | 0.6735 | 0.6389 | 0.8846 | 0.7419 |

最优结果出现在第 3 个 epoch。原始结果位于 `outputs/tables/domain_encoder_holdout_metrics.csv`，逐轮日志位于 `outputs/tables/domain_encoder_training_history.csv`，可复现的可训练层权重位于 `outputs/models/domain_encoder_bge_small_trainable_weights.pt`。

![图8-2：文本编码器实际微调的训练与留出集曲线](/Users/emi/Documents/保险/outputs/figures/domain_encoder_learning_curve.png)

**必须如实标注的限制：** `bge-small-zh-v1.5` 是预训练文本编码器，不是生成式大语言模型。本实验已经完成真实的小规模领域预训练模型微调，但若教师将“领域大模型”严格限定为生成式 LLM，则应将它表述为“实际微调的替代性实现”，不能声称已完成生成式 LLM 微调。49 条留出集较小，且标签来自用户提供的结构化 AI 技术栈字段，结果只说明本样本上的探索性分类表现，不能外推为市场水平。

## 8.3 展示型可视化（对应加分项 +6）

以下七张图均由代码从已有输出表生成，未手工填写图中数值：

| 图表 | 对应数据与作用 |
| --- | --- |
| `company_ai_direction_heatmap.png` | 15 家公司、5 类 AI 技术方向的年报语义相似度热力图，呈现业务布局差异。 |
| `product_line_knowledge_graph.png` | 基于图谱节点与边表的公司--产品线--AI 技术方向三层网络总览。 |
| `prompt_quality_comparison.png` | 三个 Prompt 的真实严格准确率、稳定性、操作性幻觉率与运行时间。 |
| `recruitment_skill_demand.png` | 242 条岗位文本中出现频率最高的技能和领域词。 |
| `ai_substitution_risk.png` | 第四部分七维规则下的最高/最低任务自动化潜力岗位组合。 |
| `rl_job_recommendation_learning_curve.png` | 强化学习模拟的滚动奖励与随机策略基线。 |
| `domain_encoder_learning_curve.png` | 小规模领域文本编码器的实际训练损失与留出集指标。 |

前四张图使用 `src/create_bonus_visualizations.py` 生成，后两项实验图分别由各自实验脚本生成，替代风险图由 `src/score_automation_risk.py` 生成。所有图均使用统一的中文字体、色板、图号、轴标签和数据来源说明，适合直接进入 PDF 报告或 10 分钟展示 PPT。

## 8.4 一键复现顺序

在仓库根目录执行：

```bash
MPLBACKEND=Agg python3 src/reinforcement_learning_job_ranking.py
python3 src/fine_tune_domain_encoder.py
MPLBACKEND=Agg python3 src/create_bonus_visualizations.py
```

微调脚本默认只读取本机已经缓存的基础模型，不会悄悄下载模型；当本地缓存缺失而且已获得网络许可时，才可显式追加 `--allow-download`。所有加分项的结论边界、输入来源和输出路径均由脚本和本节同时记录。
