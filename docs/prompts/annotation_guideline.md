# 职位抽取人工标注指南

标注对象：`data/interim/prompt_experiment_sample.csv` 的 43 个 `job_id`。本项目的招聘表已由研究团队按字段整理，经确认后可直接作为人工金标准来源。运行 `src/build_gold_annotation_draft.py` 和 `src/finalize_gold_annotations.py` 即可生成金标准；不需要重新录入岗位信息。不得根据公司名称、常识或岗位标题补写原文未披露的信息。

1. `job_title`：原职位名称，通常可直接从原表填写。
2. `salary_range`、`work_location`、`education`、`experience`：保留职位原文；缺失则留空。薪资均按人民币理解；除明确标注 `元/天`、`面议`、`未公示` 或未提及外，均为月薪，且 `1K=1,000 元`。金额换算只用于后续描述统计，金标准中的 `salary_range` 仍保留原文，例如 `15-25K`、`25-40K·16薪`、`面议`。
3. `hard_skills`：用 `|` 分隔，填写语言、框架、数据库、工具、模型、行业技术或可操作方法。
4. `soft_skills`：用 `|` 分隔，填写沟通、协作、学习、责任心、抗压等行为能力。
5. `ai_tech_stack`：用 `|` 分隔，仅填写明确出现的 AI 模型、AI框架、算法、智能体、RAG、NLP 等；“无”“无明确要求”或原文未提及均标为空。
6. `annotation_notes`：记录歧义、优先项和无法判断原因。若修订某一行，填写自己的姓名或代号至 `annotator`，并将 `review_status` 改为 `human_confirmed`。未修订行保留 `user_provided_structured_annotation`，同样可参与评价，且会在金标准中保留来源说明。

完成后运行 `src/finalize_gold_annotations.py`，它会把 `human_confirmed` 和 `user_provided_structured_annotation` 的记录写入 `data/reference/gold_job_annotations.csv`。若想比较三版 Prompt，三版均须对同一批岗位产出真实模型输出；不能把源表预填内容当成模型输出。

金标准仅用于评估 Prompt 输出，不能由同一模型自动生成后再声称“人工准确率”。
