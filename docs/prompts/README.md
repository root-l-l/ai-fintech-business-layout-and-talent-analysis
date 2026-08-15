# Prompt 实验说明

运行 `python3 src/prepare_prompts.py` 后，本目录会生成 `prompt_v1.md`、`prompt_v2.md`、`prompt_v3.md`。

## 批量实验

推荐在本项目文件夹的**终端**运行 `src/run_prompt_batch.py`，而不是在聊天网页逐条复制。它调用你自己拥有密钥的、兼容 OpenAI Chat Completions 的模型 API；脚本只发送各岗位的 `model_input`，不会发送人工金标准。

先在终端进入项目并设置 API Key（Key 只保存在当前终端窗口）：

```bash
cd "/absolute/path/to/project"
export LLM_API_KEY='粘贴你自己的API密钥'
```

先生成完整岗位文本的输入。每条输入包含源招聘记录中的职位名称、薪资范围、工作地点与职位描述正文；不包含人工金标准文件。第一轮质量比较：P1、P2、P3 各跑 43 条，自动保存 129 次真实调用的原始回答、解析 JSON、模型名与开始/结束时间。使用新的 `FULL_R1`，不要复用早期仅含描述正文的 `R1`。

```bash
.venv/bin/python src/prepare_prompts.py
.venv/bin/python src/build_prompt_run_inputs.py

.venv/bin/python src/run_prompt_batch.py \
  --base-url '你的API基础地址' \
  --model '你的实际模型ID' \
  --run-id FULL_R1 \
  --temperature 0 \
  --workers 2
```

### Gemini 免费层的建议设置

Gemini API 免费层的实际请求配额取决于 AI Studio 的项目和地区。先在 AI Studio 的 API Keys 页面确认该 Key 对应的项目和模型可用；`gemini-1.5-pro` 已停止服务，不能再作为实验模型。为尽量避免 `429` 限流，使用当前 AI Studio 控制台可见的模型 ID，设置单并发和 8 秒请求间隔。若出现 `403 PERMISSION_DENIED`，停止脚本，到 AI Studio 确认 Key 属于当前项目、该项目可调用所选模型且地区/账号具备权限后再恢复；403 不是靠重试可解决的问题。

```bash
export LLM_API_KEY='你的Gemini_API密钥'
.venv/bin/python src/run_prompt_batch.py \
  --base-url 'https://generativelanguage.googleapis.com/v1beta/openai' \
  --model '在AI Studio控制台可用的实际模型ID' \
  --run-id FULL_R1 \
  --temperature 0 \
  --workers 1 \
  --request-interval-seconds 8 \
  --no-json-mode
```

同一 `run_id` 恢复时，脚本自动跳过已经保存的岗位，不会重复调用。每次成功结果均立即写入台账；当日额度不足时，隔日再次运行同一条命令即可继续未完成的岗位。

第二轮稳定性检验仅复跑每家公司随机固定的 1 条，共 15 条 x 3 个 Prompt = 45 次调用。它是节约调用成本的分层子样本稳定性检验，报告中必须写明“15 家公司、每家 1 条岗位”，不能表述为全 43 条的稳定性。使用新的 `FULL_R2`。

```bash
.venv/bin/python src/prepare_stability_sample.py
.venv/bin/python src/run_prompt_batch.py \
  --base-url '你的API基础地址' \
  --model '与第一轮完全相同的模型ID' \
  --run-id FULL_R2 \
  --temperature 0 \
  --workers 2 \
  --job-ids-file data/interim/prompt_stability_sample.csv
```

若服务商报错 `response_format` 不支持，在两条命令最后加 `--no-json-mode`；Prompt 本身仍要求只输出 JSON。运行完成后：

```bash
.venv/bin/python src/validate_llm_outputs.py --model-id '你的实际模型ID' --run-ids FULL_R1 FULL_R2
.venv/bin/python src/evaluate_llm.py --model-id '你的实际模型ID' --run-id FULL_R1
.venv/bin/python src/evaluate_prompt_stability.py --model-id '你的实际模型ID' --run-ids FULL_R1 FULL_R2
```

输出分别写入 `outputs/tables/llm_output_validation.json`、`prompt_evaluation.csv` 和 `prompt_stability.csv`。模型不得访问未在职位原文中提供的信息；无法判断的字段必须返回 `null` 或空数组。

若曾用不同模型尝试同一 `run_id`，先运行 `src/migrate_llm_output_paths.py` 一次。它会将各 JSON 输出复制进以模型名区分的目录，并备份调用台账。质量评价和稳定性评价都必须显式传入 `--model-id`，防止混合不同模型的结果。

### DeepSeek（正式实验模型）

DeepSeek 的 Chat Completions API 兼容本脚本，支持 JSON 输出。若你的账号可用 `deepseek-v4-flash`，它适合这种短文本、字段固定的批量抽取任务；使用 `temperature=0` 和非思考模式可显著减少等待时间。DeepSeek 的 API 文档说明 `response_format={"type":"json_object"}` 需要 Prompt 内包含 JSON 指令和示例，本项目的三套 Prompt 已满足这一条件。[DeepSeek JSON Output 文档](https://api-docs.deepseek.com/guides/json_mode)

```bash
export LLM_API_KEY='你的DeepSeek_API密钥'
.venv/bin/python src/run_prompt_batch.py \
  --base-url 'https://api.deepseek.com' \
  --model 'deepseek-v4-flash' \
  --run-id FULL_R1 \
  --temperature 0 \
  --max-tokens 1200 \
  --thinking-mode deepseek \
  --workers 3 \
  --timeout-seconds 90
```

第二轮只需将 `--run-id FULL_R1` 改为 `--run-id FULL_R2`，并在命令末尾加：

```bash
--job-ids-file data/interim/prompt_stability_sample.csv
```

不要把 `--thinking-mode deepseek` 用于硅基流动或 Gemini；该参数会按服务商选择对应请求字段。
