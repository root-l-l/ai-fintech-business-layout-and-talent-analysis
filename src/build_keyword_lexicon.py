"""Build and validate a transparent AI-fintech keyword lexicon (>=100 terms)."""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


LEXICON: dict[str, list[str]] = {
    "人工智能基础": ["人工智能", "机器学习", "深度学习", "神经网络", "监督学习", "无监督学习", "强化学习", "迁移学习", "联邦学习", "自监督学习", "知识蒸馏", "特征工程", "模型训练", "模型推理", "模型部署", "模型监控", "模型治理", "模型可解释性", "MLOps", "AIGC"],
    "大模型与语言技术": ["大语言模型", "生成式人工智能", "多模态模型", "预训练模型", "微调", "提示工程", "检索增强生成", "向量数据库", "嵌入模型", "语义检索", "文本分类", "命名实体识别", "信息抽取", "文本摘要", "智能问答", "对话机器人", "智能客服", "语音识别", "语音合成", "光学字符识别"],
    "视觉与智能决策": ["计算机视觉", "图像识别", "人脸识别", "活体检测", "行为识别", "异常检测", "预测模型", "推荐系统", "决策引擎", "规则引擎", "知识图谱", "图神经网络", "时序预测", "因果推断", "数字人"],
    "金融业务场景": ["银行IT", "核心银行系统", "手机银行", "开放银行", "数字银行", "智能风控", "信贷风控", "反欺诈", "反洗钱", "信用评分", "授信审批", "贷后管理", "智能投顾", "财富管理", "资产管理", "量化投资", "智能交易", "支付清算", "数字人民币", "跨境支付", "供应链金融", "绿色金融", "保险科技", "智能理赔", "监管科技", "合规科技", "财务SaaS", "业财一体化", "财务共享", "税务数字化", "RPA"],
    "数据与安全": ["金融大数据", "数据中台", "数据湖", "数据仓库", "实时计算", "流式计算", "隐私计算", "多方安全计算", "同态加密", "联邦建模", "数据脱敏", "数据治理", "数据质量", "数据血缘", "数据资产", "主数据管理", "身份认证", "零信任", "网络安全", "安全运营", "密码技术", "区块链", "分布式账本", "智能合约"],
    "工程与基础设施": ["云计算", "金融云", "云原生", "微服务", "分布式架构", "容器化", "DevOps", "低代码", "API网关", "服务网格", "信创", "国产化", "高可用", "容灾备份", "运维自动化", "AIOps", "边缘计算", "数字孪生"],
}


def main() -> int:
    rows = []
    for category, keywords in LEXICON.items():
        for keyword in keywords:
            rows.append({"keyword": keyword, "category": category, "definition": "研究者编制的检索词；用于文本匹配，不代表公司实际采用该技术。", "source_type": "researcher_curated", "source_reference": "课程作业研究词库 v1", "verified_by": "待双人复核", "verified_date": ""})
    frame = pd.DataFrame(rows).drop_duplicates(subset="keyword").sort_values(["category", "keyword"])
    if len(frame) < 100:
        raise RuntimeError(f"Lexicon has only {len(frame)} unique keywords; assignment requires at least 100.")
    output = repo_path("data", "reference", "ai_fintech_keywords.csv")
    ensure_directories([output.parent])
    frame.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(frame)} unique keywords to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
