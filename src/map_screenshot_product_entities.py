"""Create an auditable, page-by-page review of official product screenshots.

The annotations in this module are deliberate human readings of the 37 supplied
screenshots.  A product is added to the financial-AI graph only when the page
shows both a distinct product/solution entity and explicit AI (or an accepted
AI-adjacent) technical evidence in a financial business context.  Pages that do
not meet both conditions remain in the review table instead of being discarded.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from common import ensure_directories, repo_path


def _review(
    company_code: str,
    page: int,
    title: str,
    outcome: str,
    note: str,
    entities: tuple[str, ...] = (),
    terms: str = "",
) -> dict[str, object]:
    return {
        "company_code": company_code,
        "screenshot_page": page,
        "page_title": title,
        "review_outcome": outcome,
        "review_note": note,
        "identified_product_entities": "；".join(entities),
        "explicit_ai_terms": terms,
    }


# Every image in data/processed/product_page_screenshots.csv appears once below.
# ``included_financial_ai_entity`` means at least one separately named entity is
# written to ENTITY_DETAILS.  The other labels deliberately preserve negatives.
PAGE_REVIEWS = [
    _review("300872.SZ", 1, "金融科技产品", "no_distinct_financial_ai_entity", "产品总览展示多类金融科技产品，但本页未给出可与明确 AI 证据一一对应的独立产品实体。", terms="大数据"),
    _review("300872.SZ", 2, "金融 IT 服务", "no_distinct_financial_ai_entity", "服务介绍含 AI 等技术语境，但未列出独立金融 AI 产品名称。", terms="AI；大数据"),
    _review("300872.SZ", 3, "云计算", "no_distinct_financial_ai_entity", "页面列示云计算、大数据和 RPA 方案；RPA 未在本页与独立金融 AI 产品实体形成明确对应，故不入图。", terms="RPA；大数据"),
    _review("300872.SZ", 4, "运营服务", "no_distinct_financial_ai_entity", "运营服务页面未出现可确认的独立金融 AI 产品实体。"),
    _review("300872.SZ", 5, "咨询服务", "no_distinct_financial_ai_entity", "咨询服务页面未出现可确认的独立金融 AI 产品实体。"),
    _review("300663.SZ", 1, "科蓝软件产品与解决方案", "included_financial_ai_entity", "页面同时展示具体金融产品实体及 AI/OCR/数字机器人技术描述。", ("AI原生手机银行", "智能高柜数字机器人“小蓝”", "Galaxy OCR证件识别"), "AI原生；数字机器人；OCR"),
    _review("002987.SZ", 1, "盘庚测试平台类", "included_financial_ai_entity", "测试平台页面明确说明引入人工智能和大数据技术。", ("盘庚测试平台类",), "人工智能；大数据"),
    _review("002987.SZ", 2, "影像及流程管理类", "included_financial_ai_entity", "页面列出影像、档案和协同办公产品，并明确出现 OCR、自然语言处理和大模型算法。", ("统一影像管理平台", "综合档案管理系统", "数字化协同办公平台"), "OCR；自然语言处理；大模型算法；智能机器人"),
    _review("002987.SZ", 3, "大数据类", "no_distinct_financial_ai_entity", "页面为大数据产品目录，未见可与明确 AI 技术证据对应的独立金融 AI 产品实体。", terms="大数据"),
    _review("002987.SZ", 4, "AI 人工智能类", "included_financial_ai_entity", "页面直接列示 OCR、RPA、NLP/大模型和知识助手产品。", ("OCR智能识别服务平台", "RPA机器人流程自动化平台", "NLP及大模型算法服务平台", "企业知识助手"), "OCR；RPA；NLP；大模型；知识助手"),
    _review("002987.SZ", 5, "基础技术平台类", "included_financial_ai_entity", "流程平台说明采用人工智能识别技术；统一开发平台仅为技术栈罗列，未单独计入。", ("流程平台",), "人工智能识别技术"),
    _review("002987.SZ", 6, "运营类", "included_financial_ai_entity", "四项运营产品分别出现 AI-OCR、计算机视觉、OCR/AI 和机器学习风险模型证据。", ("集约化运营系统（含集中授权）", "数字化运营管理平台", "事后监督系统", "运营风险监控系统"), "AI-OCR；计算机视觉；OCR；AI；机器学习"),
    _review("002987.SZ", 7, "渠道类", "no_distinct_financial_ai_entity", "页面为移动银行、网上银行等渠道产品，出现“智能化”表述但未给出明确 AI 技术证据。", terms="智能化"),
    _review("002987.SZ", 8, "资产管理及同业类", "included_financial_ai_entity", "资管、理财销售和基金代销产品页面分别出现 AI 模型、OCR/NLP 与智能推荐。", ("资产管理系统", "理财销售系统", "基金代销系统"), "人工智能；AI模型；OCR；NLP；智能推荐"),
    _review("002987.SZ", 9, "供应链金融及信贷类", "included_financial_ai_entity", "供应链金融服务平台页面明确出现人工智能、合同审查与 OCR。", ("供应链金融服务平台",), "人工智能；OCR"),
    _review("002987.SZ", 10, "风险及合规管理类", "included_financial_ai_entity", "反洗钱监测分析系统页面明确出现人工智能和大语言模型。", ("反洗钱监测分析系统",), "人工智能；大语言模型"),
    _review("002987.SZ", 11, "数字化管理及创新服务类", "explicit_ai_but_non_financial_scope", "企业级数字化管理系统出现 AI 描述，但该页面主体为通用企业管理，不作为金融业务范围内产品实体入图。", terms="AI"),
    _review("002987.SZ", 12, "咨询服务", "no_distinct_financial_ai_entity", "页面为咨询服务介绍，未列出满足入图条件的独立金融 AI 产品实体。", terms="大数据；智能营销"),
    _review("002987.SZ", 13, "软件开发服务", "no_distinct_financial_ai_entity", "软件开发服务页面未出现可确认的独立金融 AI 产品实体。"),
    _review("002987.SZ", 14, "软件测试服务", "no_distinct_financial_ai_entity", "测试服务包含自动化测试描述，但未出现明确 AI 技术证据和独立金融 AI 产品实体的组合。", terms="自动化测试"),
    _review("002987.SZ", 15, "IT 运维服务", "no_distinct_financial_ai_entity", "IT 运维服务页面未出现可确认的独立金融 AI 产品实体。"),
    _review("002987.SZ", 16, "数字化运营服务", "no_distinct_financial_ai_entity", "数字化运营服务页面仅见泛化“智能化”描述，未达到明确 AI 技术证据标准。", terms="智能化"),
    _review("002987.SZ", 17, "智慧营销及客户服务", "no_distinct_financial_ai_entity", "页面为营销与客服服务介绍，未列出可确认的独立金融 AI 产品实体。", terms="智能化"),
    _review("603927.SH", 1, "财产保险", "included_financial_ai_entity", "财产保险产品体系中明确列出人工智能平台，属于保险业务范围。", ("人工智能平台",), "人工智能"),
    _review("603927.SH", 2, "人寿保险", "no_distinct_financial_ai_entity", "寿险产品目录出现商业智能表述，但未给出明确 AI 技术证据与独立实体的组合。", terms="商业智能"),
    _review("603927.SH", 3, "健康医疗", "explicit_ai_but_non_financial_scope", "中医智能诊断系统属于医疗场景，不纳入金融 AI 产品图谱。", terms="智能诊断"),
    _review("603927.SH", 4, "政府领域", "no_distinct_financial_ai_entity", "政府领域页面未出现可确认的金融 AI 产品实体。"),
    _review("603927.SH", 5, "金融领域", "no_distinct_financial_ai_entity", "金融领域产品目录未给出明确 AI 技术证据与独立产品实体的组合。", terms="商业智能；智能报表"),
    _review("603927.SH", 6, "媒体领域", "explicit_ai_but_non_financial_scope", "媒体领域页面不属于本研究的金融业务产品范围。"),
    _review("603927.SH", 7, "能源领域", "explicit_ai_but_non_financial_scope", "能源领域页面不属于本研究的金融业务产品范围。"),
    _review("603927.SH", 8, "邮政领域", "explicit_ai_but_non_financial_scope", "邮政领域页面不属于本研究的金融业务产品范围。"),
    _review("603927.SH", 9, "呼叫中心", "explicit_ai_but_non_financial_scope", "呼叫中心页面未限定为金融业务场景，故不纳入金融 AI 产品实体。"),
    _review("603927.SH", 10, "交通领域", "explicit_ai_but_non_financial_scope", "交通领域页面不属于本研究的金融业务产品范围。"),
    _review("603927.SH", 11, "民航领域", "explicit_ai_but_non_financial_scope", "民航领域页面不属于本研究的金融业务产品范围。", terms="智能安检"),
    _review("603927.SH", 12, "纪检监察", "explicit_ai_but_non_financial_scope", "纪检监察业务页面不属于本研究的金融业务产品范围。", terms="智能化"),
    _review("603927.SH", 13, "组工行业", "explicit_ai_but_non_financial_scope", "组工行业页面虽出现大数据人工智能，但不属于金融业务产品范围。", terms="大数据；人工智能"),
    _review("603927.SH", 14, "企业信息化", "no_distinct_financial_ai_entity", "页面提及服务金融保险客户，但未出现明确 AI 技术证据与独立金融产品实体的组合。"),
]


# One record equals one distinct entity added to the graph.  The technical
# category is normalized to the six-category keyword taxonomy used in Part 2.
ENTITY_DETAILS = [
    ("300663.SZ", "AI原生手机银行", "人工智能基础", "智能渠道与手机银行", 1, "页面列示“AI原生手机银行”。"),
    ("300663.SZ", "智能高柜数字机器人“小蓝”", "人工智能基础", "银行网点服务", 1, "页面列示“智能高柜数字机器人‘小蓝’”。"),
    ("300663.SZ", "Galaxy OCR证件识别", "视觉与智能决策", "证件识别与网点业务", 1, "页面列示“Galaxy OCR证件识别”。"),
    ("002987.SZ", "盘庚测试平台类", "工程与基础设施", "金融软件测试", 1, "页面说明将人工智能和大数据技术引入测试平台。"),
    ("002987.SZ", "统一影像管理平台", "视觉与智能决策", "金融影像管理", 2, "页面说明使用 OCR。"),
    ("002987.SZ", "综合档案管理系统", "大模型与语言技术", "金融档案管理", 2, "页面说明使用自然语言处理技术和大模型算法。"),
    ("002987.SZ", "数字化协同办公平台", "视觉与智能决策", "金融运营协同", 2, "页面说明使用 OCR、智能纠错和智能机器人。"),
    ("002987.SZ", "OCR智能识别服务平台", "视觉与智能决策", "金融文档识别", 4, "页面直接列示 OCR 智能识别服务平台。"),
    ("002987.SZ", "RPA机器人流程自动化平台", "工程与基础设施", "金融流程自动化", 4, "页面直接列示 RPA 机器人流程自动化平台。"),
    ("002987.SZ", "NLP及大模型算法服务平台", "大模型与语言技术", "金融文本与知识处理", 4, "页面直接列示 NLP 及大模型算法服务平台。"),
    ("002987.SZ", "企业知识助手", "大模型与语言技术", "金融知识服务", 4, "页面直接列示企业知识助手。"),
    ("002987.SZ", "流程平台", "工程与基础设施", "金融业务流程设计", 5, "页面说明流程平台采用人工智能识别技术。"),
    ("002987.SZ", "集约化运营系统（含集中授权）", "视觉与智能决策", "银行集中运营", 6, "页面说明智能授权使用 AI-OCR。"),
    ("002987.SZ", "数字化运营管理平台", "视觉与智能决策", "银行运营监控", 6, "页面说明基于 AI 计算机视觉进行视频异常监控。"),
    ("002987.SZ", "事后监督系统", "视觉与智能决策", "银行运营监督", 6, "页面说明采用 OCR 和 AI 智能化手段识别凭证与记录。"),
    ("002987.SZ", "运营风险监控系统", "金融业务场景", "银行运营风险", 6, "页面说明使用大数据风险模型和机器学习。"),
    ("002987.SZ", "资产管理系统", "金融业务场景", "资产管理", 8, "页面说明融入人工智能和大数据，并使用 AI 模型。"),
    ("002987.SZ", "理财销售系统", "金融业务场景", "理财销售", 8, "页面说明使用 OCR、NLP 等技术。"),
    ("002987.SZ", "基金代销系统", "金融业务场景", "基金代销", 8, "页面说明提供智能推荐。"),
    ("002987.SZ", "供应链金融服务平台", "金融业务场景", "供应链金融与信贷", 9, "页面说明采用人工智能并支持合同审查、OCR 等。"),
    ("002987.SZ", "反洗钱监测分析系统", "金融业务场景", "反洗钱与合规", 10, "页面说明采用人工智能和大语言模型。"),
    ("603927.SH", "人工智能平台", "人工智能基础", "财产保险产品支撑", 1, "财产保险产品体系中列示人工智能平台。"),
]


STAGE1_CATEGORY = {
    "600570.SH": "大模型与语言技术", "300033.SZ": "大模型与语言技术", "300059.SZ": "大模型与语言技术",
    "600446.SH": "大模型与语言技术", "300377.SZ": "人工智能基础", "300674.SZ": "人工智能基础",
    "000555.SZ": "人工智能基础", "300348.SZ": "人工智能基础", "300872.SZ": "大模型与语言技术",
    "300663.SZ": "人工智能基础", "002987.SZ": "人工智能基础", "603927.SH": "大模型与语言技术",
    "688590.SH": "大模型与语言技术", "002230.SZ": "大模型与语言技术", "300229.SZ": "大模型与语言技术",
}


def main() -> int:
    screenshots = pd.read_csv(repo_path("data", "processed", "product_page_screenshots.csv"), dtype=str, keep_default_na=False)
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    review = pd.DataFrame(PAGE_REVIEWS)
    review["screenshot_page"] = review["screenshot_page"].astype(str)
    expected = screenshots[["company_code", "screenshot_page"]]
    actual = review[["company_code", "screenshot_page"]]
    if len(review) != len(screenshots) or set(map(tuple, actual.to_numpy())) != set(map(tuple, expected.to_numpy())):
        raise ValueError("PAGE_REVIEWS must contain exactly one review for every supplied screenshot.")

    page_mapping = screenshots.merge(review, on=["company_code", "screenshot_page"], how="left", validate="one_to_one")
    if "company_name" not in page_mapping.columns:
        page_mapping = page_mapping.merge(pool[["company_code", "company_name"]], on="company_code", how="left", validate="many_to_one")
    page_mapping = page_mapping[[
        "company_code", "company_name", "screenshot_page", "source_filename", "local_path", "sha256",
        "page_title", "identified_product_entities", "explicit_ai_terms", "review_outcome", "review_note",
    ]].sort_values(["company_code", "screenshot_page"])

    detail_columns = ["company_code", "product_line", "technical_category", "application_scenario", "screenshot_page", "evidence_text"]
    entities = pd.DataFrame(ENTITY_DETAILS, columns=detail_columns)
    entities = entities.merge(pool[["company_code", "company_name"]], on="company_code", how="left", validate="many_to_one")
    lookup = page_mapping.set_index(["company_code", "screenshot_page"])
    entities["source_filename"] = [lookup.loc[(row.company_code, str(row.screenshot_page)), "source_filename"] for row in entities.itertuples()]
    entities["source_path"] = [lookup.loc[(row.company_code, str(row.screenshot_page)), "local_path"] for row in entities.itertuples()]
    entities["entity_source"] = "official_product_page_screenshot"
    entities["verification_status"] = "page_verified_explicit_ai_financial_scope"
    entities = entities[[
        "company_code", "company_name", "product_line", "technical_category", "application_scenario",
        "screenshot_page", "source_filename", "source_path", "evidence_text", "entity_source", "verification_status",
    ]].sort_values(["company_code", "screenshot_page", "product_line"])

    stage_categories = pd.DataFrame(STAGE1_CATEGORY.items(), columns=["company_code", "technical_category"])
    output_dir = repo_path("data", "processed")
    ensure_directories([output_dir])
    page_mapping.to_csv(output_dir / "screenshot_page_mapping.csv", index=False, encoding="utf-8")
    entities.to_csv(output_dir / "screenshot_ai_product_entities.csv", index=False, encoding="utf-8")
    stage_categories.to_csv(output_dir / "stage1_product_line_technical_categories.csv", index=False, encoding="utf-8")

    outcomes = Counter(page_mapping["review_outcome"])
    print(f"Mapped {len(page_mapping)} supplied screenshot page(s): {dict(sorted(outcomes.items()))}")
    print(f"Wrote {len(entities)} page-verified financial-AI product entities to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
