"""Identify AI-fintech themes from processed report sections.

Use --method sentence-transformer in the final report to meet the pretrained
embedding requirement. The TF-IDF mode exists only for offline pipeline checks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from common import ensure_directories, repo_path, write_json


def load_inputs(input_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sections_path = repo_path(*input_path.split("/")) if input_path else repo_path("data", "processed", "annual_report_sections.csv")
    lexicon_path = repo_path("data", "reference", "ai_fintech_keywords.csv")
    sections = pd.read_csv(sections_path, dtype=str, keep_default_na=False) if sections_path.exists() else pd.DataFrame(columns=["company_code", "source_id", "section", "text"])
    keywords = pd.read_csv(lexicon_path, dtype=str, keep_default_na=False) if lexicon_path.exists() else pd.DataFrame(columns=["keyword", "category"])
    return sections[sections.get("text", pd.Series(dtype=str)).astype(str).str.len() > 0].copy(), keywords


def tfidf_scores(texts: list[str], terms: list[str]):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not texts or not terms:
        return np.empty((len(texts), len(terms)))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts + terms)
    return cosine_similarity(matrix[: len(texts)], matrix[len(texts) :])


def sentence_transformer_scores(texts: list[str], terms: list[str], model_name: str, max_characters: int):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    clipped_texts = [text[:max_characters] for text in texts]
    vectors = model.encode(clipped_texts + terms, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors[: len(texts)]) @ np.asarray(vectors[len(texts) :]).T


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["tfidf", "sentence-transformer"], default="tfidf")
    parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--input-path", help="Optional repository-relative input CSV.")
    parser.add_argument("--max-characters", type=int, default=1200, help="Maximum leading characters per extracted section for transformer encoding.")
    args = parser.parse_args()
    sections, lexicon = load_inputs(args.input_path)
    output = repo_path("outputs", "tables", "semantic_keyword_scores.csv")
    metadata = repo_path("outputs", "tables", "semantic_run_metadata.json")
    ensure_directories([output.parent])
    columns = ["company_code", "source_id", "section", "keyword", "category", "similarity", "rank"]
    if sections.empty or lexicon.empty:
        pd.DataFrame(columns=columns).to_csv(output, index=False, encoding="utf-8")
        write_json(metadata, {"method": args.method, "status": "no_input_text_or_lexicon", "run_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"No report sections available; wrote empty result schema: {output}")
        return 0
    texts = sections["text"].tolist()
    terms = lexicon["keyword"].tolist()
    scores = sentence_transformer_scores(texts, terms, args.model_name, args.max_characters) if args.method == "sentence-transformer" else tfidf_scores(texts, terms)
    rows = []
    for section_index, section_row in sections.reset_index(drop=True).iterrows():
        order = np.argsort(scores[section_index])[::-1][: args.top_k]
        for rank, keyword_index in enumerate(order, start=1):
            keyword_row = lexicon.iloc[int(keyword_index)]
            rows.append({"company_code": section_row["company_code"], "source_id": section_row["source_id"], "section": section_row["section"], "keyword": keyword_row["keyword"], "category": keyword_row["category"], "similarity": round(float(scores[section_index, keyword_index]), 6), "rank": rank})
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False, encoding="utf-8")
    write_json(metadata, {"method": args.method, "model_name": args.model_name if args.method == "sentence-transformer" else None, "max_characters_per_section": args.max_characters if args.method == "sentence-transformer" else None, "input_path": args.input_path or "data/processed/annual_report_sections.csv", "status": "completed", "section_count": len(sections), "company_count": int(sections["company_code"].nunique()), "keyword_count": len(lexicon), "run_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Wrote {len(rows)} semantic matches to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
