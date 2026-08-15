"""Fine-tune a small Chinese text encoder on observed recruitment labels.

This is an actual supervised fine-tuning run of the locally cached
``BAAI/bge-small-zh-v1.5`` encoder.  It predicts whether the user-provided
AI-technology-stack field explicitly contains an AI-related reference.  The
input deliberately excludes that target field to avoid label leakage.

The model is an encoder, not a generative LLM.  Results must therefore be
reported as small-domain text-encoder fine-tuning, not as generative-model
instruction tuning.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from common import ensure_directories, repo_path, write_json


EMPTY_MARKERS = {"", "无", "无明确要求", "无相关要求", "nan", "暂无"}
AI_TERMS = ("ai", "人工智能", "大模型", "模型", "智能体", "agent", "机器学习", "深度学习", "nlp", "自然语言", "知识图谱", "计算机视觉", "算法", "风控")
LABEL_DEFINITION = (
    "用户提供的 AI 技术栈字段不为空值标记，且包含 AI/人工智能/大模型/模型/智能体/"
    "agent/机器学习/深度学习/NLP/自然语言/知识图谱/计算机视觉/算法/风控中的至少一项。"
)


def observed_ai_reference(value: object) -> int:
    text = str(value or "").strip()
    lowered = text.casefold()
    return int(text.casefold() not in EMPTY_MARKERS and any(term in lowered for term in AI_TERMS))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class JobTextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tokenizer: Any, texts: list[str], labels: list[int], max_length: int) -> None:
        self.encoding = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encoding.items()}
        item["labels"] = self.labels[index]
        return item


class EncoderClassifier(nn.Module):
    def __init__(self, model_name: str, local_files_only: bool, unfrozen_layers: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        if not hasattr(self.encoder, "encoder") or not hasattr(self.encoder.encoder, "layer"):
            raise TypeError("This script currently supports BERT-style encoders with encoder.layer.")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = list(self.encoder.encoder.layer)
        if not 1 <= unfrozen_layers <= len(layers):
            raise ValueError(f"unfrozen_layers must be between 1 and {len(layers)}")
        for layer in layers[-unfrozen_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        if getattr(self.encoder, "pooler", None) is not None:
            for parameter in self.encoder.pooler.parameters():
                parameter.requires_grad = True
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return self.classifier(pooled).squeeze(-1)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Metrics:
    predictions = (probabilities >= 0.5).astype(int)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(float((predictions == labels).mean()), precision, recall, f1)


def predict(model: EncoderClassifier, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            logits = model(**{key: value.to(device) for key, value in batch.items()})
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probabilities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a local BERT-style encoder on recruitment text.")
    parser.add_argument("--input", type=Path, default=repo_path("data", "processed", "jobs_clean.csv"))
    parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--allow-download", action="store_true", help="Permit Hugging Face to download the base model when it is not cached.")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--unfrozen-layers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1, help="PyTorch CPU threads; one is the most portable default for this small run.")
    parser.add_argument("--output-dir", type=Path, default=repo_path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Required recruitment input is missing: {args.input}")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.num_threads < 1:
        raise ValueError("num-threads must be positive")
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    jobs = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    required = {"job_id", "company_name", "job_title", "skills_raw", "ai_tech_stack_raw"}
    missing = required - set(jobs.columns)
    if missing:
        raise ValueError(f"jobs input missing columns: {sorted(missing)}")
    data = jobs.loc[:, ["job_id", "company_name", "job_title", "skills_raw", "ai_tech_stack_raw"]].copy()
    data["text"] = (data["job_title"].str.strip() + " "+ data["skills_raw"].str.strip()).str.strip()
    data["label"] = data["ai_tech_stack_raw"].map(observed_ai_reference)
    train, holdout = train_test_split(data, test_size=0.2, random_state=args.seed, stratify=data["label"])
    train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)
    print(f"Loaded {len(data)} records: train={len(train)}, holdout={len(holdout)}; positive-label share={data['label'].mean():.3f}.", flush=True)

    local_files_only = not args.allow_download
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=local_files_only)
        model = EncoderClassifier(args.model_name, local_files_only=local_files_only, unfrozen_layers=args.unfrozen_layers)
    except OSError as exc:
        if local_files_only:
            raise FileNotFoundError(
                f"Base model {args.model_name!r} is not available in the local Hugging Face cache. "
                "Run with --allow-download only when network access is permitted."
            ) from exc
        raise

    device = torch.device("cpu")
    model.to(device)
    train_loader = DataLoader(JobTextDataset(tokenizer, train["text"].tolist(), train["label"].tolist(), args.max_length), batch_size=args.batch_size, shuffle=True)
    holdout_loader = DataLoader(JobTextDataset(tokenizer, holdout["text"].tolist(), holdout["label"].tolist(), args.max_length), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: Metrics | None = None
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad()
            logits = model(**{key: value.to(device) for key, value in batch.items()})
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
        labels, probabilities = predict(model, holdout_loader, device)
        metrics = compute_metrics(labels, probabilities)
        history.append({"epoch": epoch, "train_loss": running_loss / len(train), **metrics.__dict__})
        print(f"Epoch {epoch}/{args.epochs}: train_loss={running_loss / len(train):.4f}, holdout_f1={metrics.f1:.4f}", flush=True)
        if best_metrics is None or metrics.f1 > best_metrics.f1:
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name in {n for n, p in model.named_parameters() if p.requires_grad}}
            best_metrics = metrics
            best_epoch = epoch
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state, strict=False)

    full_loader = DataLoader(JobTextDataset(tokenizer, data["text"].tolist(), data["label"].tolist(), args.max_length), batch_size=args.batch_size, shuffle=False)
    _, full_probabilities = predict(model, full_loader, device)
    data["evaluation_split"] = np.where(data["job_id"].isin(set(holdout["job_id"])), "holdout", "train")
    data["predicted_ai_reference_probability"] = np.round(full_probabilities, 6)
    data["predicted_label_at_0_5"] = (full_probabilities >= 0.5).astype(int)
    data["label_definition"] = LABEL_DEFINITION
    tables = args.output_dir / "tables"
    models = args.output_dir / "models"
    ensure_directories([tables, models])
    data.drop(columns=["text", "ai_tech_stack_raw"]).to_csv(tables / "domain_encoder_predictions.csv", index=False, encoding="utf-8")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(tables / "domain_encoder_training_history.csv", index=False, encoding="utf-8")
    metrics_frame = pd.DataFrame([{
        "evaluation_split": "holdout",
        "holdout_n": len(holdout),
        "train_n": len(train),
        "positive_label_share_holdout": round(float(holdout["label"].mean()), 4),
        "accuracy": round(best_metrics.accuracy, 4),
        "precision": round(best_metrics.precision, 4),
        "recall": round(best_metrics.recall, 4),
        "f1": round(best_metrics.f1, 4),
        "best_epoch": best_epoch,
        "model_name": args.model_name,
        "trainable_layers": args.unfrozen_layers,
        "label_definition": LABEL_DEFINITION,
        "method_boundary": "真实监督微调的文本编码器；不是生成式大语言模型，且标签来自用户提供的 AI 技术栈字段。",
    }])
    metrics_frame.to_csv(tables / "domain_encoder_holdout_metrics.csv", index=False, encoding="utf-8")
    checkpoint_path = models / "domain_encoder_bge_small_trainable_weights.pt"
    torch.save({
        "base_model": args.model_name,
        "base_model_local_files_only": local_files_only,
        "max_length": args.max_length,
        "unfrozen_layers": args.unfrozen_layers,
        "label_definition": LABEL_DEFINITION,
        "best_epoch": best_epoch,
        "holdout_metrics": best_metrics.__dict__,
        "trainable_state_dict": best_state,
    }, checkpoint_path)
    write_json(models / "domain_encoder_training_metadata.json", {
        "input": str(args.input), "model_name": args.model_name, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length, "learning_rate": args.learning_rate,
        "unfrozen_layers": args.unfrozen_layers, "num_threads": args.num_threads, "train_n": len(train), "holdout_n": len(holdout),
        "label_definition": LABEL_DEFINITION, "method_boundary": metrics_frame.loc[0, "method_boundary"],
    })
    print(f"Fine-tuned {args.model_name} on {len(train)} training record(s); holdout n={len(holdout)}.")
    print(f"Best holdout accuracy={best_metrics.accuracy:.4f}, F1={best_metrics.f1:.4f} at epoch {best_epoch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
