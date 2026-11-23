import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from casa_norte.model import JointBert, align_slot_labels, plot_attention_map, predict, save_bundle
from casa_norte.nlu import (
    CLASSES, ID2INTENT, INTENT2ID, SLOT_LABELS,
    deduplicate_splits, find_span, load_jsonl, load_mexican, load_split, span_prf, spans_from_tags,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def featurize(rows, tokenizer, max_len, with_word_ids=True):
    examples = []
    for row in rows:
        input_ids, attention_mask, slot_labels, word_ids = align_slot_labels(
            row["tokens"], row["tags"], tokenizer, max_len
        )
        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slot_labels": slot_labels,
            "intent_label": INTENT2ID[row["intent"]],
            "tokens": row["tokens"],
            "tags": row["tags"],
            "intent": row["intent"],
        }
        if with_word_ids:
            example["word_ids"] = word_ids
        examples.append(example)
    return examples


def make_batch(examples, indices, device):
    batch = [examples[index] for index in indices]
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long, device=device)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long, device=device)
    slot_labels = torch.tensor([item["slot_labels"] for item in batch], dtype=torch.long, device=device)
    intent_labels = torch.tensor([item["intent_label"] for item in batch], dtype=torch.long, device=device)
    return input_ids, attention_mask, slot_labels, intent_labels


def decode_word_tags(slot_ids, word_ids, n_words):
    tags = ["O"] * n_words
    seen = set()
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        if word_id < n_words:
            tags[word_id] = SLOT_LABELS[slot_ids[position]]
    return tags


@torch.no_grad()
def evaluate(examples, model, device, batch_size=32):
    model.eval()
    gold_intents, pred_intents, confidences = [], [], []
    gold_spans_all, pred_spans_all = [], []
    per_example = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]
        input_ids, attention_mask, _, intent_labels = make_batch(
            examples, list(range(start, start + len(batch_examples))), device
        )
        intent_logits, slot_logits = model(input_ids, attention_mask)
        probs = torch.softmax(intent_logits, dim=-1).cpu()
        preds = probs.argmax(dim=-1).tolist()
        slot_ids = slot_logits.argmax(dim=-1).cpu().tolist()
        for offset, example in enumerate(batch_examples):
            gold_intent = example["intent"]
            pred_intent = ID2INTENT[preds[offset]]
            confidence = float(probs[offset, preds[offset]])
            word_tags = decode_word_tags(slot_ids[offset], example["word_ids"], len(example["tokens"]))
            gold_spans = spans_from_tags(example["tags"])
            pred_spans = spans_from_tags(word_tags)
            gold_intents.append(gold_intent)
            pred_intents.append(pred_intent)
            confidences.append(confidence)
            gold_spans_all.extend((gold_intent, *span) for span in gold_spans)
            pred_spans_all.extend((pred_intent, *span) for span in pred_spans)
            per_example.append({
                "text": " ".join(example["tokens"]),
                "gold_intent": gold_intent,
                "pred_intent": pred_intent,
                "confidence": confidence,
                "gold_spans": sorted(gold_spans),
                "pred_spans": sorted(pred_spans),
            })
    metrics = {
        "accuracy": float(accuracy_score(gold_intents, pred_intents)),
        "macro_f1": float(f1_score(gold_intents, pred_intents, average="macro", zero_division=0)),
        "slot_span_f1": span_prf(set(gold_spans_all), set(pred_spans_all))["f1"],
        "slot_precision": span_prf(set(gold_spans_all), set(pred_spans_all))["precision"],
        "slot_recall": span_prf(set(gold_spans_all), set(pred_spans_all))["recall"],
        "report": classification_report(
            gold_intents, pred_intents, labels=CLASSES, zero_division=0, output_dict=True
        ),
        "gold_intents": gold_intents,
        "pred_intents": pred_intents,
        "confidences": confidences,
    }
    return metrics, per_example


def plot_confusion(metrics, path):
    labels = CLASSES
    matrix = confusion_matrix(metrics["gold_intents"], metrics["pred_intents"], labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión · test (Transformer)")
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_class(metrics, path):
    labels = [name for name in CLASSES if name in metrics["report"]]
    scores = [metrics["report"][name]["f1-score"] for name in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, scores, color="#2a6f97")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1 por intención · test")
    ax.tick_params(axis="x", rotation=25)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history, path):
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", color="#c1121f")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida de entrenamiento")
    axes[0].set_title("Pérdida")
    axes[1].plot(epochs, [item["val_macro_f1"] for item in history], marker="o", label="macro-F1", color="#2a6f97")
    axes[1].plot(epochs, [item["val_slot_f1"] for item in history], marker="s", label="span-F1 slots", color="#f4a261")
    axes[1].set_xlabel("Época")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Validación")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mexican_spans(row):
    words = row["utt"].split()
    spans = set()
    for slot in row.get("slots", []):
        found = find_span(words, slot["value"])
        if found:
            spans.add((slot["type"], found[0], found[1]))
    return words, spans


def evaluate_mexican(rows, model, tokenizer, max_len, threshold, device):
    in_domain_gold, in_domain_pred = [], []
    gold_spans_all, pred_spans_all = set(), set()
    out_of_scope, rejected = 0, 0
    details = []
    for row in rows:
        words = row["utt"].split()
        result = predict(model, tokenizer, row["utt"], max_len=max_len)
        _, gold_spans = mexican_spans(row)
        pred_spans = {(span["type"], span["start"], span["end"]) for span in result["spans"]}
        is_ood = row["intent"] == "out_of_scope"
        if is_ood:
            out_of_scope += 1
            rejected += int(result["confidence"] < threshold)
        else:
            in_domain_gold.append(row["intent"])
            in_domain_pred.append(result["intent"])
            gold_spans_all |= gold_spans
            pred_spans_all |= pred_spans
        details.append({
            "id": row["id"],
            "text": row["utt"],
            "gold_intent": row["intent"],
            "pred_intent": result["intent"],
            "confidence": result["confidence"],
            "gold_spans": sorted(gold_spans),
            "pred_spans": sorted(pred_spans),
            "out_of_scope": is_ood,
        })
    span = span_prf(gold_spans_all, pred_spans_all)
    return {
        "n": len(rows),
        "n_in_domain": len(in_domain_gold),
        "n_out_of_scope": out_of_scope,
        "accuracy": float(accuracy_score(in_domain_gold, in_domain_pred)),
        "macro_f1": float(f1_score(in_domain_gold, in_domain_pred, average="macro", zero_division=0)),
        "slot_span_f1": span["f1"],
        "slot_precision": span["precision"],
        "slot_recall": span["recall"],
        "oos_rejected": rejected,
        "oos_rejection_rate": rejected / out_of_scope if out_of_scope else 0.0,
        "threshold": threshold,
        "details": details,
    }


def measure_model_latency(model, tokenizer, texts, max_len, device, limit=120):
    durations = []
    for text in texts[:limit]:
        started = time.perf_counter()
        predict(model, tokenizer, text, max_len=max_len)
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": float(np.mean(durations)),
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "samples": len(durations),
    }


def threshold_sweep(details, step=0.05):
    sweep = []
    for index in range(4, 19):
        threshold = round(index * step, 2)
        accepted = sum(
            1 for item in details
            if not item["out_of_scope"] and item["confidence"] >= threshold
        )
        rejected = sum(
            1 for item in details
            if item["out_of_scope"] and item["confidence"] < threshold
        )
        sweep.append({
            "threshold": threshold,
            "in_domain_accepted": accepted,
            "oos_rejected": rejected,
        })
    return sweep


def load_baseline_metrics(path):
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--mexican", default=str(ROOT / "data" / "mexican_test.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "modelo_casa_norte"))
    parser.add_argument("--figs", default=str(ROOT / "notebooks" / "figs"))
    parser.add_argument("--metrics", default=str(ROOT / "notebooks" / "metrics_fase2.json"))
    parser.add_argument("--errors", default=str(ROOT / "notebooks" / "error_cases.json"))
    parser.add_argument("--synthetic", default=str(ROOT / "data" / "synthetic_norte_train.jsonl"))
    parser.add_argument("--baseline-metrics", default=str(ROOT / "notebooks" / "metrics_baseline_prod.json"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)

    train_base = load_split(args.data_dir, "train", set(CLASSES))
    val = load_split(args.data_dir, "validation", set(CLASSES))
    test = load_split(args.data_dir, "test", set(CLASSES))
    train_base, val, test = deduplicate_splits(train_base, val, test)
    synthetic = []
    if Path(args.synthetic).exists():
        synthetic = load_jsonl(args.synthetic, set(CLASSES))
    held_texts = {row["text"] for row in val} | {row["text"] for row in test}
    synthetic = [row for row in synthetic if row["text"] not in held_texts]
    train = train_base + synthetic
    if args.limit:
        per_class = max(1, args.limit // len(CLASSES))
        by_class = {name: [row for row in train_base if row["intent"] == name] for name in CLASSES}
        train = []
        for name in CLASSES:
            train.extend(random.sample(by_class[name], min(per_class, len(by_class[name]))))
        random.shuffle(train)
    print(
        f"train={len(train_base)} synthetic={len(synthetic)} val={len(val)} test={len(test)} device={device}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_examples = featurize(train, tokenizer, args.max_len, with_word_ids=False)
    val_examples = featurize(val, tokenizer, args.max_len)
    test_examples = featurize(test, tokenizer, args.max_len)
    print("tokenización lista", flush=True)

    model = JointBert(args.model, len(CLASSES), len(SLOT_LABELS), dropout=args.dropout).to(device)
    counts = np.array([sum(1 for row in train if row["intent"] == name) for name in CLASSES], dtype=float)
    weights = len(train) / (len(CLASSES) * np.maximum(counts, 1))
    weights = torch.tensor(np.clip(weights, 0.5, 5.0), dtype=torch.float, device=device)
    intent_loss_fn = nn.CrossEntropyLoss(weight=weights)
    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_examples) + args.batch - 1) // args.batch) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup), total_steps
    )

    best_state, best_score, best_epoch = None, -1.0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(train_examples)))
        random.shuffle(indices)
        running_loss = 0.0
        steps = 0
        for start in range(0, len(indices), args.batch):
            batch_indices = indices[start:start + args.batch]
            input_ids, attention_mask, slot_labels, intent_labels = make_batch(
                train_examples, batch_indices, device
            )
            optimizer.zero_grad()
            intent_logits, slot_logits = model(input_ids, attention_mask)
            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            slot_loss = slot_loss_fn(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
            loss = intent_loss + slot_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            steps += 1
        val_metrics, _ = evaluate(val_examples, model, device, batch_size=args.batch)
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, steps),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_slot_f1": val_metrics["slot_span_f1"],
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - started

    test_metrics, test_examples_out = evaluate(test_examples, model, device, batch_size=args.batch)
    mexican_rows = load_mexican(args.mexican)
    mexican_metrics = evaluate_mexican(mexican_rows, model, tokenizer, args.max_len, args.threshold, device)
    latency = measure_model_latency(
        model, tokenizer, [row["text"] for row in test], args.max_len, device
    )
    attention_example = "cómo va a estar el calor hoy en Monterrey"
    attention_path, _, _ = plot_attention_map(
        attention_example, model, tokenizer, args.max_len, figs / "attention_map.png"
    )
    sweep = threshold_sweep(mexican_metrics["details"])
    baseline_metrics = load_baseline_metrics(args.baseline_metrics)

    errors = []
    for item in test_examples_out:
        if item["gold_intent"] != item["pred_intent"] or item["gold_spans"] != item["pred_spans"]:
            error_type = "intent" if item["gold_intent"] != item["pred_intent"] else "slots"
            errors.append({
                "type": error_type,
                "text": item["text"],
                "gold_intent": item["gold_intent"],
                "pred_intent": item["pred_intent"],
                "confidence": round(item["confidence"], 4),
                "gold_spans": item["gold_spans"],
                "pred_spans": item["pred_spans"],
                "cause": (
                    f"Confundió {item['gold_intent']} con {item['pred_intent']}; comparten vocabulario de acción."
                    if error_type == "intent" else
                    "Los límites o el tipo de algún slot no coinciden con la anotación."
                ),
            })
    errors = sorted(errors, key=lambda item: (item["type"] != "intent", -item["confidence"]))[:8]

    plot_confusion(test_metrics, figs / "confusion_matrix.png")
    plot_per_class(test_metrics, figs / "per_class_f1.png")
    plot_history(history, figs / "training_curves.png")

    config = {
        "model_name": args.model,
        "n_intents": len(CLASSES),
        "n_slots": len(SLOT_LABELS),
        "dropout": args.dropout,
        "max_len": args.max_len,
        "classes": CLASSES,
        "slot_labels": SLOT_LABELS,
        "threshold": args.threshold,
        "output_attentions": True,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch,
        "weight_decay": args.weight_decay,
        "train_size": len(train_base),
        "synthetic_size": len(synthetic),
        "val_size": len(val),
        "test_size": len(test),
    }
    save_bundle(model, tokenizer, args.out, config)

    summary = {
        "model": args.model,
        "device": str(device),
        "config": {
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "max_len": args.max_len,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "warmup": args.warmup,
            "threshold": args.threshold,
            "seed": args.seed,
            "class_weights": {name: round(float(weight), 3) for name, weight in zip(CLASSES, weights.cpu().tolist())},
        },
        "sizes": {
            "train": len(train_base),
            "synthetic": len(synthetic),
            "validation": len(val),
            "test": len(test),
        },
        "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 1),
        "history": history,
        "latency": latency,
        "attention_map": str(attention_path),
        "threshold_sweep": sweep,
        "baseline": baseline_metrics,
        "test": {key: value for key, value in test_metrics.items() if key not in {"gold_intents", "pred_intents", "confidences"}},
        "mexican": {key: value for key, value in mexican_metrics.items() if key != "details"},
        "mexican_details": mexican_metrics["details"],
    }
    with Path(args.metrics).open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, ensure_ascii=False, indent=2)
    with Path(args.errors).open("w", encoding="utf-8") as handle:
        json.dump(jsonable(errors), handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_slot_f1": test_metrics["slot_span_f1"],
        "mexican_accuracy": mexican_metrics["accuracy"],
        "mexican_slot_f1": mexican_metrics["slot_span_f1"],
        "mexican_oos_rejection": mexican_metrics["oos_rejection_rate"],
        "transformer_mean_ms": latency["mean_ms"],
        "attention_map": str(attention_path),
        "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 1),
    }, ensure_ascii=False, indent=2), flush=True)
    for detail in mexican_metrics["details"]:
        print(detail["text"], "->", detail["pred_intent"], round(detail["confidence"], 3), detail["pred_spans"], flush=True)


if __name__ == "__main__":
    main()
