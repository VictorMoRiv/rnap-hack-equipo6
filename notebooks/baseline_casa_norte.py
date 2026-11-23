import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casa_norte.nlu import CLASSES, deduplicate_splits, load_jsonl, load_split


def build_model(texts, labels, C):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=C, max_iter=2000, random_state=42)
    classifier.fit(matrix, labels)
    return vectorizer, classifier


def predict_bundle(bundle, texts):
    vectorizer, classifier = bundle
    return classifier.predict(vectorizer.transform(texts))


def measure_latency(bundle, texts):
    durations = []
    for text in texts:
        started = time.perf_counter()
        predict_bundle(bundle, [text])
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": float(np.mean(durations)),
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--synthetic", default=str(ROOT / "data" / "synthetic_norte_train.jsonl"))
    parser.add_argument("--metrics", default=str(ROOT / "notebooks" / "metrics_baseline_prod.json"))
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()

    train = load_split(args.data_dir, "train", set(CLASSES))
    val = load_split(args.data_dir, "validation", set(CLASSES))
    test = load_split(args.data_dir, "test", set(CLASSES))
    train, val, test = deduplicate_splits(train, val, test)
    synthetic = []
    if not args.no_synthetic and Path(args.synthetic).exists():
        synthetic = load_jsonl(args.synthetic, set(CLASSES))
    held_texts = {row["text"] for row in val} | {row["text"] for row in test}
    synthetic = [row for row in synthetic if row["text"] not in held_texts]
    train_all = train + synthetic
    print(f"train={len(train)} synthetic={len(synthetic)} val={len(val)} test={len(test)}", flush=True)

    best = None
    for C in [0.1, 1.0, 10.0, 100.0]:
        bundle = build_model([row["text"] for row in train_all], [row["intent"] for row in train_all], C)
        preds = predict_bundle(bundle, [row["text"] for row in val])
        score = f1_score([row["intent"] for row in val], preds, average="macro")
        print(f"C={C} val_macro_f1={score:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, C, bundle)
    best_f1, best_C, bundle = best

    preds = predict_bundle(bundle, [row["text"] for row in test])
    gold = [row["intent"] for row in test]
    latency = measure_latency(bundle, [row["text"] for row in test])
    summary = {
        "vectorizer": {"ngram_range": [1, 2], "max_features": 5000},
        "classifier": "LogisticRegression",
        "selected_C": best_C,
        "val_macro_f1": float(best_f1),
        "sizes": {"train": len(train), "synthetic": len(synthetic), "validation": len(val), "test": len(test)},
        "test_accuracy": float(accuracy_score(gold, preds)),
        "test_macro_f1": float(f1_score(gold, preds, average="macro")),
        "latency": latency,
        "leakage_check": {
            "official_partitions_by_id_kept": True,
            "duplicate_surface_forms_removed_from_train_and_validation": True,
            "vectorizer_fit_on": "deduplicated train+synthetic only",
            "classifier_fit_on": "deduplicated train+synthetic only",
            "C_selected_on": "deduplicated validation only",
            "test_used_only_for_final_metrics": True,
        },
    }
    Path(args.metrics).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
