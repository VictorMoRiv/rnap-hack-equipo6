import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casa_norte.nlu import (  # noqa: E402
    CLASSES, find_span, load_mexican, load_split, span_prf, spans_from_tags,
)

SEED = 42


def as32(matrix):
    matrix = matrix.tocsr()
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix


def fit_intent(rows, c_value):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000
    )
    texts = [row["raw"] for row in rows]
    matrix = as32(hstack([word.fit_transform(texts), char.fit_transform(texts)]))
    clf = OneVsRestClassifier(
        LogisticRegression(C=c_value, max_iter=1000, solver="liblinear", random_state=SEED)
    ).fit(matrix, [row["intent"] for row in rows])
    return word, char, clf


def predict_intent(bundle, texts):
    word, char, clf = bundle
    matrix = as32(hstack([word.transform(texts), char.transform(texts)]))
    return clf.predict(matrix), clf.predict_proba(matrix)


def token_features(tokens, index):
    token = tokens[index].lower()
    return {
        "w": token,
        "p3": token[:3],
        "s3": token[-3:],
        "dig": token.isdigit(),
        "first": index == 0,
        "pw": tokens[index - 1].lower() if index else "<s>",
        "pw2": tokens[index - 2].lower() if index > 1 else "<s>",
        "nw": tokens[index + 1].lower() if index < len(tokens) - 1 else "</s>",
    }


def fit_slots(rows):
    features, labels = [], []
    for row in rows:
        features += [token_features(row["tokens"], index) for index in range(len(row["tokens"]))]
        labels += row["tags"]
    vectorizer = DictVectorizer()
    clf = LogisticRegression(C=10, max_iter=300, solver="saga", random_state=SEED)
    clf.fit(as32(vectorizer.fit_transform(features)), labels)
    return vectorizer, clf


def predict_slots(bundle, text):
    vectorizer, clf = bundle
    tokens = text.split()
    features = [token_features(tokens, index) for index in range(len(tokens))]
    return tokens, list(clf.predict(as32(vectorizer.transform(features))))


def mexican_gold(row):
    words = row["utt"].split()
    spans = set()
    for slot in row.get("slots", []):
        found = find_span(words, slot["value"])
        if found:
            spans.add((slot["type"], found[0], found[1]))
    return words, spans


def main():
    train = load_split(ROOT / "data", "train", set(CLASSES))
    test = load_split(ROOT / "data", "test", set(CLASSES))
    mexican = load_mexican(ROOT / "data" / "mexican_test.jsonl")

    intent_bundle = fit_intent(train, 30)
    predicted, probabilities = predict_intent(intent_bundle, [row["raw"] for row in test])
    gold = [row["intent"] for row in test]
    test_metrics = {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "report": classification_report(gold, predicted, labels=CLASSES, zero_division=0, output_dict=True),
    }

    slot_bundle = fit_slots(train)
    gold_spans_all, pred_spans_all = set(), set()
    for row in test:
        tokens, tags = predict_slots(slot_bundle, " ".join(row["tokens"]))
        gold_spans_all |= spans_from_tags(row["tags"])
        pred_spans_all |= spans_from_tags(tags)
    span = span_prf(gold_spans_all, pred_spans_all)
    test_metrics.update({
        "slot_span_f1": span["f1"],
        "slot_precision": span["precision"],
        "slot_recall": span["recall"],
    })

    in_gold, in_pred = [], []
    mx_gold, mx_pred = set(), set()
    out_of_scope = rejected = 0
    details = []
    for row in mexican:
        words = row["utt"].split()
        predicted_intent, probs = predict_intent(intent_bundle, [row["utt"]])
        confidence = float(np.max(probs))
        _, gold_spans = mexican_gold(row)
        _, tags = predict_slots(slot_bundle, row["utt"])
        pred_spans = spans_from_tags(tags)
        if row["intent"] == "out_of_scope":
            out_of_scope += 1
            rejected += int(confidence < 0.5)
        else:
            in_gold.append(row["intent"])
            in_pred.append(predicted_intent[0])
            mx_gold |= gold_spans
            mx_pred |= pred_spans
        details.append({
            "id": row["id"],
            "text": row["utt"],
            "gold_intent": row["intent"],
            "pred_intent": predicted_intent[0],
            "confidence": confidence,
            "gold_spans": sorted(gold_spans),
            "pred_spans": sorted(pred_spans),
            "out_of_scope": row["intent"] == "out_of_scope",
        })
    mx_span = span_prf(mx_gold, mx_pred)
    mexican_metrics = {
        "accuracy": float(accuracy_score(in_gold, in_pred)),
        "macro_f1": float(f1_score(in_gold, in_pred, average="macro", zero_division=0)),
        "slot_span_f1": mx_span["f1"],
        "oos_rejected": rejected,
        "oos_rejection_rate": rejected / out_of_scope if out_of_scope else 0.0,
        "threshold": 0.5,
        "details": details,
    }

    summary = {
        "model": "TF-IDF + LogReg (baseline Fase 1)",
        "sizes": {"train": len(train), "test": len(test)},
        "test": test_metrics,
        "mexican": mexican_metrics,
    }
    out = ROOT / "notebooks" / "metrics_baseline.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=float)
    print(json.dumps({
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_slot_f1": test_metrics["slot_span_f1"],
        "mexican_accuracy": mexican_metrics["accuracy"],
        "mexican_slot_f1": mexican_metrics["slot_span_f1"],
        "mexican_oos_rejection": mexican_metrics["oos_rejection_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
