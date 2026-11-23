import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .nlu import CLASSES, ID2INTENT, ID2SLOT, SLOT2ID, spans_from_tags


class JointBert(nn.Module):
    def __init__(self, model_name, n_intents, n_slots, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, n_intents)
        self.slot_head = nn.Linear(hidden, n_slots)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = self.dropout(output.last_hidden_state)
        pooled = sequence[:, 0]
        return self.intent_head(pooled), self.slot_head(sequence)


def save_bundle(model, tokenizer, out_dir, config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    torch.save(model.state_dict(), out_dir / "model.pt")
    with (out_dir / "casa_norte_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def load_bundle(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    with (model_dir / "casa_norte_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = JointBert(config["model_name"], config["n_intents"], config["n_slots"], config.get("dropout", 0.2))
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tokenizer, config


def align_slot_labels(tokens, tags, tokenizer, max_len):
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    word_ids = encoded.word_ids()
    labels = []
    seen = set()
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id not in seen:
            seen.add(word_id)
            labels.append(SLOT2ID.get(tags[word_id], 0))
        else:
            labels.append(-100)
    return encoded["input_ids"], encoded["attention_mask"], labels, word_ids


@torch.no_grad()
def predict(model, tokenizer, text, max_len=48):
    device = next(model.parameters()).device
    words = text.split()
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    word_ids = encoded.word_ids(0)
    intent_logits, slot_logits = model(encoded["input_ids"], encoded["attention_mask"])
    probs = torch.softmax(intent_logits, dim=-1)[0]
    confidence, intent_id = torch.max(probs, dim=-1)
    slot_ids = torch.argmax(slot_logits, dim=-1)[0].tolist()
    word_tags = ["O"] * len(words)
    seen = set()
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        if word_id < len(word_tags):
            word_tags[word_id] = ID2SLOT.get(slot_ids[position], "O")
    spans = []
    for slot_type, start, end in sorted(spans_from_tags(word_tags), key=lambda item: item[1]):
        spans.append({
            "type": slot_type,
            "value": " ".join(words[start:end]),
            "start": start,
            "end": end,
        })
    ordered = sorted(
        range(len(CLASSES)),
        key=lambda index: float(probs[index]),
        reverse=True,
    )
    return {
        "intent": ID2INTENT[int(intent_id)],
        "confidence": float(confidence),
        "probs": {ID2INTENT[index]: float(probs[index]) for index in ordered},
        "top": [(ID2INTENT[index], float(probs[index])) for index in ordered[:3]],
        "words": words,
        "tags": word_tags,
        "spans": spans,
    }
