import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .nlu import CLASSES, ID2INTENT, ID2SLOT, SLOT2ID, spans_from_tags

DEFAULT_ATTN = "sdpa"


class JointBert(nn.Module):
    def __init__(self, model_name, n_intents, n_slots, dropout=0.2, encoder_config=None, encoder=None):
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        elif encoder_config is None:
            self.encoder = AutoModel.from_pretrained(model_name, attn_implementation=DEFAULT_ATTN)
        else:
            encoder_config._attn_implementation = DEFAULT_ATTN
            self.encoder = AutoModel.from_config(encoder_config)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, n_intents)
        self.slot_head = nn.Linear(hidden, n_slots)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        sequence = self.dropout(output.last_hidden_state)
        pooled = sequence[:, 0]
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(sequence)
        if output_attentions:
            return intent_logits, slot_logits, output.attentions
        return intent_logits, slot_logits


def save_bundle(model, tokenizer, out_dir, config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    model.encoder.config.save_pretrained(out_dir / "encoder_config")
    torch.save(model.state_dict(), out_dir / "model.pt")
    with (out_dir / "casa_norte_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def load_bundle(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    with (model_dir / "casa_norte_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    encoder_config = None
    encoder = None
    config_dir = model_dir / "encoder_config"
    if config_dir.exists():
        encoder_config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
        encoder_config._attn_implementation = DEFAULT_ATTN
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        with torch.device("meta"):
            encoder = AutoModel.from_config(encoder_config)
        for module in encoder.modules():
            for name, buffer in list(module.named_buffers(recurse=False)):
                if not buffer.is_meta:
                    continue
                if name == "position_ids":
                    value = torch.arange(buffer.shape[-1]).expand(1, -1).clone()
                else:
                    value = torch.zeros(buffer.shape, dtype=buffer.dtype)
                setattr(module, name, value)
        encoder.load_state_dict(encoder_state, strict=True, assign=True)
    model = JointBert(
        config["model_name"],
        config["n_intents"],
        config["n_slots"],
        config.get("dropout", 0.2),
        encoder_config=encoder_config,
        encoder=encoder,
    )
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


@torch.no_grad()
def plot_attention_map(text, model, tokenizer, max_len=48, output_path=None):
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
    previous_impl = model.encoder.config._attn_implementation
    model.encoder.set_attn_implementation("eager")
    try:
        _, _, attentions = model(
            encoded["input_ids"], encoded["attention_mask"], output_attentions=True
        )
    finally:
        model.encoder.set_attn_implementation(previous_impl)
    weights = attentions[-1][0].mean(dim=0).cpu().numpy()
    positions, labels = [], []
    previous = object()
    for position, word_id in enumerate(word_ids):
        if word_id is None:
            positions.append(position)
            labels.append("[CLS]" if position == 0 else "[SEP]")
        elif word_id != previous:
            positions.append(position)
            labels.append(words[word_id] if word_id < len(words) else f"w{word_id}")
        previous = word_id
    matrix = weights[np.ix_(positions, positions)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = min(15.0, max(7.0, 0.55 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    image = ax.imshow(matrix, cmap="Oranges", vmin=0.0, vmax=float(matrix.max() or 1.0))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Atención promedio · última capa · BETO")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    target = Path(output_path or Path.cwd() / "attention_map.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(target), matrix, labels
