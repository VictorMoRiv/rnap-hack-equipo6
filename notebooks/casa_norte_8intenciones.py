"""
Casa Norte - 8 intenciones: alarm_set, alarm_query, weather_query, lists_createoradd,
lists_remove, calendar_set, play_music, control_luces (= 5 intenciones iot_hue_* fusionadas)
Modelos: TF-IDF + LogReg (base) y zero-shot NLI multilingüe. Datos: MASSIVE es-ES.

pip install datasets scikit-learn transformers torch sentencepiece
"""
import re, json, random
from collections import Counter, defaultdict
import numpy as np
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------ DATOS
ds = load_dataset("AmazonScience/massive", "es-ES", trust_remote_code=True)
NAMES = ds["train"].features["intent"].names
LIGHTS = {"iot_hue_lighton", "iot_hue_lightoff", "iot_hue_lightup", "iot_hue_lightdim", "iot_hue_lightchange"}
KEEP = {"alarm_set", "alarm_query", "weather_query", "lists_createoradd", "lists_remove",
        "calendar_set", "play_music"} | LIGHTS
CLASSES = ["alarm_set", "alarm_query", "weather_query", "lists_createoradd", "lists_remove",
           "calendar_set", "play_music", "control_luces"]

def prep(split):
    rows = []
    for ex in split:
        name = NAMES[ex["intent"]]
        if name in KEEP:
            rows.append({"utt": ex["utt"], "annot": ex["annot_utt"],
                         "intent": "control_luces" if name in LIGHTS else name,
                         "sub": name})            # subtipo original (útil para luces)
    return rows

train, val, test = prep(ds["train"]), prep(ds["validation"]), prep(ds["test"])
print("Tamaños train/val/test:", len(train), len(val), len(test))
print("Distribución train:", Counter(r["intent"] for r in train))

SLOT_RE = re.compile(r"\[([^:\]]+?) : ([^\]]+?)\]")
def parse_bio(annot):
    toks, tags, pos = [], [], 0
    for m in SLOT_RE.finditer(annot):
        out = annot[pos:m.start()].split(); toks += out; tags += ["O"] * len(out)
        ent, s = m.group(2).split(), m.group(1).strip()
        toks += ent; tags += [f"B-{s}"] + [f"I-{s}"] * (len(ent) - 1); pos = m.end()
    rest = annot[pos:].split(); toks += rest; tags += ["O"] * len(rest)
    return toks, tags

# slots reales por intención (según train) -> para armar las tarjetas
slot_inv = defaultdict(Counter)
for r in train:
    for m in SLOT_RE.finditer(r["annot"]):
        slot_inv[r["intent"]][m.group(1).strip()] += 1
for k in CLASSES:
    print(f"Slots en {k}:", dict(slot_inv[k].most_common(6)))

# ------------------------------------------------------------------ TF-IDF + LogReg
def fit_tfidf(rows, labels, C):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000)
    texts = [r["utt"] for r in rows]
    X = hstack([w.fit_transform(texts), c.fit_transform(texts)]).tocsr()
    clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=1000, solver="liblinear", random_state=SEED)).fit(X, labels)
    return w, c, clf

def predict_tfidf(m, texts):
    w, c, clf = m
    X = hstack([w.transform(texts), c.transform(texts)]).tocsr()
    return clf.predict(X), clf.predict_proba(X)

yv = [r["intent"] for r in val]; vt = [r["utt"] for r in val]
best = None
for C in [1, 3, 10, 30]:                                   # se elige SOLO con validation
    m = fit_tfidf(train, [r["intent"] for r in train], C)
    f1 = f1_score(yv, predict_tfidf(m, vt)[0], average="macro")
    print(f"TF-IDF C={C}: macro-F1 val={f1:.4f}")
    if best is None or f1 > best[0]: best = (f1, C, m)
tfidf, C_BEST = best[2], best[1]
print("C elegido:", C_BEST)

# subtipo de luces (encender/apagar/subir/bajar/cambiar color)
lt = [r for r in train if r["intent"] == "control_luces"]
light_sub = fit_tfidf(lt, [r["sub"] for r in lt], C_BEST)

# ------------------------------------------------------------------ Extractor de slots
def tok_feats(toks, i):
    t = toks[i].lower()
    return {"w": t, "p3": t[:3], "s3": t[-3:], "dig": t.isdigit(), "first": i == 0,
            "pw": toks[i-1].lower() if i else "<s>", "pw2": toks[i-2].lower() if i > 1 else "<s>",
            "nw": toks[i+1].lower() if i < len(toks)-1 else "</s>"}

Xs, ys = [], []
for r in train:
    toks, tags = parse_bio(r["annot"])
    Xs += [tok_feats(toks, i) for i in range(len(toks))]; ys += tags
dv = DictVectorizer()
slot_clf = LogisticRegression(C=10, max_iter=300, solver="saga", random_state=SEED).fit(dv.fit_transform(Xs), ys)

def predict_slots(text):
    toks = text.split()
    return toks, list(slot_clf.predict(dv.transform([tok_feats(toks, i) for i in range(len(toks))])))

def spans(tags):
    out, cur = set(), None
    for i, t in enumerate(tags + ["O"]):
        if cur and not t.startswith("I-"): out.add((cur[0], cur[1], i)); cur = None
        if t.startswith("B-"): cur = (t[2:], i)
    return out

tp = fp = fn = 0
for r in test:
    toks, g = parse_bio(r["annot"])
    p = list(slot_clf.predict(dv.transform([tok_feats(toks, i) for i in range(len(toks))])))
    G, P = spans(g), spans(p); tp += len(G & P); fp += len(P - G); fn += len(G - P)
pr, rc = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
print("Slots span-F1 test:", round(2 * pr * rc / (pr + rc + 1e-9), 4))

# ------------------------------------------------------------------ Zero-shot multilingüe
from transformers import pipeline
ZS_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"       # alt.: joeddav/xlm-roberta-large-xnli
zs = pipeline("zero-shot-classification", model=ZS_MODEL, device=-1)   # device=0 con GPU
DESC = {"alarm_set": "poner una alarma", "alarm_query": "consultar mis alarmas",
        "weather_query": "saber el clima", "lists_createoradd": "agregar algo a una lista",
        "lists_remove": "quitar algo de una lista", "calendar_set": "agendar un evento en el calendario",
        "play_music": "escuchar música", "control_luces": "controlar las luces de la casa"}
labels_es = [DESC[c] for c in CLASSES]

def predict_zs(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        res = zs(texts[i:i+bs], candidate_labels=labels_es, hypothesis_template="El usuario quiere {}.",
                 multi_label=False, batch_size=bs)
        res = res if isinstance(res, list) else [res]
        out += [CLASSES[labels_es.index(o["labels"][0])] for o in res]
    return out

tt = [r["utt"] for r in test]; yt = [r["intent"] for r in test]
p_tfidf = list(predict_tfidf(tfidf, tt)[0]); p_zs = predict_zs(tt)
for name, p in [("TF-IDF + LogReg", p_tfidf), ("Zero-shot NLI", p_zs)]:
    print(f"\n== {name}: acc={accuracy_score(yt, p):.4f}  macro-F1={f1_score(yt, p, average='macro'):.4f}")
    print(classification_report(yt, p, labels=CLASSES, digits=3))

# ------------------------------------------------------------------ Intención -> componente de UI
UI = {"alarm_set": ("AlarmCard", "Activar alarma"), "alarm_query": ("AlarmListCard", "Ver alarmas"),
      "weather_query": ("WeatherCard", None), "lists_createoradd": ("ShoppingCard", "Agregar a la lista"),
      "lists_remove": ("ShoppingCard", "Quitar de la lista"), "calendar_set": ("EventCard", "Guardar evento"),
      "play_music": ("MusicCard", "Reproducir"), "control_luces": ("LightCard", "Aplicar")}
ACTION = {"iot_hue_lighton": "encender", "iot_hue_lightoff": "apagar", "iot_hue_lightup": "subir",
          "iot_hue_lightdim": "bajar", "iot_hue_lightchange": "cambiar color"}
THRESH = 0.50

def understand(text):
    _, pr = predict_tfidf(tfidf, [text]); k = int(pr[0].argmax())
    intent, conf = tfidf[2].classes_[k], float(pr[0][k])
    if conf < THRESH: return {"ui": "TextReply", "confidence": conf}
    toks, tags = predict_slots(text); ents = {}
    for t, g in zip(toks, tags):
        if g != "O": ents.setdefault(g[2:], []).append(t)
    out = {"intent": intent, "ui": UI[intent][0], "button": UI[intent][1],
           "data": {s: " ".join(v) for s, v in ents.items()}, "confidence": conf}
    if intent == "control_luces":
        out["data"]["accion"] = ACTION[predict_tfidf(light_sub, [text])[0][0]]
    return out

for frase in ["despiértame mañana a las seis y media", "agrega tortillas y frijoles a la lista del súper",
              "apaga la luz de la sala", "qué clima hace hoy en monterrey"]:
    print(frase, "->", json.dumps(understand(frase), ensure_ascii=False))
