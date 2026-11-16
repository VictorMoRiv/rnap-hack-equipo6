import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path

INTENTS = [
    "datetime_query", "iot_hue_lightchange", "transport_ticket", "takeaway_query",
    "qa_stock", "general_greet", "recommendation_events", "music_dislikeness",
    "iot_wemo_off", "cooking_recipe", "qa_currency", "transport_traffic",
    "general_quirky", "weather_query", "audio_volume_up", "email_addcontact",
    "takeaway_order", "email_querycontact", "iot_hue_lightup", "recommendation_locations",
    "play_audiobook", "lists_createoradd", "news_query", "alarm_query", "iot_wemo_on",
    "general_joke", "qa_definition", "social_query", "music_settings",
    "audio_volume_other", "calendar_remove", "iot_hue_lightdim", "calendar_query",
    "email_sendemail", "iot_cleaning", "audio_volume_down", "play_radio",
    "cooking_query", "datetime_convert", "qa_maths", "iot_hue_lightoff",
    "iot_hue_lighton", "transport_query", "music_likeness", "email_query",
    "play_music", "audio_volume_mute", "social_post", "alarm_set", "qa_factoid",
    "calendar_set", "play_game", "alarm_remove", "lists_remove", "transport_taxi",
    "recommendation_movies", "iot_coffee", "music_query", "play_podcasts", "lists_query",
]

LIGHTS = {
    "iot_hue_lighton", "iot_hue_lightoff", "iot_hue_lightup",
    "iot_hue_lightdim", "iot_hue_lightchange",
}

CLASSES = [
    "alarm_set", "alarm_query", "weather_query", "lists_createoradd",
    "lists_remove", "calendar_set", "play_music", "control_luces",
]

CLASS_ES = {
    "alarm_set": "poner una alarma",
    "alarm_query": "consultar alarmas",
    "weather_query": "consultar el clima",
    "lists_createoradd": "agregar a una lista",
    "lists_remove": "quitar de una lista",
    "calendar_set": "agendar un evento",
    "play_music": "reproducir música",
    "control_luces": "controlar las luces",
    "out_of_scope": "fuera de las acciones de Casa Norte",
}

CANONICAL_SLOTS = [
    "time", "date", "timeofday", "place_name", "event_name", "person",
    "list_name", "artist_name", "song_name", "playlist_name", "music_genre",
    "media_type", "house_place", "device_type", "color_type", "change_amount",
    "weather_descriptor", "general_frequency", "relation",
]

SLOT_LABELS = ["O"] + [f"B-{s}" for s in CANONICAL_SLOTS] + [f"I-{s}" for s in CANONICAL_SLOTS]
SLOT2ID = {label: i for i, label in enumerate(SLOT_LABELS)}
ID2SLOT = {i: label for label, i in SLOT2ID.items()}
INTENT2ID = {name: i for i, name in enumerate(CLASSES)}
ID2INTENT = {i: name for name, i in INTENT2ID.items()}

SLOT_RE = re.compile(r"\[([^:\]]+?)\s*:\s*([^\]]+?)\]")

MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

WEEKDAYS = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2, "jueves": 3,
    "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
}

NUMBERS = {
    "cero": 0, "una": 1, "uno": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciséis": 16, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiún": 21,
    "veintidós": 22, "veintidos": 22, "veintitrés": 23, "veintitres": 23,
}

INTENT_ACTIONS = {
    "alarm_set": "Guardar alarma",
    "alarm_query": "Ver alarmas",
    "weather_query": "Consultar clima",
    "lists_createoradd": "Agregar a la lista",
    "lists_remove": "Quitar de la lista",
    "calendar_set": "Guardar evento",
    "play_music": "Reproducir",
    "control_luces": "Aplicar",
}


def parse_annot(annot):
    tokens, tags, pos = [], [], 0
    for match in SLOT_RE.finditer(annot):
        before = annot[pos:match.start()].split()
        tokens += before
        tags += ["O"] * len(before)
        slot = match.group(1).strip()
        words = match.group(2).split()
        if slot in CANONICAL_SLOTS and words:
            tags += [f"B-{slot}"] + [f"I-{slot}"] * (len(words) - 1)
        else:
            tags += ["O"] * len(words)
        tokens += words
        pos = match.end()
    rest = annot[pos:].split()
    tokens += rest
    tags += ["O"] * len(rest)
    return tokens, tags


def spans_from_tags(tags):
    spans, current = set(), None
    for index, tag in enumerate(tags + ["O"]):
        if current and tag != f"I-{current[0]}":
            spans.add((current[0], current[1], index))
            current = None
        if tag.startswith("B-"):
            current = (tag[2:], index)
    return spans


def strip_accents(value):
    return "".join(
        char for char in unicodedata.normalize("NFD", value.lower())
        if unicodedata.category(char) != "Mn"
    ).strip("¿?¡!.,;:")


def find_span(words, value):
    wanted = [strip_accents(token) for token in value.split()]
    normalized = [strip_accents(token) for token in words]
    for start in range(len(normalized) - len(wanted) + 1):
        if normalized[start:start + len(wanted)] == wanted:
            return start, start + len(wanted)
    return None


def span_prf(gold, pred):
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def load_jsonl(path, keep_classes=None):
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            name = INTENTS[record["intent"]] if isinstance(record["intent"], int) else record["intent"]
            intent = "control_luces" if name in LIGHTS else name
            if keep_classes is not None and intent not in keep_classes:
                continue
            tokens, tags = parse_annot(record["annot_utt"])
            rows.append({
                "id": record.get("id"),
                "text": " ".join(tokens),
                "raw": record["utt"],
                "tokens": tokens,
                "tags": tags,
                "intent": intent,
                "sub": name,
            })
    return rows


def load_split(data_dir, split, keep_classes=None):
    return load_jsonl(Path(data_dir) / f"massive_es-ES_{split}.jsonl", keep_classes)


def load_mexican(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _to_number(token):
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return NUMBERS.get(token)


def normalize_time(value):
    if not value:
        return None
    text = value.lower().strip()
    text = text.replace(".", ":").replace(" h", "")
    if "mediodía" in text or "mediodia" in text:
        return "12:00"
    if "medianoche" in text:
        return "00:00"
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if "tarde" in text or "noche" in text:
            if hour < 12:
                hour += 12
        return f"{hour:02d}:{minute:02d}"
    match = re.search(r"(\d{1,2})\s*(?:y\s*media|y\s*30|:30)", text)
    if match:
        return f"{int(match.group(1)):02d}:30"
    match = re.search(r"(\d{1,2})\s*(?:y\s*cuarto|y\s*15|:15)", text)
    if match:
        return f"{int(match.group(1)):02d}:15"
    match = re.search(r"(\d{1,2})\s*(?:menos\s*cuarto)", text)
    if match:
        return f"{(int(match.group(1)) - 1) % 24:02d}:45"
    match = re.search(r"(\d{1,2})\s*(?:de la\s*)?(mañana|tarde|noche|pm|am)?", text)
    if match:
        hour = int(match.group(1))
        suffix = match.group(2) or ""
        if suffix in {"tarde", "noche", "pm"} and hour < 12:
            hour += 12
        return f"{hour:02d}:00"
    for word, number in NUMBERS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            hour = number
            if any(marker in text for marker in ("tarde", "noche", "pm")) and hour < 12:
                hour += 12
            if "media" in text:
                return f"{hour:02d}:30"
            if "cuarto" in text:
                return f"{hour:02d}:15"
            return f"{hour:02d}:00"
    return None


def normalize_date(value, today=None):
    if not value:
        return None
    text = value.lower().strip()
    today = today or date.today()
    if "pasado mañana" in text or "pasado manana" in text:
        return (today + timedelta(days=2)).isoformat()
    if "mañana" in text or "manana" in text:
        return (today + timedelta(days=1)).isoformat()
    if "hoy" in text:
        return today.isoformat()
    if "ayer" in text:
        return (today - timedelta(days=1)).isoformat()
    for name, index in WEEKDAYS.items():
        if re.search(rf"\b{name}\b", text):
            delta = (index - today.weekday()) % 7
            if delta == 0:
                delta = 7
            return (today + timedelta(days=delta)).isoformat()
    match = re.search(r"(\d{1,2})\s+de\s+([a-záéíóú]+)", text)
    if match:
        day = int(match.group(1))
        month = MONTHS.get(match.group(2))
        if month:
            year = today.year
            candidate = date(year, month, day)
            if candidate < today:
                candidate = date(year + 1, month, day)
            return candidate.isoformat()
    match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        year = int(match.group(3) or today.year)
        if year < 100:
            year += 2000
        return date(year, month, day).isoformat()
    return None


def extract_item(text):
    patterns = [
        r"(?:agrega|añade|anade|anota|pon|quita|borra|elimina|saca)\s+(.+?)\s+(?:a|en|de)\s+(?:la|mi|el)\s+lista",
        r"(?:hay que|necesito|quiero|debemos)\s+comprar\s+(.+)",
        r"(?:agrega|añade|anade|anota|pon|quita|borra|elimina|saca)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            item = match.group(1).strip(" .,")
            item = re.sub(r"^(los|las|el|la|un|una)\s+", "", item, flags=re.IGNORECASE)
            return item or None
    return None


def extract_list_name(text):
    match = re.search(r"lista\s+(?:de|del|de la|de el)\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        name = re.sub(r"^(la|el|los|las)\s+", "", match.group(1).strip(" .,"), flags=re.IGNORECASE)
        return name or None
    return None


def light_action(text):
    lowered = text.lower()
    if re.search(r"\b(apaga|apagar|apágala|apagala|off)\b", lowered):
        return "apagar"
    if re.search(r"\b(prende|prender|enciende|encender|prende|on)\b", lowered):
        return "encender"
    if re.search(r"\b(sube|subir|aumenta|aumentar|más|mas)\b", lowered):
        return "subir"
    if re.search(r"\b(baja|bajar|disminuye|disminuir|menos)\b", lowered):
        return "bajar"
    if re.search(r"\b(color|cálida|calida|fría|fria|rojo|azul|verde|morado)\b", lowered):
        return "cambiar color"
    return "encender"
