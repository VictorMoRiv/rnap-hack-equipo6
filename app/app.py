import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casa_norte.model import load_bundle, plot_attention_map, predict  # noqa: E402
from casa_norte.nlu import (  # noqa: E402
    CLASS_ES, clean_entity, extract_items, light_action, normalize_date,
    normalize_time,
)

NEW_MODEL_DIR = ROOT / "modelo_casa_norte"
OLD_MODEL_DIR = ROOT / "models" / "beto_joint"
MODEL_DIR = NEW_MODEL_DIR if NEW_MODEL_DIR.exists() else OLD_MODEL_DIR
THRESHOLD = 0.65
BUNDLE = None
LOAD_ERROR = None

try:
    if MODEL_DIR.exists():
        BUNDLE = load_bundle(MODEL_DIR)
        THRESHOLD = float(BUNDLE[2].get("threshold", THRESHOLD))
except Exception as error:
    LOAD_ERROR = str(error)

EXAMPLES = [
    "despiértame mañana a las seis y cuarto",
    "cómo va a estar el calor hoy en Monterrey",
    "agrega tortillas, aguacate y un doce a la lista del súper",
    "recuérdame jugar gta vi a las 4",
    "pon la luz cálida en la recámara",
    "transfiere 500 pesos a mi hermana",
]

WEATHER_DEMO = {
    "monterrey": ("28 °C", "soleado"),
    "guadalajara": ("24 °C", "parcialmente nublado"),
    "cdmx": ("21 °C", "nublado"),
    "ciudad de méxico": ("21 °C", "nublado"),
    "saltillo": ("26 °C", "soleado"),
    "zapopan": ("23 °C", "parcialmente nublado"),
}

GROUP_ORDER = ["alarm", "alarm_query", "weather", "lists", "calendar", "music", "lights", "oos"]
FIELD_ORDER = [
    "alarm_time", "alarm_date", "alarm_event",
    "weather_place", "weather_date",
    "list_name", "list_item", "list_mode",
    "cal_event", "cal_date", "cal_time", "cal_person",
    "music_song", "music_artist", "music_playlist",
    "light_place", "light_action", "light_color",
]

INTENT_GROUP = {
    "alarm_set": "alarm",
    "alarm_query": "alarm_query",
    "weather_query": "weather",
    "lists_createoradd": "lists",
    "lists_remove": "lists",
    "calendar_set": "calendar",
    "play_music": "music",
    "control_luces": "lights",
}

OOS_PATTERNS = [
    r"\b(dólar|dolar|cotización|tipo de cambio)\b",
    r"\b(pizza|pide|pedir|domicilio)\b",
    r"\b(correo|email|correos)\b",
    r"\b(chiste|chistes)\b",
    r"\b(película|pelicula|películas|peliculas)\b",
    r"\b(receta|recetas|cocinar)\b",
    r"\b(noticias|traducir)\b",
    r"\b(transfiere|transferir|transferencia|envía dinero|envia dinero)\b",
    r"\b(pagar|paga)\b.{0,20}\b(pesos|dólar|dolar)\b",
]

CALENDAR_CUES = r"\b(recuérdame|recuerdame|recordatorio|avísame|avisame|agenda|apunta|programa)\b"

CSS = """
.gradio-container {
    background: #F7F5F0 !important;
    color: #1E242B !important;
    font-family: Inter, system-ui, -apple-system, sans-serif !important;
}
.cn-banner {
    background: #1E242B !important;
    color: #F7F5F0 !important;
    border-left: 6px solid #D16638 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
}
.cn-banner b { color: #D16638 !important; }
.cn-step {
    color: #D16638 !important;
    font-weight: 700 !important;
}
.gr-button-primary {
    background: #D16638 !important;
    border-color: #D16638 !important;
    color: #FFFFFF !important;
}
.gr-button-primary:hover { background: #b8572e !important; }
.gr-group {
    border: 1px solid rgba(30, 36, 43, 0.12) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    background: rgba(255, 255, 255, 0.65) !important;
}
.gr-markdown h3 { color: #1E242B !important; }
.gr-markdown strong { color: #D16638 !important; }
footer { visibility: hidden !important; }
"""

BANNER = """
<div class="cn-banner">
<div style="font-size:20px;font-weight:700;letter-spacing:.2px">CASA NORTE <span style="color:#D16638">·</span> asistente de acciones</div>
<div style="margin-top:6px;font-size:14px;opacity:.92">
<span class="cn-step">1</span> escribes → <span class="cn-step">2</span> entendemos intención y datos →
<span class="cn-step">3</span> revisas una tarjeta editable → <span class="cn-step">4</span> ejecutamos y actualizamos el estado
</div>
<div style="margin-top:6px;font-size:12px;opacity:.7">
Si algo no está entre las 8 acciones disponibles, lo decimos claro en vez de inventar una respuesta.
</div>
</div>
"""


def clean_value(value):
    if not value:
        return ""
    return clean_entity(value)


def slot_values(result):
    values = {}
    spans = result["spans"]
    index = 0
    while index < len(spans):
        span = spans[index]
        value = span["value"]
        end = span["end"]
        next_index = index + 1
        while (
            next_index < len(spans)
            and spans[next_index]["type"] == span["type"]
            and spans[next_index]["start"] == end
        ):
            value += " " + spans[next_index]["value"]
            end = spans[next_index]["end"]
            next_index += 1
        values.setdefault(span["type"], clean_value(value))
        index = next_index
    return values


def guess_time(text):
    patterns = [
        r"(?:a\s+las?|para\s+las?)\s+([^,.;]+)",
        r"\b(en\s+\d{1,3}\s+minutos?)\b",
        r"\b(a\s+mediod[ií]a)\b",
        r"\b(a\s+medianoche)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_time(match.group(1))
    return None


def guess_date(text):
    return normalize_date(text)


def guess_music(text):
    match = re.search(r"(?:reproduce|toca|escuchar)\s+(.+)", text, flags=re.IGNORECASE)
    if match:
        query = match.group(1).strip(" .,")
    else:
        match = re.search(r"pon\s+(.+)", text, flags=re.IGNORECASE)
        if not match:
            return None
        query = re.sub(r"^la canción\s+", "", match.group(1).strip(" .,"), flags=re.IGNORECASE)
    lowered = query.lower()
    if lowered.startswith("música de") or lowered.startswith("musica de") or lowered.startswith("mi playlist"):
        return None
    return query or None


def guess_playlist(text):
    match = re.search(r"playlist\s+(?:de\s+)?(.+)", text, flags=re.IGNORECASE)
    if match:
        return clean_value(match.group(1)) or None
    return None


def split_items(value):
    if not value:
        return []
    parts = re.split(r"\s*,\s*|\s+y\s+|\s+e\s+|\s+and\s+", str(value), flags=re.IGNORECASE)
    items = []
    for part in parts:
        item = clean_value(part)
        if item and item.lower() not in {entry.lower() for entry in items}:
            items.append(item)
    return items


def out_of_scope(text):
    lowered = text.lower()
    if re.search(CALENDAR_CUES, lowered):
        return False
    return any(re.search(pattern, lowered) for pattern in OOS_PATTERNS)


def render_state(alarms, shopping, events, lights, music):
    lines = ["### Estado de la casa"]
    lines.append(f"**Alarmas:** {len(alarms)}")
    for alarm in alarms[-3:]:
        lines.append(f"- {alarm['hora'] or 'sin hora'} · {alarm['fecha'] or 'sin fecha'} · {alarm['evento']}")
    lines.append(f"**Lista:** {', '.join(shopping) if shopping else '(vacía)'}")
    lines.append(f"**Eventos:** {len(events)}")
    for event in events[-3:]:
        lines.append(f"- {event['evento']} · {event['fecha'] or 'sin fecha'} · {event['hora'] or 'sin hora'}")
    if lights:
        lines.append("**Luces:**")
        for place, state in lights.items():
            lines.append(f"- {place}: {state['accion']} {state.get('color') or ''}".rstrip())
    else:
        lines.append("**Luces:** (sin cambios)")
    lines.append(f"**Música:** {music or '(nada reproduciéndose)'}")
    return "\n".join(lines)


def no_model_message():
    detail = f"\n\n`{LOAD_ERROR}`" if LOAD_ERROR else ""
    return (
        "### Modelo no disponible\n"
        f"Entrena con `python notebooks/fase2_transformer.py`. Salida esperada: `{MODEL_DIR}`."
        f"{detail}"
    )


def unknown_card(reason):
    return (
        "### No reconozco esa acción todavía\n"
        f"{reason}\n\n"
        "Puedo ayudarte con **alarma, clima, lista, calendario, música y luces**."
    )


def group_updates(active):
    return [gr.update(visible=name == active) for name in GROUP_ORDER]


def blank_fields():
    return [gr.update(value="") for _ in FIELD_ORDER]


def empty_result(summary, card, state_md):
    return (
        summary, "", card, None,
        *group_updates(None),
        *blank_fields(),
        state_md,
    )


def understand(text, alarms, shopping, events, lights, music):
    state_md = render_state(alarms, shopping, events, lights, music)
    if BUNDLE is None:
        return empty_result("No hay modelo cargado.", no_model_message(), state_md)
    text = (text or "").strip()
    if not text:
        return empty_result("Escribe una frase para Casa Norte.", "", state_md)

    started = time.perf_counter()
    model, tokenizer, config = BUNDLE
    result = predict(model, tokenizer, text, max_len=config.get("max_len", 40))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    intent = result["intent"]
    confidence = result["confidence"]
    slots = slot_values(result)
    active_group = INTENT_GROUP.get(intent)
    details_md = (
        "#### Slots detectados\n"
        + "\n".join(f"- `{span['type']}`: {span['value']}" for span in result["spans"])
        + "\n\n#### Intenciones probables\n"
        + "\n".join(f"- {name}: {prob:.3f}" for name, prob in result["top"])
        + f"\n\n**Inferencia:** {elapsed_ms:.0f} ms · **umbral:** {THRESHOLD:.2f}"
    )
    groups = group_updates(active_group)
    values = {name: "" for name in FIELD_ORDER}
    values["list_mode"] = "Agregar"

    guard = out_of_scope(text)
    if guard or confidence < THRESHOLD:
        if guard:
            reason = "La frase pide una acción fuera de las 8 que Casa Norte puede ejecutar."
            summary = f"**Fuera de alcance** · no intentaré adivinar · {elapsed_ms:.0f} ms."
        else:
            reason = (
                f"La mejor opción fue **{CLASS_ES[intent]}**, pero la confianza fue "
                f"{confidence:.2f} y el corte es {THRESHOLD:.2f}."
            )
            summary = f"**No reconozco esa acción todavía** · confianza {confidence:.2f} · {elapsed_ms:.0f} ms."
        card = unknown_card(reason)
        return (
            summary, details_md, card, None,
            *group_updates("oos"),
            *[gr.update(value=values[name]) for name in FIELD_ORDER],
            state_md,
        )

    summary = (
        f"**Entendí:** {CLASS_ES[intent]} · confianza **{confidence:.2f}** · "
        f"{elapsed_ms:.0f} ms"
    )
    pending = {"intent": intent, "text": text, "slots": slots}

    if intent == "alarm_set":
        hora = normalize_time(slots.get("time")) or guess_time(text) or ""
        fecha = normalize_date(slots.get("date")) or guess_date(text) or ""
        evento = clean_value(slots.get("event_name")) or "Alarma"
        values.update({"alarm_time": hora, "alarm_date": fecha, "alarm_event": evento})
        card = (
            "### Tarjeta de alarma\n"
            f"- Hora: **{hora or 'por definir'}**\n"
            f"- Fecha: **{fecha or 'sin fecha'}**\n"
            f"- Evento: **{evento}**\n\n"
            "Verifica los datos y guarda la alarma."
        )
    elif intent == "alarm_query":
        card = "### Tus alarmas\n" + (
            "\n".join(f"- {item['hora']} · {item['fecha']} · {item['evento']}" for item in alarms)
            if alarms else "Todavía no hay alarmas guardadas."
        )
    elif intent == "weather_query":
        lugar = clean_value(slots.get("place_name")) or ""
        fecha = normalize_date(slots.get("date") or slots.get("timeofday")) or ""
        values.update({"weather_place": lugar, "weather_date": fecha})
        card = (
            "### Tarjeta de clima\n"
            f"- Lugar: **{lugar or 'por definir'}**\n"
            f"- Fecha: **{fecha or 'hoy'}**\n\n"
            "Consulta el pronóstico demo."
        )
    elif intent in {"lists_createoradd", "lists_remove"}:
        lista = clean_value(slots.get("list_name")) or ""
        if not lista:
            match = re.search(r"lista\s+(?:de|del|de la|de el)\s+(.+)$", text, flags=re.IGNORECASE)
            lista = clean_value(match.group(1)) if match else ""
        items = extract_items(text)
        if not items:
            match = re.search(r"(?:agrega|anota|pon|quita|borra|elimina|saca)\s+(.+)$", text, flags=re.IGNORECASE)
            items = split_items(match.group(1)) if match else []
        values.update({
            "list_name": lista or "compras",
            "list_item": ", ".join(items),
            "list_mode": "Agregar" if intent == "lists_createoradd" else "Quitar",
        })
        card = (
            f"### Tarjeta de lista · {values['list_mode']}\n"
            f"- Lista: **{values['list_name']}**\n"
            f"- Artículos: **{values['list_item'] or 'por definir'}**\n\n"
            "Separa varios artículos con comas o con «y»."
        )
    elif intent == "calendar_set":
        evento = clean_value(slots.get("event_name")) or ""
        fecha = normalize_date(slots.get("date")) or ""
        hora = normalize_time(slots.get("time")) or guess_time(text) or ""
        persona = clean_value(slots.get("person")) or ""
        values.update({
            "cal_event": evento, "cal_date": fecha, "cal_time": hora, "cal_person": persona,
        })
        card = (
            "### Tarjeta de calendario\n"
            f"- Evento: **{evento or 'por definir'}**\n"
            f"- Fecha: **{fecha or 'sin fecha'}**\n"
            f"- Hora: **{hora or 'sin hora'}**\n"
            f"- Persona: **{persona or 'sin persona'}**\n\n"
            "Revisa y guarda el evento."
        )
    elif intent == "play_music":
        artista = clean_value(slots.get("artist_name")) or ""
        playlist = clean_value(slots.get("playlist_name")) or guess_playlist(text) or ""
        cancion = clean_value(slots.get("song_name")) or ""
        guessed = guess_music(text) if not artista and not playlist else None
        if guessed and (not cancion or cancion.lower() in guessed.lower()):
            cancion = clean_value(guessed)
        values.update({
            "music_song": cancion, "music_artist": artista, "music_playlist": playlist,
        })
        card = (
            "### Tarjeta de música\n"
            f"- Canción: **{cancion or 'sin canción'}**\n"
            f"- Artista: **{artista or 'sin artista'}**\n"
            f"- Playlist: **{playlist or 'sin playlist'}**\n\n"
            "La reproducción se abre en modo demo."
        )
    else:
        lugar = clean_value(slots.get("house_place")) or ""
        accion = light_action(text)
        color = clean_value(slots.get("color_type")) or ""
        values.update({"light_place": lugar, "light_action": accion, "light_color": color})
        card = (
            "### Tarjeta de luces\n"
            f"- Lugar: **{lugar or 'por definir'}**\n"
            f"- Acción: **{accion}**\n"
            f"- Color: **{color or 'sin cambio'}**\n\n"
            "Ajusta y aplica el cambio."
        )

    return (
        summary, details_md, card, pending,
        *groups,
        *[gr.update(value=values[name]) for name in FIELD_ORDER],
        state_md,
    )


def invalid_action(message, alarms, shopping, events, lights, music):
    return (
        message, render_state(alarms, shopping, events, lights, music),
        alarms, shopping, events, lights, music,
    )


def commit_action(intent, data, alarms, shopping, events, lights, music):
    alarms = list(alarms)
    shopping = list(shopping)
    events = list(events)
    lights = dict(lights)
    if intent == "alarm_set":
        if not data.get("hora"):
            message = "Falta la hora de la alarma."
        else:
            alarms.append({
                "hora": data["hora"], "fecha": data.get("fecha", ""),
                "evento": data.get("evento") or "Alarma",
            })
            message = f"Alarma guardada: {data.get('evento') or 'Alarma'} a las {data['hora']}."
    elif intent == "alarm_query":
        message = f"Tienes {len(alarms)} alarmas guardadas."
    elif intent == "weather_query":
        place = data.get("lugar") or "tu ciudad"
        temp, sky = WEATHER_DEMO.get(place.lower(), ("24 °C", "parcialmente nublado"))
        message = f"Pronóstico demo para {place} ({data.get('fecha') or 'hoy'}): {temp}, {sky}."
    elif intent in {"lists_createoradd", "lists_remove"}:
        items = split_items(data.get("articulos"))
        if not items:
            message = "Falta el artículo."
        elif intent == "lists_createoradd":
            shopping.extend(items)
            message = f"Agregados {', '.join(items)} a la lista **{data.get('lista') or 'compras'}**."
        else:
            removed = [item for item in items if item in shopping]
            for item in removed:
                shopping.remove(item)
            message = (
                f"Quitados {', '.join(removed)} de la lista."
                if removed else "Ninguno de esos artículos estaba en la lista."
            )
    elif intent == "calendar_set":
        if not data.get("evento"):
            message = "Falta el nombre del evento."
        else:
            events.append({
                "evento": data["evento"], "fecha": data.get("fecha", ""),
                "hora": data.get("hora", ""), "persona": data.get("persona", ""),
            })
            message = f"Evento guardado: **{data['evento']}**."
    elif intent == "play_music":
        query = " ".join(part for part in [data.get("cancion"), data.get("artista"), data.get("playlist")] if part).strip()
        if not query:
            message = "Falta la canción, artista o playlist."
        else:
            music = query
            link = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
            message = f"Reproduciendo **{query}** en modo demo. [Abrir búsqueda]({link})"
    else:
        place = data.get("lugar") or "sala"
        action = data.get("accion") or "encender"
        color = data.get("color") or ""
        lights[place] = {"accion": action, "color": color}
        message = f"Luces de **{place}**: {action} {color}".rstrip() + "."
    return message, render_state(alarms, shopping, events, lights, music), alarms, shopping, events, lights, music


def save_alarm(pending, hora, fecha, evento, alarms, shopping, events, lights, music):
    if not pending or pending["intent"] != "alarm_set":
        return invalid_action("Primero entiende una frase de alarma.", alarms, shopping, events, lights, music)
    return commit_action("alarm_set", {"hora": hora, "fecha": fecha, "evento": evento}, alarms, shopping, events, lights, music)


def show_alarms(pending, alarms, shopping, events, lights, music):
    return commit_action("alarm_query", {}, alarms, shopping, events, lights, music)


def query_weather(pending, lugar, fecha, alarms, shopping, events, lights, music):
    if not pending or pending["intent"] != "weather_query":
        return invalid_action("Primero entiende una frase de clima.", alarms, shopping, events, lights, music)
    return commit_action("weather_query", {"lugar": lugar, "fecha": fecha}, alarms, shopping, events, lights, music)


def apply_list(pending, lista, articulos, modo, alarms, shopping, events, lights, music):
    if not pending or pending["intent"] not in {"lists_createoradd", "lists_remove"}:
        return invalid_action("Primero entiende una frase de lista.", alarms, shopping, events, lights, music)
    return commit_action(pending["intent"], {"lista": lista, "articulos": articulos}, alarms, shopping, events, lights, music)


def save_event(pending, evento, fecha, hora, persona, alarms, shopping, events, lights, music):
    if not pending or pending["intent"] != "calendar_set":
        return invalid_action("Primero entiende una frase de calendario.", alarms, shopping, events, lights, music)
    return commit_action("calendar_set", {
        "evento": evento, "fecha": fecha, "hora": hora, "persona": persona,
    }, alarms, shopping, events, lights, music)


def play_music(pending, cancion, artista, playlist, alarms, shopping, events, lights, music):
    if not pending or pending["intent"] != "play_music":
        return invalid_action("Primero entiende una frase de música.", alarms, shopping, events, lights, music)
    return commit_action("play_music", {
        "cancion": cancion, "artista": artista, "playlist": playlist,
    }, alarms, shopping, events, lights, music)


def apply_lights(pending, lugar, accion, color, alarms, shopping, events, lights, music):
    if not pending or pending["intent"] != "control_luces":
        return invalid_action("Primero entiende una frase de luces.", alarms, shopping, events, lights, music)
    return commit_action("control_luces", {
        "lugar": lugar, "accion": accion, "color": color,
    }, alarms, shopping, events, lights, music)


def reset_state(alarms, shopping, events, lights, music):
    return (
        "Estado reiniciado.",
        render_state([], [], [], {}, None),
        [], [], [], {}, None,
    )


def complete_state(alarms, shopping, events, lights, music):
    return (
        "Acciones marcadas como completadas.",
        render_state([], [], [], {}, None),
        [], [], [], {}, None,
    )


def render_attention(text):
    if BUNDLE is None:
        return None, no_model_message()
    text = (text or "").strip() or "cómo va a estar el calor hoy en Monterrey"
    model, tokenizer, config = BUNDLE
    target = ROOT / "notebooks" / "figs" / "attention_app.png"
    path, _, labels = plot_attention_map(
        text, model, tokenizer, config.get("max_len", 40), target
    )
    return path, f"Mapa generado para **{text}** · {len(labels)} tokens visibles · última capa, promedio de heads."


with gr.Blocks(title="Casa Norte · asistente de acciones") as demo:
    gr.HTML(BANNER, elem_classes=["cn-banner"])
    with gr.Tabs():
        with gr.Tab("Asistente"):
            with gr.Row():
                textbox = gr.Textbox(
                    label="¿Qué necesitas?",
                    placeholder="Ej. despiértame a las seis y cuarto",
                    scale=5,
                )
                submit = gr.Button("Entender", variant="primary", scale=1)
                clear = gr.Button("Limpiar", scale=1)
            gr.Examples(examples=[[item] for item in EXAMPLES], inputs=[textbox])
            with gr.Row():
                with gr.Column(scale=3):
                    summary = gr.Markdown()
                    with gr.Accordion("Ver slots, probabilidades y latencia", open=False):
                        details = gr.Markdown()
                    action_output = gr.Markdown()

                    with gr.Group(visible=False) as alarm_group:
                        gr.Markdown("#### Alarma")
                        with gr.Row():
                            alarm_time = gr.Textbox(label="Hora")
                            alarm_date = gr.Textbox(label="Fecha")
                            alarm_event = gr.Textbox(label="Evento")
                        alarm_save = gr.Button("Guardar alarma", variant="primary")

                    with gr.Group(visible=False) as alarm_query_group:
                        gr.Markdown("#### Alarmas guardadas")
                        alarm_show = gr.Button("Actualizar lista", variant="primary")

                    with gr.Group(visible=False) as weather_group:
                        gr.Markdown("#### Clima")
                        with gr.Row():
                            weather_place = gr.Textbox(label="Lugar")
                            weather_date = gr.Textbox(label="Fecha")
                        weather_go = gr.Button("Consultar clima", variant="primary")

                    with gr.Group(visible=False) as lists_group:
                        gr.Markdown("#### Lista")
                        with gr.Row():
                            list_name = gr.Textbox(label="Lista")
                            list_mode = gr.Radio(["Agregar", "Quitar"], value="Agregar", label="Acción")
                        list_item = gr.Textbox(label="Artículos (separa con comas o «y»)")
                        list_go = gr.Button("Aplicar a la lista", variant="primary")

                    with gr.Group(visible=False) as calendar_group:
                        gr.Markdown("#### Calendario")
                        with gr.Row():
                            cal_event = gr.Textbox(label="Evento")
                            cal_person = gr.Textbox(label="Persona")
                        with gr.Row():
                            cal_date = gr.Textbox(label="Fecha")
                            cal_time = gr.Textbox(label="Hora")
                        cal_save = gr.Button("Guardar evento", variant="primary")

                    with gr.Group(visible=False) as music_group:
                        gr.Markdown("#### Música")
                        with gr.Row():
                            music_song = gr.Textbox(label="Canción")
                            music_artist = gr.Textbox(label="Artista")
                            music_playlist = gr.Textbox(label="Playlist")
                        music_go = gr.Button("Reproducir", variant="primary")

                    with gr.Group(visible=False) as lights_group:
                        gr.Markdown("#### Luces")
                        with gr.Row():
                            light_place = gr.Textbox(label="Lugar")
                            light_action = gr.Textbox(label="Acción")
                            light_color = gr.Textbox(label="Color")
                        lights_go = gr.Button("Aplicar", variant="primary")

                    with gr.Group(visible=False) as oos_group:
                        gr.Markdown(
                            "#### Sin acción\n"
                            "Cuando no hay una intención soportada, Casa Norte no ejecuta nada. "
                            "Prueba con una frase de alarma, clima, lista, calendario, música o luces."
                        )
                with gr.Column(scale=1):
                    state_output = gr.Markdown(render_state([], [], [], {}, None))
                    with gr.Row():
                        clear_state = gr.Button("Limpiar estado", size="sm")
                        complete = gr.Button("Marcar listo", size="sm", variant="primary")

        with gr.Tab("Técnico"):
            gr.Markdown(
                "### Interpretabilidad\n"
                "El mapa muestra la atención promedio de la última capa de BETO hacia cada token. "
                "Es una ayuda para explicar la predicción, no una prueba causal."
            )
            attn_text = gr.Textbox(
                label="Frase para inspeccionar",
                value="cómo va a estar el calor hoy en Monterrey",
            )
            attn_button = gr.Button("Ver mapa de atención", variant="primary")
            attn_image = gr.Image(label="Atención · última capa", type="filepath", interactive=False)
            attn_status = gr.Markdown()

    alarms_state = gr.State([])
    shopping_state = gr.State([])
    events_state = gr.State([])
    lights_state = gr.State({})
    music_state = gr.State(None)
    pending_state = gr.State(None)

    group_components = [
        alarm_group, alarm_query_group, weather_group, lists_group,
        calendar_group, music_group, lights_group, oos_group,
    ]
    field_components = [
        alarm_time, alarm_date, alarm_event,
        weather_place, weather_date,
        list_name, list_item, list_mode,
        cal_event, cal_date, cal_time, cal_person,
        music_song, music_artist, music_playlist,
        light_place, light_action, light_color,
    ]
    state_components = [
        alarms_state, shopping_state, events_state, lights_state, music_state,
    ]
    base_outputs = [
        summary, details, action_output, pending_state,
        *group_components, *field_components, state_output,
    ]
    action_outputs = [action_output, state_output, *state_components]

    understand_inputs = [textbox, *state_components]
    submit.click(understand, inputs=understand_inputs, outputs=base_outputs)
    textbox.submit(understand, inputs=understand_inputs, outputs=base_outputs)

    alarm_save.click(
        save_alarm,
        inputs=[pending_state, alarm_time, alarm_date, alarm_event, *state_components],
        outputs=action_outputs,
    )
    alarm_show.click(
        show_alarms,
        inputs=[pending_state, *state_components],
        outputs=action_outputs,
    )
    weather_go.click(
        query_weather,
        inputs=[pending_state, weather_place, weather_date, *state_components],
        outputs=action_outputs,
    )
    list_go.click(
        apply_list,
        inputs=[pending_state, list_name, list_item, list_mode, *state_components],
        outputs=action_outputs,
    )
    cal_save.click(
        save_event,
        inputs=[pending_state, cal_event, cal_date, cal_time, cal_person, *state_components],
        outputs=action_outputs,
    )
    music_go.click(
        play_music,
        inputs=[pending_state, music_song, music_artist, music_playlist, *state_components],
        outputs=action_outputs,
    )
    lights_go.click(
        apply_lights,
        inputs=[pending_state, light_place, light_action, light_color, *state_components],
        outputs=action_outputs,
    )

    clear.click(
        lambda: (
            "", "", "", "", None,
            *group_updates(None), *blank_fields(),
            render_state([], [], [], {}, None),
            [], [], [], {}, None,
        ),
        inputs=[],
        outputs=[textbox, *base_outputs, *state_components],
    )
    clear_state.click(reset_state, inputs=state_components, outputs=action_outputs)
    complete.click(complete_state, inputs=state_components, outputs=action_outputs)
    attn_button.click(render_attention, inputs=[attn_text], outputs=[attn_image, attn_status])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7860, show_error=True,
        css=CSS, theme=gr.themes.Soft(),
    )
