import sys
from pathlib import Path
from urllib.parse import quote_plus

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casa_norte.model import load_bundle, predict  # noqa: E402
from casa_norte.nlu import (  # noqa: E402
    CLASS_ES, INTENT_ACTIONS, extract_item, extract_list_name, light_action,
    normalize_date, normalize_time,
)

MODEL_DIR = ROOT / "models" / "beto_joint"
THRESHOLD = 0.55
BUNDLE = None
LOAD_ERROR = None

try:
    if MODEL_DIR.exists():
        BUNDLE = load_bundle(MODEL_DIR)
except Exception as error:
    LOAD_ERROR = str(error)

EXAMPLES = [
    "despiértame mañana a las seis y media",
    "¿va a llover hoy en Monterrey?",
    "agrega tortillas y frijoles a la lista del súper",
    "quita los huevos de la lista de compras",
    "agenda una junta con Luis el martes a las 10",
    "pon música de Bad Bunny",
    "apaga las luces del cuarto",
    "pon la luz cálida en la recámara",
]

WEATHER_DEMO = {
    "monterrey": ("28 °C", "soleado"),
    "guadalajara": ("24 °C", "parcialmente nublado"),
    "cdmx": ("21 °C", "nublado"),
    "ciudad de méxico": ("21 °C", "nublado"),
}


def clean_value(value):
    if not value:
        return ""
    return value.strip(" ¿?¡!.,;:")


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
    import re
    match = re.search(r"(?:a\s+las?|para\s+las?)\s+([^,.;]+)", text, flags=re.IGNORECASE)
    if match:
        return normalize_time(match.group(1))
    return None


def guess_date(text):
    return normalize_date(text)


def guess_music(text):
    import re
    match = re.search(r"(?:reproduce|toca)\s+(.+)", text, flags=re.IGNORECASE)
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
    import re
    match = re.search(r"playlist\s+(?:de\s+)?(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .,") or None
    return None


OOS_PATTERNS = [
    r"\b(dólar|dolar|cotización|tipo de cambio)\b",
    r"\b(pizza|pide|pedir|domicilio)\b",
    r"\b(correo|email|correos)\b",
    r"\b(chiste|chistes)\b",
    r"\b(película|pelicula|películas|peliculas)\b",
    r"\b(receta|recetas|cocinar)\b",
    r"\b(noticias|jugar|juego|traducir)\b",
]


def out_of_scope(text):
    import re
    lowered = text.lower()
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
        f"Entrena el Transformer con `python notebooks/fase2_transformer.py` y coloca la salida en `{MODEL_DIR}`."
        f"{detail}"
    )


def understand(text, alarms, shopping, events, lights, music):
    state_md = render_state(alarms, shopping, events, lights, music)
    empty_fields = [gr.update(visible=False, value="", label="Dato") for _ in range(4)]
    if BUNDLE is None:
        return (
            "No hay modelo cargado.", no_model_message(), "", None,
            gr.update(interactive=False, value="Sin modelo"),
            *empty_fields, "", state_md,
        )
    text = (text or "").strip()
    if not text:
        return (
            "Escribe una frase para Casa Norte.", "", "", None,
            gr.update(interactive=False, value="Entender"),
            *empty_fields, "", state_md,
        )

    if out_of_scope(text):
        summary = "**Fuera de alcance:** eso no es una acción de Casa Norte (demo)."
        card = (
            "### Todavía no puedo hacer eso\n"
            "Puedo con alarma, clima, lista, calendario, música y luces. "
            "Un clasificador `out_of_scope` entrenado es trabajo de la siguiente fase."
        )
        return (
            summary, card, "", None,
            gr.update(interactive=False, value="Fuera de alcance"),
            *empty_fields, "", state_md,
        )

    model, tokenizer, config = BUNDLE
    result = predict(model, tokenizer, text, max_len=config.get("max_len", 40))
    intent = result["intent"]
    confidence = result["confidence"]
    slots = slot_values(result)
    details = "\n".join(
        f"- `{span['type']}`: {span['value']}" for span in result["spans"]
    ) or "(sin slots detectados)"
    details_md = (
        "#### Slots detectados\n"
        f"{details}\n\n"
        "#### Probabilidades\n"
        + "\n".join(f"- {name}: {prob:.3f}" for name, prob in result["top"])
    )

    if confidence < THRESHOLD:
        summary = f"No estoy seguro. La mejor opción es **{CLASS_ES[intent]}** ({confidence:.2f})."
        card = "### Necesito que me lo digas de otra forma\nPrueba con una acción de Casa Norte: alarma, clima, lista, calendario, música o luces."
        return (
            summary, card, details_md, None,
            gr.update(interactive=False, value="Aclarar"),
            *empty_fields, "", state_md,
        )

    summary = f"**Entendí:** {CLASS_ES[intent]} · confianza **{confidence:.2f}**"
    pending = {"intent": intent, "text": text, "slots": slots}

    if intent == "alarm_set":
        hora = normalize_time(slots.get("time")) or guess_time(text) or ""
        fecha = normalize_date(slots.get("date")) or guess_date(text) or ""
        evento = slots.get("event_name") or "Alarma"
        card = (
            "### Tarjeta de alarma\n"
            f"- Hora: **{hora or 'por definir'}**\n"
            f"- Fecha: **{fecha or 'sin fecha'}**\n"
            f"- Evento: **{evento}**\n\n"
            "Edita los campos y guarda la alarma."
        )
        fields = [
            gr.update(label="Hora", value=hora, visible=True),
            gr.update(label="Fecha", value=fecha, visible=True),
            gr.update(label="Evento", value=evento, visible=True),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Guardar alarma")
    elif intent == "alarm_query":
        card = "### Tus alarmas\n" + (
            "\n".join(f"- {item['hora']} · {item['fecha']} · {item['evento']}" for item in alarms)
            if alarms else "Todavía no hay alarmas guardadas."
        )
        fields = [
            gr.update(label="Hora", value="", visible=False),
            gr.update(label="Fecha", value="", visible=False),
            gr.update(label="Evento", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=False, value="Ver alarmas")
    elif intent == "weather_query":
        lugar = slots.get("place_name") or ""
        fecha = normalize_date(slots.get("date")) or ""
        card = (
            "### Tarjeta de clima\n"
            f"- Lugar: **{lugar or 'por definir'}**\n"
            f"- Fecha: **{fecha or 'hoy'}**\n\n"
            "Consulta el pronóstico demo."
        )
        fields = [
            gr.update(label="Lugar", value=lugar, visible=True),
            gr.update(label="Fecha", value=fecha, visible=True),
            gr.update(label="", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Consultar clima")
    elif intent in {"lists_createoradd", "lists_remove"}:
        lista = slots.get("list_name") or extract_list_name(text) or "compras"
        item = extract_item(text) or ""
        accion = "Agregar" if intent == "lists_createoradd" else "Quitar"
        card = (
            f"### Tarjeta de lista · {accion}\n"
            f"- Lista: **{lista}**\n"
            f"- Artículo: **{item or 'por definir'}**\n\n"
            "Edita el artículo y aplica el cambio."
        )
        fields = [
            gr.update(label="Lista", value=lista, visible=True),
            gr.update(label="Artículo", value=item, visible=True),
            gr.update(label="", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value=accion)
    elif intent == "calendar_set":
        evento = slots.get("event_name") or ""
        fecha = normalize_date(slots.get("date")) or ""
        hora = normalize_time(slots.get("time")) or ""
        persona = slots.get("person") or ""
        card = (
            "### Tarjeta de calendario\n"
            f"- Evento: **{evento or 'por definir'}**\n"
            f"- Fecha: **{fecha or 'sin fecha'}**\n"
            f"- Hora: **{hora or 'sin hora'}**\n"
            f"- Persona: **{persona or 'sin persona'}**\n\n"
            "Edita los campos y guarda el evento."
        )
        fields = [
            gr.update(label="Evento", value=evento, visible=True),
            gr.update(label="Fecha", value=fecha, visible=True),
            gr.update(label="Hora", value=hora, visible=True),
            gr.update(label="Persona", value=persona, visible=True),
        ]
        button = gr.update(interactive=True, value="Guardar evento")
    elif intent == "play_music":
        artista = slots.get("artist_name") or ""
        playlist = slots.get("playlist_name") or guess_playlist(text) or ""
        cancion = slots.get("song_name") or ""
        guessed = guess_music(text) if not artista and not playlist else None
        if guessed and (not cancion or cancion.lower() in guessed.lower()):
            cancion = guessed
        cancion = cancion or ""
        card = (
            "### Tarjeta de música\n"
            f"- Canción: **{cancion or 'sin canción'}**\n"
            f"- Artista: **{artista or 'sin artista'}**\n"
            f"- Playlist: **{playlist or 'sin playlist'}**\n\n"
            "Reproduce la búsqueda en YouTube o ajusta los campos."
        )
        fields = [
            gr.update(label="Canción", value=cancion, visible=True),
            gr.update(label="Artista", value=artista, visible=True),
            gr.update(label="Playlist", value=playlist, visible=True),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Reproducir")
    else:
        lugar = slots.get("house_place") or ""
        accion = light_action(text)
        color = slots.get("color_type") or ""
        card = (
            "### Tarjeta de luces\n"
            f"- Lugar: **{lugar or 'por definir'}**\n"
            f"- Acción: **{accion}**\n"
            f"- Color: **{color or 'sin cambio'}**\n\n"
            "Ajusta los campos y aplica el cambio."
        )
        fields = [
            gr.update(label="Lugar", value=lugar, visible=True),
            gr.update(label="Acción", value=accion, visible=True),
            gr.update(label="Color", value=color, visible=True),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Aplicar")

    return summary, card, details_md, pending, button, *fields, "", state_md


def apply_action(pending, field1, field2, field3, field4, alarms, shopping, events, lights, music):
    if not pending:
        state_md = render_state(alarms, shopping, events, lights, music)
        return "Primero escribe una frase y pulsa Entender.", state_md, alarms, shopping, events, lights, music
    intent = pending["intent"]
    alarms = list(alarms)
    shopping = list(shopping)
    events = list(events)
    lights = dict(lights)
    if intent == "alarm_set":
        if not field1:
            message = "Falta la hora de la alarma."
        else:
            alarms.append({"hora": field1, "fecha": field2, "evento": field3 or "Alarma"})
            message = f"Alarma guardada: {field3 or 'Alarma'} a las {field1} ({field2 or 'sin fecha'})."
    elif intent == "alarm_query":
        message = f"Tienes {len(alarms)} alarmas guardadas."
    elif intent == "weather_query":
        place = (field1 or "tu ciudad").strip()
        key = place.lower()
        temp, sky = WEATHER_DEMO.get(key, ("24 °C", "parcialmente nublado"))
        message = f"Pronóstico demo para {place} ({field2 or 'hoy'}): {temp}, {sky}."
    elif intent == "lists_createoradd":
        if not field2:
            message = "Falta el artículo que quieres agregar."
        else:
            shopping.append(field2)
            message = f"Agregado **{field2}** a la lista **{field1 or 'compras'}**."
    elif intent == "lists_remove":
        if not field2:
            message = "Falta el artículo que quieres quitar."
        elif field2 in shopping:
            shopping.remove(field2)
            message = f"Quitado **{field2}** de la lista **{field1 or 'compras'}**."
        else:
            message = f"**{field2}** no estaba en la lista; no hice cambios."
    elif intent == "calendar_set":
        if not field1:
            message = "Falta el nombre del evento."
        else:
            events.append({"evento": field1, "fecha": field2, "hora": field3, "persona": field4})
            message = f"Evento guardado: **{field1}** ({field2 or 'sin fecha'}, {field3 or 'sin hora'})."
    elif intent == "play_music":
        query = " ".join(part for part in [field1, field2, field3] if part).strip()
        if not query:
            message = "Falta la canción, artista o playlist."
        else:
            music = query
            link = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
            message = f"Reproduciendo **{query}** en modo demo. [Abrir búsqueda en YouTube]({link})"
    else:
        place = field1 or "sala"
        action = field2 or "encender"
        color = field3 or ""
        lights[place] = {"accion": action, "color": color}
        message = f"Luces de **{place}**: {action} {color}".rstrip() + "."
    state_md = render_state(alarms, shopping, events, lights, music)
    return message, state_md, alarms, shopping, events, lights, music


def clear_all():
    empty = render_state([], [], [], {}, None)
    return (
        "", "", "", None, gr.update(interactive=False, value="Entender"),
        gr.update(visible=False, value=""), gr.update(visible=False, value=""),
        gr.update(visible=False, value=""), gr.update(visible=False, value=""),
        "", empty, [], [], [], {}, None,
    )


with gr.Blocks(title="Casa Norte · 8 intenciones") as demo:
    gr.Markdown(
        "# Casa Norte\n"
        "Escribe una frase como si hablaras con tu casa. El modelo detecta la intención y los datos clave, "
        "y muestra una tarjeta editable antes de ejecutar la acción."
    )
    with gr.Row():
        textbox = gr.Textbox(
            label="¿Qué necesitas?",
            placeholder="Ej. despiértame mañana a las seis y media",
            scale=4,
        )
        submit = gr.Button("Entender", variant="primary", scale=1)
        clear = gr.Button("Limpiar", scale=1)
    gr.Examples(examples=[[item] for item in EXAMPLES], inputs=[textbox])

    summary = gr.Markdown()
    card = gr.Markdown()
    with gr.Accordion("Ver slots y probabilidades", open=False):
        details = gr.Markdown()
    with gr.Row():
        field1 = gr.Textbox(label="Dato 1", visible=False)
        field2 = gr.Textbox(label="Dato 2", visible=False)
        field3 = gr.Textbox(label="Dato 3", visible=False)
        field4 = gr.Textbox(label="Dato 4", visible=False)
    apply_button = gr.Button("Aplicar acción", interactive=False)
    action_output = gr.Markdown()
    state_output = gr.Markdown(render_state([], [], [], {}, None))

    alarms_state = gr.State([])
    shopping_state = gr.State([])
    events_state = gr.State([])
    lights_state = gr.State({})
    music_state = gr.State(None)
    pending_state = gr.State(None)

    outputs = [
        summary, card, details, pending_state, apply_button,
        field1, field2, field3, field4, action_output, state_output,
    ]
    submit.click(
        understand,
        inputs=[textbox, alarms_state, shopping_state, events_state, lights_state, music_state],
        outputs=outputs,
    )
    textbox.submit(
        understand,
        inputs=[textbox, alarms_state, shopping_state, events_state, lights_state, music_state],
        outputs=outputs,
    )
    apply_button.click(
        apply_action,
        inputs=[
            pending_state, field1, field2, field3, field4,
            alarms_state, shopping_state, events_state, lights_state, music_state,
        ],
        outputs=[
            action_output, state_output,
            alarms_state, shopping_state, events_state, lights_state, music_state,
        ],
    )
    clear.click(
        clear_all,
        inputs=[],
        outputs=[
            textbox, summary, card, pending_state, apply_button,
            field1, field2, field3, field4, action_output, state_output,
            alarms_state, shopping_state, events_state, lights_state, music_state,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
