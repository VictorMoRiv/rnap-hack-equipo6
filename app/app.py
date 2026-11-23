import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from casa_norte.model import load_bundle, plot_attention_map, predict  # noqa: E402
except Exception as error:
    load_bundle = None
    predict = None
    LOAD_ERROR = str(error)
else:
    LOAD_ERROR = None

from casa_norte.nlu import (  # noqa: E402
    CLASS_ES, INTENT_ACTIONS, extract_item, extract_list_name, light_action,
    normalize_date, normalize_time,
)

NEW_MODEL_DIR = ROOT / "modelo_casa_norte"
OLD_MODEL_DIR = ROOT / "models" / "beto_joint"
MODEL_DIR = NEW_MODEL_DIR if NEW_MODEL_DIR.exists() else OLD_MODEL_DIR
THRESHOLD = 0.55
BUNDLE = None

if LOAD_ERROR is None and load_bundle is not None:
    try:
        if MODEL_DIR.exists():
            BUNDLE = load_bundle(MODEL_DIR)
    except Exception as error:
        LOAD_ERROR = str(error)

EXAMPLES = [
    "despiértame dentro de 20 minutos",
    "quita la alarma de las 09:20",
    "agenda una junta con Luis mañana a las 10",
    "cancela la junta con Luis",
    "agrega tortillas y frijoles a la lista del súper",
    "quita las tortillas de la lista",
    "¿va a llover hoy en Monterrey?",
    "apaga las luces del cuarto",
    "transfiere 500 pesos a mi hermana",
]

WEATHER_DEMO = {
    "monterrey": ("28 °C", "Soleado"),
    "guadalajara": ("24 °C", "Parcialmente nublado"),
    "cdmx": ("21 °C", "Nublado"),
    "ciudad de méxico": ("21 °C", "Nublado"),
}

INTENT_STYLES = {
    "alarm_set": {
        "title": "Configurar Alarma",
        "tag": "ALARMA",
        "gradient": "linear-gradient(135deg, #b45309 0%, #dc2626 100%)",
        "border": "#f59e0b",
        "tip": "Revisa la hora y pulsa Guardar alarma para programarla en tu dispositivo.",
    },
    "alarm_query": {
        "title": "Alarmas Guardadas",
        "tag": "CONSULTA",
        "gradient": "linear-gradient(135deg, #b45309 0%, #d97706 100%)",
        "border": "#d97706",
        "tip": "Revisa el listado completo en la pestaña Estado de la Casa.",
    },
    "alarm_remove": {
        "title": "Eliminar Alarma",
        "tag": "REMOVER",
        "gradient": "linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%)",
        "border": "#f87171",
        "tip": "Confirma la hora o etiqueta de la alarma que deseas dar de baja.",
    },
    "weather_query": {
        "title": "Pronóstico del Clima",
        "tag": "CLIMA",
        "gradient": "linear-gradient(135deg, #0284c7 0%, #2563eb 100%)",
        "border": "#38bdf8",
        "tip": "Reporte de temperatura y condiciones meteorológicas.",
    },
    "lists_createoradd": {
        "title": "Lista del Súper / Tareas · Agregar",
        "tag": "COMPRAS",
        "gradient": "linear-gradient(135deg, #059669 0%, #047857 100%)",
        "border": "#34d399",
        "tip": "Verifica el producto o tarea antes de añadirlo a la lista.",
    },
    "lists_remove": {
        "title": "Lista del Súper / Tareas · Quitar",
        "tag": "REMOVER",
        "gradient": "linear-gradient(135deg, #dc2626 0%, #991b1b 100%)",
        "border": "#f87171",
        "tip": "Confirma el elemento que deseas eliminar de la lista.",
    },
    "calendar_set": {
        "title": "Agendar en Calendario",
        "tag": "AGENDA",
        "gradient": "linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%)",
        "border": "#a78bfa",
        "tip": "Revisa los detalles del evento antes de programarlo.",
    },
    "calendar_remove": {
        "title": "Cancelar Evento",
        "tag": "CANCELAR",
        "gradient": "linear-gradient(135deg, #991b1b 0%, #450a0a 100%)",
        "border": "#ef4444",
        "tip": "Confirma el evento de la agenda que deseas cancelar.",
    },
    "play_music": {
        "title": "Reproducción Musical",
        "tag": "MÚSICA",
        "gradient": "linear-gradient(135deg, #db2777 0%, #e11d48 100%)",
        "border": "#f472b6",
        "tip": "Ajusta la canción, artista o abre el enlace en YouTube.",
    },
    "control_luces": {
        "title": "Control de Iluminación",
        "tag": "LUCES",
        "gradient": "linear-gradient(135deg, #d97706 0%, #ea580c 100%)",
        "border": "#fbbf24",
        "tip": "Modifica la habitación, acción o color de la luz.",
    },
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Fondo oscuro general y texto de alto contraste */
body, .gradio-container {
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
  background-color: #0b0f19 !important;
  color: #f8fafc !important;
}

/* Ocultar botones del pie de página de Gradio EXCEPTO 'Runs' */
footer a:not([href*="/runs"]),
footer button,
footer .api-docs {
  display: none !important;
}

footer a[href*="/runs"] {
  display: inline-flex !important;
  visibility: visible !important;
  color: #94a3b8 !important;
  font-size: 0.85rem !important;
  text-decoration: none !important;
  padding: 0.4rem 0.8rem !important;
  border-radius: 6px !important;
  border: 1px solid #2a3756 !important;
}

footer a[href*="/runs"]:hover {
  color: #ffffff !important;
  border-color: #6366f1 !important;
}

/* Encabezado en la esquina superior */
.cn-header-corner {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  padding: 0.6rem 0.2rem 1.2rem 0.2rem;
  border-bottom: 1px solid #1e293b;
  margin-bottom: 1.2rem;
}

.cn-brand-box {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
}

.cn-brand-title {
  font-size: 1.7rem;
  font-weight: 800;
  color: #ffffff !important;
  letter-spacing: -0.02em;
}

.cn-brand-tag {
  font-size: 0.82rem;
  font-weight: 700;
  color: #818cf8 !important;
  background: rgba(99, 102, 241, 0.15);
  border: 1px solid rgba(99, 102, 241, 0.35);
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  letter-spacing: 0.03em;
}

/* Tabs */
.gradio-container .tab-nav {
  border-bottom: 2px solid #2a3756 !important;
  margin-bottom: 1.4rem !important;
  gap: 0.5rem !important;
}

.gradio-container .tab-nav button {
  color: #94a3b8 !important;
  font-size: 1.02rem !important;
  font-weight: 700 !important;
  padding: 0.7rem 1.6rem !important;
  border-radius: 10px 10px 0 0 !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  transition: all 0.2s ease !important;
}

.gradio-container .tab-nav button.selected {
  color: #ffffff !important;
  border: 1px solid #2a3756 !important;
  border-bottom: 3px solid #6366f1 !important;
  background: #151e32 !important;
}

/* Animaciones */
@keyframes cnCardPop {
  0% {
    opacity: 0;
    transform: translateY(12px) scale(0.98);
  }
  65% {
    transform: translateY(-2px) scale(1.005);
  }
  100% {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

/* Entradas de texto */
.gradio-container textarea,
.gradio-container input[type="text"] {
  background-color: #162035 !important;
  color: #ffffff !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
  font-size: 0.98rem !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: #64748b !important;
}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
}

.gradio-container label span {
  color: #e2e8f0 !important;
  font-weight: 700 !important;
  font-size: 0.92rem !important;
}

/* Acordeón Datos detectados */
.gradio-container .accordion {
  background-color: #151e32 !important;
  border: 1px solid #2a3756 !important;
  border-radius: 12px !important;
  color: #e2e8f0 !important;
}

/* Tarjeta Animada */
.cn-card {
  animation: cnCardPop 0.38s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  background: #151e32;
  border-radius: 16px;
  padding: 1.4rem 1.8rem;
  margin: 1rem 0;
  border: 1px solid #2a3756;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.45);
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.cn-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.6);
  border-color: #4f46e5;
}

.cn-card-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
  color: #ffffff !important;
  font-weight: 800;
  font-size: 1.05rem;
  padding: 0.4rem 1.15rem;
  border-radius: 9999px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
  margin-bottom: 1rem;
}

.cn-badge-tag {
  font-size: 0.72rem;
  font-weight: 800;
  background: rgba(0, 0, 0, 0.35);
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  letter-spacing: 0.05em;
  color: #ffffff !important;
}

.cn-card-slots {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin: 0.8rem 0 1.2rem 0;
}

.cn-slot-pill {
  background: #1b263f;
  border: 1px solid #2e3e63;
  border-radius: 10px;
  padding: 0.55rem 1rem;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.95rem;
  transition: transform 0.2s ease, background 0.2s ease;
}

.cn-slot-pill:hover {
  background: #233152;
  border-color: #475569;
  transform: translateY(-2px);
}

.cn-slot-key {
  color: #94a3b8 !important;
  font-weight: 700;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.cn-slot-val {
  color: #ffffff !important;
  font-weight: 700;
}

.cn-card-tip {
  font-size: 0.9rem;
  color: #94a3b8 !important;
  border-top: 1px dashed #2a3756;
  padding-top: 0.8rem;
  margin-top: 0.6rem;
}

/* Alertas de Acción */
.cn-action-banner {
  animation: cnCardPop 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  background: #064e3b;
  border: 1px solid #059669;
  border-radius: 12px;
  padding: 1rem 1.4rem;
  margin: 0.9rem 0;
  color: #ecfdf5 !important;
  font-weight: 700;
  font-size: 1rem;
  box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25);
}

.cn-action-alarm {
  background: linear-gradient(135deg, #1e3a8a 0%, #1e1b4b 100%) !important;
  border: 1px solid #3b82f6 !important;
  color: #eff6ff !important;
}

.cn-action-remove {
  background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%) !important;
  border: 1px solid #ef4444 !important;
  color: #fef2f2 !important;
}

.cn-action-error {
  background: #7f1d1d !important;
  border-color: #dc2626 !important;
  color: #fef2f2 !important;
}

.cn-action-warning {
  background: #78350f !important;
  border-color: #d97706 !important;
  color: #fef3c7 !important;
}

/* Botones */
.gradio-container button.primary,
.gradio-container button[variant="primary"] {
  background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
  transition: all 0.2s ease !important;
}

.gradio-container button.primary:hover,
.gradio-container button[variant="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 18px rgba(79, 70, 229, 0.55) !important;
}

.gradio-container button.secondary,
.gradio-container button:not(.primary):not([variant="primary"]) {
  background: #1e293b !important;
  color: #e2e8f0 !important;
  border: 1px solid #334155 !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}

/* Dashboard Estado de la Casa */
.cn-dashboard {
  background: #151e32;
  border: 1px solid #2a3756;
  border-radius: 18px;
  padding: 1.6rem 1.8rem;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.4);
}

.cn-dash-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.4rem;
  padding-bottom: 0.9rem;
  border-bottom: 1px solid #2a3756;
}

.cn-dash-title {
  font-size: 1.35rem;
  font-weight: 800;
  color: #ffffff !important;
}

.cn-dash-badge {
  font-size: 0.8rem;
  font-weight: 800;
  background: #064e3b;
  color: #6ee7b7 !important;
  border: 1px solid #059669;
  padding: 0.25rem 0.8rem;
  border-radius: 9999px;
  letter-spacing: 0.05em;
}

.cn-dash-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 1rem;
  margin-bottom: 1.6rem;
}

.cn-dash-tile {
  background: #1b263f;
  border: 1px solid #2e3e63;
  border-radius: 14px;
  padding: 1.1rem 1.25rem;
  transition: all 0.25s ease;
}

.cn-dash-tile:hover {
  transform: translateY(-3px);
  border-color: #6366f1;
  box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25);
}

.cn-dash-tile-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 800;
  color: #94a3b8 !important;
  margin-bottom: 0.5rem;
}

.cn-dash-tile-val {
  font-size: 1.05rem;
  font-weight: 700;
  color: #ffffff !important;
  line-height: 1.4;
}

.cn-dash-section {
  background: #1b263f;
  border: 1px solid #2e3e63;
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  margin-top: 1rem;
}

.cn-section-title {
  font-size: 0.95rem;
  font-weight: 800;
  color: #e2e8f0 !important;
  margin-bottom: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.cn-items-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}

.cn-detail-tag {
  background: #162035;
  border: 1px solid #334155;
  color: #f1f5f9 !important;
  padding: 0.4rem 0.8rem;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 600;
}
"""

CSS = CUSTOM_CSS


def clean_value(value):
    if not value:
        return ""
    return str(value).strip(" ¿?¡!.,;:")


def slot_values(result):
    values = {}
    spans = result.get("spans", [])
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


def parse_smart_time_and_date(text):
    """
    Detecta la hora y fecha del sistema para calcular tanto horas absolutas
    como relativas (ej. 'dentro de 20 minutos', 'en media hora', 'en 1 hora', etc.).
    """
    now = datetime.now()
    lowered = text.lower()

    # Minutos relativos: "dentro de 20 minutos", "en 20 minutos", "en 15 mins"
    m_min = re.search(r"(?:en|dentro de)\s+(\d+)\s*(?:minutos?|mins?|m)\b", lowered)
    if m_min:
        mins = int(m_min.group(1))
        target = now + timedelta(minutes=mins)
        return target.strftime("%H:%M"), target.strftime("%Y-%m-%d"), f"Alarma en {mins} min"

    # Horas relativas: "en 2 horas", "dentro de 1 hora"
    m_hr = re.search(r"(?:en|dentro de)\s+(\d+)\s*(?:horas?|hrs?|h)\b", lowered)
    if m_hr:
        hrs = int(m_hr.group(1))
        target = now + timedelta(hours=hrs)
        return target.strftime("%H:%M"), target.strftime("%Y-%m-%d"), f"Alarma en {hrs} h"

    # Fracciones de hora comunes
    if re.search(r"(?:en|dentro de)\s+media\s+hora\b", lowered):
        target = now + timedelta(minutes=30)
        return target.strftime("%H:%M"), target.strftime("%Y-%m-%d"), "Alarma en 30 min"
    if re.search(r"(?:en|dentro de)\s+un\s+cuarto\s+de\s+hora\b", lowered):
        target = now + timedelta(minutes=15)
        return target.strftime("%H:%M"), target.strftime("%Y-%m-%d"), "Alarma en 15 min"
    if re.search(r"(?:en|dentro de)\s+hora\s+y\s+media\b", lowered):
        target = now + timedelta(minutes=90)
        return target.strftime("%H:%M"), target.strftime("%Y-%m-%d"), "Alarma en 90 min"

    # Hora explícita: "a las seis y media", "a las 15:00", "de las 09:20", etc.
    hora = ""
    match_abs = re.search(r"(?:a\s+las?|para\s+las?|de\s+las?)\s+([^,.;]+)", text, flags=re.IGNORECASE)
    if match_abs:
        hora = normalize_time(match_abs.group(1)) or match_abs.group(1).strip()

    # Fecha relativa o absoluta
    fecha = ""
    if "mañana" in lowered and "pasado mañana" not in lowered:
        fecha = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "pasado mañana" in lowered:
        fecha = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    elif "hoy" in lowered:
        fecha = now.strftime("%Y-%m-%d")
    else:
        norm_d = normalize_date(text)
        if norm_d:
            fecha = norm_d
        elif hora:
            try:
                h, m = map(int, hora.split(":"))
                today_alarm = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if today_alarm <= now:
                    fecha = (now + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    fecha = now.strftime("%Y-%m-%d")
            except Exception:
                fecha = now.strftime("%Y-%m-%d")

    return hora or "", fecha or "", "Alarma"


def split_multiple_items(raw_items_str):
    if not raw_items_str:
        return []
    cleaned = re.sub(r"\s+(?:a|en|de)\s+(?:la|mi|el)\s+lista.*$", "", raw_items_str, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:al|del)\s+s[uú]per.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:a|de)\s+compras.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:y\s+tambi[eé]n|y\s+adem[aá]s).*$", "", cleaned, flags=re.IGNORECASE)

    parts = re.split(r",|\s+y\s+|\s+e\s+", cleaned)
    items = []
    for part in parts:
        item = part.strip(" .,")
        item = re.sub(r"^(los|las|el|la|un|una|unos|unas)\s+", "", item, flags=re.IGNORECASE).strip()
        if item and item.lower() not in {"y", "e", "lista", "compras", "súper", "super", "que", "quiero"}:
            items.append(item)
    return items


def guess_music(text):
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
    r"\b(noticias|traducir)\b",
    r"\b(transfiere|transferir|transferencia|pagar|pago)\b.*\b(pesos|dólares|dolares|mxn|usd)\b",
]


def out_of_scope(text):
    lowered = text.lower()
    # Si el usuario pide un recordatorio, alarma, evento o temporizador,
    # la acción principal es un recordatorio/alarma (aunque el contenido mencione un juego, receta, etc.)
    if re.search(r"\b(recuerda|recuérdame|recuerdame|avisa|avísame|avisame|alarma|despierta|despiértame|despiertame|agenda|agendar|junta|reunión|reunion|cita)\b", lowered):
        return False
    if re.search(r"(?:en|dentro de)\s+\d+\s*(?:min|hora)", lowered):
        return False
    # Fuera de alcance si el comando directo pide jugar a un juego
    if re.search(r"^\s*(?:quiero\s+)?(?:jugar|juega|vamos a jugar)\b", lowered):
        return True
    return any(re.search(pattern, lowered) for pattern in OOS_PATTERNS)


def fallback_predict(text):
    lowered = text.lower()
    if out_of_scope(text):
        return None

    # Reconocimiento explícito de eliminación/cancelación
    is_remove = bool(re.search(r"\b(quita|elimina|borra|cancela|quitar|eliminar|borrar|cancelar|apaga|apagar)\b", lowered))

    if is_remove and re.search(r"\b(alarma|alarmas|despertador)\b", lowered):
        intent = "alarm_remove"
    elif is_remove and re.search(r"\b(evento|eventos|junta|juntas|reunión|reunion|cita|citas|agenda)\b", lowered):
        intent = "calendar_remove"
    elif is_remove and (re.search(r"\b(lista|súper|super|compras|despensa|tarea|tareas|artículo|articulo)\b", lowered) or extract_item(text)):
        intent = "lists_remove"
    elif re.search(r"\b(alarma|alarmas|despiertame|despiértame|despertador|recordatorio|avísame|avisame)\b", lowered) or re.search(r"(?:en|dentro de)\s+\d+\s*(?:min|hora)", lowered):
        if re.search(r"\b(que|qué|cuales|cuáles|ver|mis|consultar|tengo)\b", lowered) and not re.search(r"\b(pon|crea|guarda|nueva|a las|en|dentro de)\b", lowered):
            intent = "alarm_query"
        else:
            intent = "alarm_set"
    elif re.search(r"\b(clima|tiempo|llover|lluvia|lloverá|llovera|temperatura|calor|frío|frio|pronóstico|pronostico)\b", lowered):
        intent = "weather_query"
    elif re.search(r"\b(lista|súper|super|compras|despensa|tarea|tareas)\b", lowered) or re.search(r"\b(agrega|añade|anota|apunta|metele)\b", lowered):
        intent = "lists_createoradd"
    elif re.search(r"\b(agenda|agendar|cita|reunión|reunion|junta|evento|calendario)\b", lowered):
        intent = "calendar_set"
    elif re.search(r"\b(música|musica|canción|cancion|rola|canciones|reproduce|reproducir|toca|tocar|playlist)\b", lowered):
        intent = "play_music"
    elif re.search(r"\b(luz|luces|foco|focos|lampara|lámpara|iluminación|iluminacion)\b", lowered) or re.search(r"\b(enciende|prende|baja|sube)\b", lowered):
        intent = "control_luces"
    else:
        if any(w in lowered for w in ["recámara", "recamara", "cuarto", "sala"]):
            intent = "control_luces"
        else:
            intent = "alarm_set"

    slots = {}
    time_val, date_val, default_label = parse_smart_time_and_date(text)
    if time_val:
        slots["time"] = time_val
    if date_val:
        slots["date"] = date_val

    if intent in {"alarm_set", "alarm_remove"}:
        slots["event_name"] = default_label
    elif intent == "weather_query":
        for city in ["monterrey", "guadalajara", "cdmx", "ciudad de méxico", "cancún", "puebla", "tijuana"]:
            if city in lowered:
                slots["place_name"] = city.capitalize()
                break
    elif intent in {"lists_createoradd", "lists_remove"}:
        slots["list_name"] = extract_list_name(text) or "súper / tareas"
        item = extract_item(text)
        if item:
            slots["item"] = item
    elif intent == "play_music":
        playlist = guess_playlist(text)
        if playlist:
            slots["playlist_name"] = playlist
        song = guess_music(text)
        if song:
            slots["song_name"] = song
        m_art = re.search(r"(?:de|por)\s+([A-Za-z0-9\s]+)", text, flags=re.IGNORECASE)
        if m_art and not playlist:
            slots["artist_name"] = m_art.group(1).strip()
    elif intent == "control_luces":
        for room in ["sala", "cuarto", "recámara", "recamara", "cocina", "baño", "comedor", "patio"]:
            if room in lowered:
                slots["house_place"] = room
                break
        for col in ["cálida", "calida", "blanca", "azul", "roja", "verde"]:
            if col in lowered:
                slots["color_type"] = col
                break
    elif intent in {"calendar_set", "calendar_remove"}:
        m_pers = re.search(r"(?:con)\s+([A-ZÁÉÍÓÚ][a-záéíóú]+)", text)
        if m_pers:
            slots["person"] = m_pers.group(1)
        m_ev = re.search(r"(?:junta|reunión|reunion|cita|evento)\s*(?:con\s+[A-Za-z]+)?", text, flags=re.IGNORECASE)
        if m_ev:
            slots["event_name"] = m_ev.group(0).strip()
        else:
            slots["event_name"] = "Reunión"

    spans = [{"type": k, "value": v, "start": 0, "end": 0} for k, v in slots.items()]
    return {
        "intent": intent,
        "confidence": 0.95,
        "slots": slots,
        "spans": spans,
        "top": [(intent, 0.95)],
    }


def schedule_device_alarm(hora_str, fecha_str, label="Alarma"):
    """
    Programa una alarma real en el sistema operativo / dispositivo.
    Espera en segundo plano y emite beeps sonoros directos por los altavoces
    del equipo al llegar el momento exacto, además de abrir la app de Reloj de Windows.
    """
    try:
        now = datetime.now()
        target_date = datetime.strptime(fecha_str, "%Y-%m-%d").date() if fecha_str else now.date()
        h, m = map(int, hora_str.split(":"))
        target_dt = datetime.combine(target_date, datetime.min.time()).replace(hour=h, minute=m)

        diff_seconds = (target_dt - now).total_seconds()
        if diff_seconds < -30:
            if not fecha_str or fecha_str == now.strftime("%Y-%m-%d"):
                target_dt += timedelta(days=1)
                diff_seconds = (target_dt - now).total_seconds()

        if diff_seconds >= 0:
            def ring_alarm():
                if diff_seconds > 0:
                    time.sleep(diff_seconds)
                try:
                    import winsound
                    for _ in range(8):
                        winsound.Beep(950, 280)
                        time.sleep(0.08)
                        winsound.Beep(1350, 380)
                        time.sleep(0.18)
                except Exception:
                    pass

            t = threading.Thread(target=ring_alarm, daemon=True)
            t.start()

            try:
                os.system('start ms-clock:alarm')
            except Exception:
                pass

            min_diff = max(1, int(round(diff_seconds / 60)))
            return True, f"Alarma activa en tu dispositivo para las {hora_str} (sonará en ~{min_diff} min por los altavoces)."
        else:
            return False, "La hora indicada ya pasó el día de hoy."
    except Exception as e:
        return False, f"No se pudo programar el temporizador del dispositivo: {e}"


def render_state(alarms, shopping, events, lights, music):
    alarm_preview = f"{alarms[-1]['hora']} · {alarms[-1]['evento']}" if alarms else "Sin alarmas activas"
    lista_preview = ", ".join(shopping[-3:]) if shopping else "Lista vacía"
    if len(shopping) > 3:
        lista_preview += f" (+{len(shopping)-3} más)"
    event_preview = f"{events[-1]['evento']} ({events[-1]['fecha'] or 'hoy'})" if events else "Sin eventos próximos"
    if lights:
        lights_preview = ", ".join(f"{p}: {s['accion']}" for p, s in list(lights.items())[-2:])
    else:
        lights_preview = "Todo apagado"
    music_preview = music if music else "En silencio"

    alarms_list_html = "".join(
        f'<span class="cn-detail-tag">{item["hora"]} · {item["evento"]} ({item["fecha"] or "sin fecha"})</span>'
        for item in alarms
    ) if alarms else '<span style="color:#94a3b8; font-size:0.9rem;">No hay alarmas configuradas.</span>'

    shopping_list_html = "".join(
        f'<span class="cn-detail-tag">{item}</span>'
        for item in shopping
    ) if shopping else '<span style="color:#94a3b8; font-size:0.9rem;">La lista de tareas/súper está vacía.</span>'

    events_list_html = "".join(
        f'<span class="cn-detail-tag">{e["evento"]} · {e["fecha"] or "hoy"} {e["hora"] or ""} ({e.get("persona") or "general"})</span>'
        for e in events
    ) if events else '<span style="color:#94a3b8; font-size:0.9rem;">No hay eventos en el calendario.</span>'

    lights_list_html = "".join(
        f'<span class="cn-detail-tag">{place}: {data["accion"]} {data.get("color") or ""}</span>'
        for place, data in lights.items()
    ) if lights else '<span style="color:#94a3b8; font-size:0.9rem;">Sin modificaciones de luces activas.</span>'

    return f"""
<div class="cn-dashboard">
  <div class="cn-dash-head">
    <div class="cn-dash-title">Estado de Casa Norte</div>
    <div class="cn-dash-badge">EN LÍNEA</div>
  </div>

  <div class="cn-dash-grid">
    <div class="cn-dash-tile">
      <div class="cn-dash-tile-title">Alarmas ({len(alarms)})</div>
      <div class="cn-dash-tile-val">{alarm_preview}</div>
    </div>
    <div class="cn-dash-tile">
      <div class="cn-dash-tile-title">Tareas y Lista ({len(shopping)})</div>
      <div class="cn-dash-tile-val">{lista_preview}</div>
    </div>
    <div class="cn-dash-tile">
      <div class="cn-dash-tile-title">Eventos ({len(events)})</div>
      <div class="cn-dash-tile-val">{event_preview}</div>
    </div>
    <div class="cn-dash-tile">
      <div class="cn-dash-tile-title">Luces</div>
      <div class="cn-dash-tile-val">{lights_preview}</div>
    </div>
    <div class="cn-dash-tile">
      <div class="cn-dash-tile-title">Música</div>
      <div class="cn-dash-tile-val">{music_preview}</div>
    </div>
  </div>

  <div class="cn-dash-section">
    <div class="cn-section-title">Detalle de Alarmas</div>
    <div class="cn-items-list">{alarms_list_html}</div>
  </div>

  <div class="cn-dash-section">
    <div class="cn-section-title">Tareas y Artículos en la Lista</div>
    <div class="cn-items-list">{shopping_list_html}</div>
  </div>

  <div class="cn-dash-section">
    <div class="cn-section-title">Agenda y Eventos</div>
    <div class="cn-items-list">{events_list_html}</div>
  </div>

  <div class="cn-dash-section">
    <div class="cn-section-title">Iluminación de Habitaciones</div>
    <div class="cn-items-list">{lights_list_html}</div>
  </div>
</div>
"""


def make_card_html(intent, slots_items, tip=""):
    style = INTENT_STYLES.get(intent, {
        "title": intent,
        "tag": "ACCIÓN",
        "gradient": "linear-gradient(135deg, #4f46e5, #6366f1)",
        "border": "#6366f1",
        "tip": tip or "Confirma los datos antes de aplicar la acción.",
    })
    slots_html = "".join(
        f'<div class="cn-slot-pill"><span class="cn-slot-key">{k}:</span><span class="cn-slot-val">{v}</span></div>'
        for k, v in slots_items if v
    ) or '<div class="cn-slot-pill"><span class="cn-slot-val">Sin datos específicos detectados</span></div>'

    card_tip = tip or style.get("tip", "Revisa los campos y confirma la acción.")

    return f"""
<div class="cn-card" style="border-left: 5px solid {style['border']};">
  <div class="cn-card-badge" style="background: {style['gradient']};">
    <span class="cn-badge-tag">{style['tag']}</span>
    <span>{style['title']}</span>
  </div>
  <div class="cn-card-slots">
    {slots_html}
  </div>
  <div class="cn-card-tip">
    <span>{card_tip}</span>
  </div>
</div>
"""


SLOT_NAME_MAP = {
    "time": "Hora",
    "date": "Fecha",
    "event_name": "Etiqueta / Evento",
    "place_name": "Lugar / Ciudad",
    "person": "Persona",
    "list_name": "Nombre de lista",
    "item": "Artículo / Tarea",
    "artist_name": "Artista",
    "song_name": "Canción",
    "playlist_name": "Playlist",
    "house_place": "Habitación",
    "color_type": "Tono / Color",
    "action": "Acción",
}


def understand(text, alarms, shopping, events, lights, music):
    state_md = render_state(alarms, shopping, events, lights, music)
    empty_fields = [gr.update(visible=False, value="", label="Dato") for _ in range(4)]
    text = (text or "").strip()

    if not text:
        return (
            "", "", "", None,
            gr.update(interactive=False, value="Enviar"),
            *empty_fields, "", state_md,
        )

    if out_of_scope(text):
        card = """
<div class="cn-card" style="border-left: 5px solid #64748b;">
  <div class="cn-card-badge" style="background: linear-gradient(135deg, #64748b, #475569);">
    <span class="cn-badge-tag">AVISO</span>
    <span>Fuera de Alcance</span>
  </div>
  <p style="color: #cbd5e1; margin: 0.8rem 0; font-size: 0.95rem;">
    Actualmente Casa Norte maneja: <strong>alarmas, clima, listas/tareas, calendario, música e iluminación</strong>.
  </p>
</div>
"""
        return (
            "", card, "", None,
            gr.update(interactive=False, value="Fuera de alcance"),
            *empty_fields, "", state_md,
        )

    if BUNDLE is not None:
        model, tokenizer, config = BUNDLE
        result = predict(model, tokenizer, text, max_len=config.get("max_len", 40))
        intent = result["intent"]
        confidence = result["confidence"]
        slots = slot_values(result)
        spans = result.get("spans", [])
    else:
        res = fallback_predict(text)
        if res is None:
            intent = "alarm_set"
            confidence = 0.50
            slots = {}
            spans = []
        else:
            intent = res["intent"]
            confidence = res["confidence"]
            slots = res["slots"]
            spans = res["spans"]

    # Revisa si hay tiempo relativo detectado para alarmas
    rel_time, rel_date, rel_label = parse_smart_time_and_date(text)
    if intent in {"alarm_set", "alarm_remove"}:
        if rel_time:
            slots["time"] = rel_time
        if rel_date:
            slots["date"] = rel_date
        lowered = text.lower()
        m_rec = re.search(r"(?:recuérdame|recuerdame|avísame|avisame)\s+(?:de\s+|a\s+|que\s+)?(.+?)(?:\s+(?:en|dentro de|a las?|para las?|el|mañana)\b|$)", lowered)
        if m_rec:
            slots["event_name"] = m_rec.group(1).strip()
        elif rel_label and not slots.get("event_name"):
            slots["event_name"] = rel_label

    # Generar SOLO datos detectados (sin probabilidades)
    if slots:
        details_lines = []
        for k, v in slots.items():
            label = SLOT_NAME_MAP.get(k, k.capitalize())
            details_lines.append(f"- **{label}:** {v}")
        details_md = "\n".join(details_lines)
    else:
        details_md = "No se detectaron datos adicionales en la frase."

    if confidence < THRESHOLD:
        card = """
<div class="cn-card" style="border-left: 5px solid #f59e0b;">
  <div class="cn-card-badge" style="background: linear-gradient(135deg, #b45309, #d97706);">
    <span class="cn-badge-tag">REINTENTO</span>
    <span>Aclaración requerida</span>
  </div>
  <p style="color: #cbd5e1; margin: 0.8rem 0; font-size: 0.95rem;">
    Prueba reformular tu frase con una acción directa como poner una alarma, apagar la luz o agregar a la lista.
  </p>
</div>
"""
        return (
            "", card, details_md, None,
            gr.update(interactive=False, value="Aclarar"),
            *empty_fields, "", state_md,
        )

    pending = {"intent": intent, "text": text, "slots": slots}

    if intent == "alarm_set":
        hora = slots.get("time") or rel_time or ""
        fecha = slots.get("date") or rel_date or ""
        evento = slots.get("event_name") or rel_label or "Alarma"
        card = make_card_html("alarm_set", [("Hora calculada", hora or "por definir"), ("Fecha", fecha or "hoy"), ("Etiqueta", evento)])
        fields = [
            gr.update(label="Hora", value=hora, visible=True),
            gr.update(label="Fecha", value=fecha, visible=True),
            gr.update(label="Etiqueta", value=evento, visible=True),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Guardar alarma en el dispositivo")
    elif intent == "alarm_query":
        items_preview = [("Alarmas registradas", f"{len(alarms)} activas")]
        card = make_card_html("alarm_query", items_preview, tip="Revisa el listado en la pestaña 'Estado de la Casa'.")
        fields = [
            gr.update(label="Hora", value="", visible=False),
            gr.update(label="Fecha", value="", visible=False),
            gr.update(label="Evento", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=False, value="Alarmas consultadas")
    elif intent == "alarm_remove":
        hora = slots.get("time") or rel_time or ""
        evento = slots.get("event_name") or ""
        card = make_card_html("alarm_remove", [("Hora de alarma a quitar", hora or "última configurada"), ("Etiqueta", evento or "cualquiera")])
        fields = [
            gr.update(label="Hora a eliminar", value=hora, visible=True),
            gr.update(label="Etiqueta", value=evento, visible=True),
            gr.update(label="", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Eliminar alarma")
    elif intent == "weather_query":
        lugar = slots.get("place_name") or ""
        fecha = normalize_date(slots.get("date")) or ""
        card = make_card_html("weather_query", [("Ciudad / Lugar", lugar or "tu ciudad"), ("Fecha", fecha or "hoy")])
        fields = [
            gr.update(label="Lugar", value=lugar, visible=True),
            gr.update(label="Fecha", value=fecha, visible=True),
            gr.update(label="", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Consultar clima")
    elif intent == "lists_createoradd":
        lista = slots.get("list_name") or extract_list_name(text) or "súper / tareas"
        raw_item = slots.get("item") or extract_item(text) or ""
        items_list = split_multiple_items(raw_item)
        if len(items_list) > 1:
            items_display = ", ".join(items_list)
            slots_shown = [("Lista", lista), ("Artículos detectados", f"{items_display} ({len(items_list)} ítems)")]
            btn_label = f"Agregar {len(items_list)} artículos a la lista"
            input_val = ", ".join(items_list)
        else:
            items_display = items_list[0] if items_list else (raw_item or "por definir")
            slots_shown = [("Lista", lista), ("Artículo / Tarea", items_display)]
            btn_label = "Agregar a la lista"
            input_val = items_display
        card = make_card_html(intent, slots_shown)
        fields = [
            gr.update(label="Lista", value=lista, visible=True),
            gr.update(label="Artículos (separados por comas)", value=input_val, visible=True),
            gr.update(label="", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value=btn_label)
    elif intent == "lists_remove":
        lista = slots.get("list_name") or extract_list_name(text) or "súper / tareas"
        raw_item = slots.get("item") or extract_item(text) or ""
        items_list = split_multiple_items(raw_item)
        if len(items_list) > 1:
            items_display = ", ".join(items_list)
            slots_shown = [("Lista", lista), ("Artículos a quitar", f"{items_display} ({len(items_list)} ítems)")]
            btn_label = f"Eliminar {len(items_list)} artículos de la lista"
            input_val = ", ".join(items_list)
        else:
            items_display = items_list[0] if items_list else (raw_item or "último agregado")
            slots_shown = [("Lista", lista), ("Artículo / Tarea a quitar", items_display)]
            btn_label = "Eliminar de la lista"
            input_val = items_display
        card = make_card_html(intent, slots_shown)
        fields = [
            gr.update(label="Lista", value=lista, visible=True),
            gr.update(label="Artículos a quitar (separados por comas)", value=input_val, visible=True),
            gr.update(label="", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value=btn_label)
    elif intent == "calendar_set":
        evento = slots.get("event_name") or ""
        fecha = slots.get("date") or rel_date or ""
        hora = slots.get("time") or rel_time or ""
        persona = slots.get("person") or ""
        card = make_card_html("calendar_set", [
            ("Evento", evento or "Reunión"),
            ("Fecha", fecha or "próximamente"),
            ("Hora", hora or "por definir"),
            ("Con", persona or "sin contacto"),
        ])
        fields = [
            gr.update(label="Evento", value=evento, visible=True),
            gr.update(label="Fecha", value=fecha, visible=True),
            gr.update(label="Hora", value=hora, visible=True),
            gr.update(label="Persona", value=persona, visible=True),
        ]
        button = gr.update(interactive=True, value="Guardar en agenda")
    elif intent == "calendar_remove":
        evento = slots.get("event_name") or ""
        persona = slots.get("person") or ""
        card = make_card_html("calendar_remove", [
            ("Evento a cancelar", evento or "último agendado"),
            ("Persona", persona or "cualquiera"),
        ])
        fields = [
            gr.update(label="Evento a cancelar", value=evento, visible=True),
            gr.update(label="Persona", value=persona, visible=True),
            gr.update(label="", value="", visible=False),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Cancelar evento")
    elif intent == "play_music":
        artista = slots.get("artist_name") or ""
        playlist = slots.get("playlist_name") or guess_playlist(text) or ""
        cancion = slots.get("song_name") or ""
        guessed = guess_music(text) if not artista and not playlist else None
        if guessed and (not cancion or cancion.lower() in guessed.lower()):
            cancion = guessed
        card = make_card_html("play_music", [
            ("Canción", cancion or "sugerencia"),
            ("Artista", artista or "varios"),
            ("Playlist", playlist or "predeterminada"),
        ])
        fields = [
            gr.update(label="Canción", value=cancion, visible=True),
            gr.update(label="Artista", value=artista, visible=True),
            gr.update(label="Playlist", value=playlist, visible=True),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Reproducir música")
    else:  # control_luces
        lugar = slots.get("house_place") or "sala"
        accion = light_action(text)
        color = slots.get("color_type") or ""
        card = make_card_html("control_luces", [("Lugar", lugar), ("Acción", accion), ("Tono / Color", color or "normal")])
        fields = [
            gr.update(label="Lugar", value=lugar, visible=True),
            gr.update(label="Acción", value=accion, visible=True),
            gr.update(label="Color", value=color, visible=True),
            gr.update(label="", value="", visible=False),
        ]
        button = gr.update(interactive=True, value="Actualizar luces")

    return "", card, details_md, pending, button, *fields, "", state_md


def apply_action(pending, field1, field2, field3, field4, alarms, shopping, events, lights, music):
    if not pending:
        state_md = render_state(alarms, shopping, events, lights, music)
        return (
            '<div class="cn-action-banner cn-action-warning">Primero ingresa una orden y pulsa Enviar.</div>',
            state_md, alarms, shopping, events, lights, music,
        )

    intent = pending["intent"]
    alarms = list(alarms)
    shopping = list(shopping)
    events = list(events)
    lights = dict(lights)

    if intent == "alarm_set":
        if not field1:
            msg = '<div class="cn-action-banner cn-action-error">Falta especificar la hora de la alarma.</div>'
        else:
            alarms.append({"hora": field1, "fecha": field2, "evento": field3 or "Alarma"})
            ok, alarm_status = schedule_device_alarm(field1, field2, field3 or "Alarma")
            msg = f'<div class="cn-action-banner cn-action-alarm">Alarma configurada: <strong>{field1}</strong> ({field3 or "Alarma"}). {alarm_status}</div>'
    elif intent == "alarm_query":
        msg = f'<div class="cn-action-banner">Tienes <strong>{len(alarms)}</strong> alarma(s) configurada(s). Consulta la pestaña "Estado de la Casa".</div>'
    elif intent == "alarm_remove":
        if not alarms:
            msg = '<div class="cn-action-banner cn-action-warning">No hay alarmas configuradas para eliminar.</div>'
        else:
            target_hora = (field1 or "").strip()
            target_tag = (field2 or "").strip().lower()
            removed = None
            if target_hora:
                for a in list(alarms):
                    if target_hora in a.get("hora", ""):
                        alarms.remove(a)
                        removed = a
                        break
            if not removed and target_tag:
                for a in list(alarms):
                    if target_tag in a.get("evento", "").lower():
                        alarms.remove(a)
                        removed = a
                        break
            if not removed:
                removed = alarms.pop()
            msg = f'<div class="cn-action-banner cn-action-remove">Alarma eliminada: <strong>{removed["evento"]}</strong> ({removed["hora"]}).</div>'
    elif intent == "weather_query":
        place = (field1 or "tu ciudad").strip()
        key = place.lower()
        temp, sky = WEATHER_DEMO.get(key, ("24 °C", "Agradable"))
        msg = f'<div class="cn-action-banner">Clima en <strong>{place}</strong> ({field2 or "hoy"}): <strong>{temp}</strong>, {sky}.</div>'
    elif intent == "lists_createoradd":
        items = split_multiple_items(field2 or "")
        if not items:
            msg = '<div class="cn-action-banner cn-action-error">Indica el artículo o tarea que deseas agregar.</div>'
        else:
            added_tags = []
            for it in items:
                shopping.append(it)
                added_tags.append(f"<strong>{it}</strong>")
            if len(items) > 1:
                items_str = ", ".join(added_tags[:-1]) + " y " + added_tags[-1]
                msg = f'<div class="cn-action-banner">Agregados {len(items)} artículos a la lista <strong>{field1 or "súper / tareas"}</strong>: {items_str}.</div>'
            else:
                msg = f'<div class="cn-action-banner">Agregado <strong>{items[0]}</strong> a la lista <strong>{field1 or "súper / tareas"}</strong>.</div>'
    elif intent == "lists_remove":
        if not shopping:
            msg = '<div class="cn-action-banner cn-action-warning">La lista está vacía; no hay elementos para quitar.</div>'
        else:
            items = split_multiple_items(field2 or "")
            if not items:
                removed = shopping.pop()
                msg = f'<div class="cn-action-banner cn-action-remove">Removido <strong>{removed}</strong> de la lista de tareas/súper.</div>'
            else:
                removed_list = []
                not_found = []
                for it in items:
                    it_low = it.lower()
                    found = None
                    for existing in list(shopping):
                        if it_low == existing.lower() or it_low in existing.lower() or existing.lower() in it_low:
                            shopping.remove(existing)
                            found = existing
                            break
                    if found:
                        removed_list.append(f"<strong>{found}</strong>")
                    else:
                        not_found.append(it)
                parts = []
                if removed_list:
                    parts.append(f"Removido(s) {len(removed_list)} artículo(s): {', '.join(removed_list)}")
                if not_found:
                    parts.append(f"no encontrado(s): {', '.join(not_found)}")
                msg = f'<div class="cn-action-banner cn-action-remove">{"; ".join(parts)}.</div>'
    elif intent == "calendar_set":
        if not field1:
            msg = '<div class="cn-action-banner cn-action-error">Falta el nombre del evento.</div>'
        else:
            events.append({"evento": field1, "fecha": field2, "hora": field3, "persona": field4})
            msg = f'<div class="cn-action-banner">Evento agendado: <strong>{field1}</strong> ({field2 or "fecha próxima"}).</div>'
    elif intent == "calendar_remove":
        if not events:
            msg = '<div class="cn-action-banner cn-action-warning">No hay eventos guardados para cancelar.</div>'
        else:
            target_ev = (field1 or "").strip().lower()
            target_pers = (field2 or "").strip().lower()
            removed = None
            for e in list(events):
                if target_ev and target_ev in e.get("evento", "").lower():
                    events.remove(e)
                    removed = e
                    break
                elif target_pers and target_pers in e.get("persona", "").lower():
                    events.remove(e)
                    removed = e
                    break
            if not removed:
                removed = events.pop()
            msg = f'<div class="cn-action-banner cn-action-remove">Evento cancelado: <strong>{removed["evento"]}</strong> ({removed["fecha"] or "próximo"}).</div>'
    elif intent == "play_music":
        query = " ".join(part for part in [field1, field2, field3] if part).strip()
        if not query:
            msg = '<div class="cn-action-banner cn-action-error">Falta especificar la canción o artista.</div>'
        else:
            music = query
            link = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
            msg = f'<div class="cn-action-banner">Reproduciendo <strong>{query}</strong> · <a href="{link}" target="_blank" style="color:#a7f3d0; text-decoration:underline;">Abrir en YouTube</a></div>'
    else:  # control_luces
        place = field1 or "sala"
        action = field2 or "encender"
        color = field3 or ""
        lights[place] = {"accion": action, "color": color}
        msg = f'<div class="cn-action-banner">Luces de <strong>{place}</strong>: {action} {color}.</div>'

    state_md = render_state(alarms, shopping, events, lights, music)
    return msg, state_md, alarms, shopping, events, lights, music


def clear_all():
    empty = render_state([], [], [], {}, None)
    return (
        "", "", "", None, gr.update(interactive=False, value="Enviar"),
        gr.update(visible=False, value=""), gr.update(visible=False, value=""),
        gr.update(visible=False, value=""), gr.update(visible=False, value=""),
        "", empty, [], [], [], {}, None,
    )


def render_attention(text):
    if BUNDLE is None:
        return None, f"Modelo no disponible: {LOAD_ERROR or 'bundle no cargado'}"
    sample = (text or "").strip() or "cómo va a estar el calor hoy en Monterrey"
    model, tokenizer, _ = BUNDLE
    target = ROOT / "notebooks" / "figs" / "attention_app.png"
    path, _, _ = plot_attention_map(sample, model, tokenizer, output_path=target)
    return path, f"Mapa de atención de la última capa para: **{sample}**"


theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="amber",
    neutral_hue="slate",
)

with gr.Blocks(title="Casa Norte", css=CUSTOM_CSS, theme=theme) as demo:
    # Encabezado en la esquina SIN la hora local en pantalla
    gr.HTML("""
<div class="cn-header-corner">
  <div class="cn-brand-box">
    <span class="cn-brand-title">Casa Norte</span>
    <span class="cn-brand-tag">Asistente</span>
  </div>
</div>
""")

    with gr.Tabs() as main_tabs:
        with gr.Tab("Asistente y Acciones", id="tab_assistant"):
            with gr.Row():
                textbox = gr.Textbox(
                    label="¿Qué necesitas hacer?",
                    placeholder="Ej. despiértame dentro de 20 minutos, quita la alarma, cancela la junta...",
                    scale=5,
                )
                submit = gr.Button("Enviar", variant="primary", scale=1)
                clear = gr.Button("Limpiar", scale=1)

            gr.Examples(
                examples=[[item] for item in EXAMPLES],
                inputs=[textbox],
                label="Frases de ejemplo",
            )

            summary = gr.HTML(visible=False)
            card = gr.HTML()

            with gr.Accordion("Datos detectados", open=False):
                details = gr.Markdown()

            with gr.Row():
                field1 = gr.Textbox(label="Dato 1", visible=False)
                field2 = gr.Textbox(label="Dato 2", visible=False)
                field3 = gr.Textbox(label="Dato 3", visible=False)
                field4 = gr.Textbox(label="Dato 4", visible=False)

            apply_button = gr.Button("Aplicar acción", variant="primary", interactive=False)
            action_output = gr.HTML()

        with gr.Tab("Estado de la Casa", id="tab_status"):
            state_output = gr.HTML(render_state([], [], [], {}, None))

        with gr.Tab("Técnico", id="tab_technical"):
            gr.Markdown("Texto → intención + slots → atención de la última capa de BETO.")
            attention_text = gr.Textbox(
                label="Frase para inspeccionar",
                value="cómo va a estar el calor hoy en Monterrey",
            )
            attention_button = gr.Button("Ver atención", variant="secondary")
            attention_image = gr.Image(label="Atención promedio", type="filepath")
            attention_note = gr.Markdown()

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
    attention_button.click(
        render_attention,
        inputs=[attention_text],
        outputs=[attention_image, attention_note],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=theme,
        css=CUSTOM_CSS,
    )
