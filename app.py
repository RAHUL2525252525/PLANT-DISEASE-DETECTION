"""
AI Plant Doctor 🌿  (v5 - new light theme)
================================================
Files:  app.py  +  knowledge.py  (+ your trained .keras / .h5 model)
Optional: .streamlit/config.toml  (light base theme, see config.toml)

Features (all kept from v3/v4, logic unchanged)
  • Camera capture / upload  → trained Keras model prediction (demo fallback)
  • Cure, medicines (Flipkart / Amazon links), organic options, prevention
  • PDF report, WhatsApp share, care-plan calendar (.ics)
  • Photo-quality check, risk level, confidence gauge, AI focus heat-map
  • Batch scan (field health score), Weather advisor (7-day fungal risk
    + safe spray windows), Disease library, smart chatbot with quick chips
  • Field/plot tracking, scan gallery, trend & recovery tracker
  • 4 light colour themes, large-text mode, English / Kannada / Hindi
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import re
import io
import json
import math
import uuid
import random
import hashlib
import shutil
import textwrap
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus, quote

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black, white, HexColor
from reportlab.lib.utils import ImageReader

from knowledge import disease_treatments

# Optional: Text-to-Speech (only speaks on the machine running the app)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ---------------- Page Config ----------------
st.set_page_config(page_title="AI Plant Doctor", page_icon="🌿", layout="wide",
                   initial_sidebar_state="expanded")

# Newer Streamlit replaced use_container_width with width="stretch"
_m = re.match(r"(\d+)\.(\d+)", st.__version__)
_ST_VER = (int(_m.group(1)), int(_m.group(2))) if _m else (1, 0)
STRETCH = {"width": "stretch"} if _ST_VER >= (1, 50) else {"use_container_width": True}

HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
SCANS_DIR = os.path.join(BASE_DIR, "scans")
RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC

RISK_ICON = {"healthy": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}
RISK_SCORE = {"healthy": 0, "moderate": 1, "high": 2, "critical": 3}
HEALTH_POINTS = {"healthy": 100, "moderate": 55, "high": 25, "critical": 0}
EXPERT_HTML = ("<div class='help-card'>📞 <b>Need an expert?</b> Kisan Call Centre (India, toll-free): "
               "<b>1800-180-1551</b> · or visit your nearest Krishi Vigyan Kendra (KVK).</div>")

# ---------------- THEMES (all light) ----------------
# acc = bright accent, acc2 = deep accent (text/buttons), bg = three soft page tints
THEMES = {
    "🌿 Fresh Green": dict(acc="#22b573", acc2="#0b6b45", bg=("#f4fbf7", "#e2f6ea", "#fbfffc")),
    "🌊 Sky Blue":    dict(acc="#2f9bf0", acc2="#0b57a8", bg=("#f3f9ff", "#dfeefc", "#fbfdff")),
    "🌅 Peach":       dict(acc="#ff8f4a", acc2="#b84a0c", bg=("#fff9f3", "#ffeadb", "#fffdfa")),
    "💜 Lavender":    dict(acc="#8f6bf2", acc2="#5128b8", bg=("#faf7ff", "#ece4ff", "#fefdff")),
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


# Small leaf logos (inline SVG, no external files needed)
LEAF_LOGO = ("<svg viewBox='0 0 200 200' width='26' height='26'>"
             "<path d='M100 12C170 42 188 112 100 188C12 112 30 42 100 12Z' fill='#ffffff'/>"
             "<path d='M100 40V176M100 80L70 62M100 112L134 90M100 144L68 124' "
             "style='stroke:var(--acc2)' stroke-width='9' fill='none' stroke-linecap='round'/></svg>")
LEAF_HERO = ("<svg class='hero-leaf' viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'>"
             "<path d='M100 8C176 40 192 114 100 192C8 114 24 40 100 8Z' fill='rgba(255,255,255,.18)'/>"
             "<path d='M100 30V190M100 72L62 50M100 104L142 78M100 136L58 112M100 164L136 142' "
             "stroke='rgba(255,255,255,.6)' stroke-width='4' fill='none' stroke-linecap='round'/></svg>")


# ---------------- TRANSLATIONS ----------------
TRANSLATIONS = {
    "en": {
        "title": "AI Plant Doctor",
        "subtitle": "Snap a leaf. Get a diagnosis, cure and medicines in seconds.",
        "analyze": "🔍 Analyze Leaf", "medicines": "Recommended Medicines",
        "treatment": "Suggested Treatment", "suggestions": "Additional Suggestions",
        "nutrients": "Nutrient Focus", "clear_history": "Clear History",
        "download": "Download CSV",
        "low_confidence": "⚠️ Low confidence: please confirm the diagnosis visually or retake the photo.",
        "top_predictions": "Other possibilities", "alert_title": "🚨 Farmer Alert System",
        "langs_line": "Available in English, Kannada and Hindi.",
    },
    "kn": {
        "title": "ಎಐ ಗಿಡ ವೈದ್ಯ",
        "subtitle": "ಎಲೆಯ ಫೋಟೋ ತೆಗೆಯಿರಿ. ರೋಗ, ಚಿಕಿತ್ಸೆ ಮತ್ತು ಔಷಧಿ ಪಡೆಯಿರಿ.",
        "analyze": "🔍 ವಿಶ್ಲೇಷಿಸಿ", "medicines": "ಔಷಧಿಗಳು", "treatment": "ಉಪಚಾರ",
        "suggestions": "ಹೆಚ್ಚುವರಿ ಸಲಹೆಗಳು", "nutrients": "ಪೋಷಕಾಂಶಗಳು",
        "clear_history": "ಇತಿಹಾಸ ಅಳಿಸಿ", "download": "ಸಿಎಸ್ವಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "low_confidence": "⚠️ ಕಡಿಮೆ ವಿಶ್ವಾಸ: ದಯವಿಟ್ಟು ದೃಷ್ಟಿ ದೃಢೀಕರಿಸಿ ಅಥವಾ ಮತ್ತೊಮ್ಮೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.",
        "top_predictions": "ಇತರ ಸಾಧ್ಯತೆಗಳು", "alert_title": "🚨 ರೈತ ಎಚ್ಚರಿಕೆ ವ್ಯವಸ್ಥೆ",
        "langs_line": "ಇಂಗ್ಲಿಷ್, ಕನ್ನಡ ಮತ್ತು ಹಿಂದಿಯಲ್ಲಿ ಲಭ್ಯ.",
    },
    "hi": {
        "title": "एआई प्लांट डॉक्टर",
        "subtitle": "पत्ती की फोटो लें। कुछ ही सेकंड में रोग, इलाज और दवा पाएँ।",
        "analyze": "🔍 पत्ती जाँचें", "medicines": "अनुशंसित दवाइयाँ",
        "treatment": "सुझाया गया उपचार", "suggestions": "अतिरिक्त सुझाव",
        "nutrients": "पोषक तत्व", "clear_history": "इतिहास मिटाएँ",
        "download": "CSV डाउनलोड करें",
        "low_confidence": "⚠️ कम भरोसा: कृपया खुद जाँच लें या दोबारा फोटो लें।",
        "top_predictions": "अन्य संभावनाएँ", "alert_title": "🚨 किसान चेतावनी प्रणाली",
        "langs_line": "अंग्रेज़ी, कन्नड़ और हिन्दी में उपलब्ध।",
    },
}
LANG_NAMES = {"en": "English", "kn": "ಕನ್ನಡ (Kannada)", "hi": "हिन्दी (Hindi)"}

# ---------------- ORGANIC / PREVENTION KNOWLEDGE ----------------
ORGANIC = {
    "fungal": [
        "Neem oil spray on both leaf surfaces (follow the product label for dilution)",
        "Bio-fungicides based on <i>Trichoderma</i> or <i>Bacillus subtilis</i>",
        "Baking-soda + mild soap spray for mildew-type problems (test on a few leaves first)",
        "Remove and destroy infected leaves; water at the base, in the morning",
    ],
    "bacterial": [
        "Strict sanitation: remove infected leaves and never handle plants when wet",
        "Copper-based sprays are accepted in organic farming, use exactly as labelled",
        "Use drip irrigation instead of overhead watering",
        "Use certified disease-free seed / seedlings",
    ],
    "viral": [
        "No spray cures a virus. Remove and destroy infected plants early",
        "Control aphids / whiteflies: yellow sticky traps and neem-based sprays",
        "Disinfect tools and hands between plants",
        "Plant resistant varieties and use certified virus-free seed",
    ],
    "pest": [
        "Spray water forcefully on leaf undersides to knock mites off",
        "Neem oil or insecticidal soap on the underside of leaves",
        "Encourage predatory mites and ladybirds; avoid broad-spectrum sprays",
        "Keep plants well watered. Dusty, stressed plants get mites faster",
    ],
    "vector": [
        "Control psyllids with yellow sticky traps and neem-based products",
        "Remove infected trees to protect the rest of the orchard",
        "Plant only certified disease-free saplings",
        "Scout new flushes every week",
    ],
    "healthy": [
        "Mulch and add compost to keep the soil healthy",
        "Rotate crops every season",
        "Scout leaves weekly so problems are caught early",
        "Water at the base, in the morning",
    ],
}
PREVENTION = [
    "Choose resistant varieties where available",
    "Space plants for good airflow; prune dense growth",
    "Clean tools, pots and hands regularly",
    "Rotate crops; do not leave infected residue in the field",
    "Inspect the underside of leaves once a week",
]


def disease_category(cls):
    low = cls.lower()
    if "healthy" in low:
        return "healthy"
    if "virus" in low:
        return "viral"
    if "greening" in low:
        return "vector"
    if "bacterial" in low:
        return "bacterial"
    if "mite" in low:
        return "pest"
    return "fungal"


# ---------------- CLASS NAMES ----------------
def load_class_names():
    """class_names.json (list) > class_indices.json (dict) > alphabetical KB order."""
    for fname in ("class_names.json", "class_indices.json"):
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [str(x) for x in data]
                if isinstance(data, dict):
                    return [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
            except Exception:
                pass
    return sorted(disease_treatments.keys())


class_names = load_class_names()


def pretty_name(cls):
    return cls.replace("___", " - ").replace("__", " - ").replace("_", " ").strip()


def crop_of(cls):
    return re.split(r"_{2,}", cls)[0].replace("_", " ").strip()


# ---------------- MODEL ----------------
def find_model_path():
    env = os.environ.get("PLANT_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    for folder in (BASE_DIR, os.path.join(BASE_DIR, "model"), os.path.join(BASE_DIR, "models")):
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".keras", ".h5")):
                    return os.path.join(folder, f)
    return None


@st.cache_resource(show_spinner="Loading AI model...")
def load_model_cached(path):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False), None
    except Exception as e:
        return None, str(e)


def get_model():
    path = find_model_path()
    if not path:
        return None, None, "No model file (.keras / .h5) found next to app.py"
    model, err = load_model_cached(path)
    return model, path, err


def model_input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    try:
        h, w = shape[1], shape[2]
        return (int(w) if w else 224, int(h) if h else 224)
    except Exception:
        return (224, 224)


def _to_probs(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = p[None]
    bad = (p.min(axis=1) < 0) | (np.abs(p.sum(axis=1) - 1) > 1e-2)
    if bad.any():
        e = np.exp(p - p.max(axis=1, keepdims=True))
        p = np.where(bad[:, None], e / e.sum(axis=1, keepdims=True), p)
    return p


def preprocess(img, size, normalize):
    arr = np.asarray(img.convert("RGB").resize(size), dtype=np.float32)
    return arr / 255.0 if normalize else arr


def predict_disease(img, top_n=3, normalize=True):
    """Returns (results, mode) where mode is 'model' or 'demo'."""
    model, _, _ = get_model()
    if model is None:
        names = list(disease_treatments.keys())
        picks = random.sample(names, min(top_n, len(names)))
        res = [{"class": c, "confidence": random.uniform(70, 99)} for c in picks]
        return sorted(res, key=lambda x: x["confidence"], reverse=True), "demo"
    x = preprocess(img, model_input_size(model), normalize)[None]
    probs = _to_probs(model.predict(x, verbose=0))[0]
    names = class_names if len(class_names) == len(probs) else [f"Class_{i}" for i in range(len(probs))]
    idx = np.argsort(probs)[::-1][:top_n]
    return [{"class": names[i], "confidence": float(probs[i] * 100)} for i in idx], "model"


# ---------------- PHOTO QUALITY ----------------
def photo_quality(img):
    small = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    lap = (small[1:-1, 1:-1] * 4 - small[:-2, 1:-1] - small[2:, 1:-1]
           - small[1:-1, :-2] - small[1:-1, 2:])
    sharp, bright = float(lap.var()), float(small.mean())
    issues, score = [], 100
    if sharp < 40:
        issues.append(("🔴", "Photo looks blurry. Hold steady and tap to focus."))
        score -= 55
    elif sharp < 90:
        issues.append(("🟡", "Photo is slightly soft. A sharper shot improves accuracy."))
        score -= 30
    if bright < 55:
        issues.append(("🔴", "Too dark. Move into daylight (avoid flash)."))
        score -= 30
    elif bright > 215:
        issues.append(("🟡", "Very bright / overexposed. Move out of direct sun."))
        score -= 30
    return max(score, 5), issues


# ---------------- RISK ----------------
def risk_level(cls):
    low = cls.lower()
    meds = disease_treatments.get(cls, {}).get("medicines", "").lower()
    if "healthy" in low:
        return "Healthy", "healthy"
    if "no cure" in meds or "no chemical cure" in meds:
        return "Critical", "critical"
    if any(k in low for k in ("late_blight", "black_rot", "greening", "esca", "virus")):
        return "High", "high"
    return "Moderate", "moderate"


# ---------------- AI FOCUS MAP (occlusion) ----------------
def occlusion_heatmap(img, target_name, normalize=True, grid=7):
    model, _, _ = get_model()
    if model is None or target_name not in class_names:
        return None
    t = class_names.index(target_name)
    size = model_input_size(model)
    base = np.asarray(img.convert("RGB").resize(size), dtype=np.float32)
    H, W = base.shape[:2]

    def prep(a):
        return a / 255.0 if normalize else a

    base_p = _to_probs(model.predict(prep(base)[None], verbose=0))[0][t]
    ch, cw = max(H // grid, 1), max(W // grid, 1)
    batch = []
    for i in range(grid):
        for j in range(grid):
            a = base.copy()
            y2 = H if i == grid - 1 else (i + 1) * ch
            x2 = W if j == grid - 1 else (j + 1) * cw
            a[i * ch:y2, j * cw:x2, :] = 127.5
            batch.append(prep(a))
    preds = _to_probs(model.predict(np.stack(batch), verbose=0, batch_size=16))
    heat = np.clip(base_p - preds[:, t], 0, None).reshape(grid, grid)
    return heat / heat.max() if heat.max() > 0 else heat


def overlay_heatmap(img, heat, alpha=0.6):
    base = img.convert("RGB").copy()
    base.thumbnail((640, 640))
    hm = Image.fromarray((heat * 255).astype("uint8")).resize(base.size, RESAMPLE)
    h = (np.asarray(hm, dtype=np.float32) / 255.0)[..., None]
    b = np.asarray(base, dtype=np.float32)
    color = np.zeros_like(b)
    color[..., 0] = 255
    color[..., 1] = 220 * (1 - h[..., 0])
    out = b * (1 - alpha * h) + color * (alpha * h)
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


# ---------------- CARE PLAN ----------------
def care_plan(cls):
    _, key = risk_level(cls)
    if key == "healthy":
        return [(14, "Routine leaf inspection"), (30, "Fertilise as per nutrient plan")]
    if key == "critical":
        return [(0, "Remove / isolate infected plants immediately"), (0, "Disinfect tools and hands"),
                (3, "Inspect neighbouring plants for symptoms"), (7, "Re-check the whole field / garden")]
    if key == "high":
        return [(0, "Start recommended treatment"), (0, "Remove and destroy infected leaves"),
                (5, "Inspect for spread"), (10, "Second application if symptoms persist"),
                (21, "Final check and record results")]
    return [(0, "Apply recommended treatment"), (7, "Inspect the plant for new spots"),
            (14, "Repeat application if needed"), (28, "Final check")]


def build_ics(cls, plan):
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AI Plant Doctor//EN"]
    for d, task in plan:
        day = (datetime.now() + timedelta(days=d)).strftime("%Y%m%d")
        lines += ["BEGIN:VEVENT", f"UID:{uuid.uuid4()}@plantdoctor", f"DTSTAMP:{now}",
                  f"DTSTART;VALUE=DATE:{day}", f"SUMMARY:Plant Doctor - {task} ({pretty_name(cls)})",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines).encode("utf-8")


# ---------------- WEATHER ADVISOR (Open-Meteo, no API key) ----------------
def http_json(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "AIPlantDoctor/4.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


@st.cache_data(ttl=1800, show_spinner=False)
def geocode(city):
    url = ("https://geocoding-api.open-meteo.com/v1/search?name=" + quote_plus(city)
           + "&count=5&language=en&format=json")
    return http_json(url).get("results", []) or []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat, lon):
    url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
           "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum"
           "&hourly=relative_humidity_2m&timezone=auto&forecast_days=7" % (lat, lon))
    return http_json(url)


FUNGAL_LABELS = [("Low", "healthy"), ("Low", "healthy"), ("Moderate", "moderate"),
                 ("High", "high"), ("Very high", "critical")]


def fungal_score(tmax, tmin, hum, rain_mm):
    t, s = (tmax + tmin) / 2, 0
    if hum >= 85:
        s += 2
    elif hum >= 70:
        s += 1
    if 15 <= t <= 28:
        s += 1
    if rain_mm >= 2:
        s += 1
    return min(s, 4)


def spray_ok(d):
    return d["rain_prob"] < 40 and d["rain_mm"] < 1


def build_days(fc):
    daily, hourly = fc["daily"], fc["hourly"]
    buckets = {}
    for t, v in zip(hourly["time"], hourly["relative_humidity_2m"]):
        if v is not None:
            buckets.setdefault(t[:10], []).append(v)
    hum = {k: sum(v) / len(v) for k, v in buckets.items()}
    days = []
    for i, date in enumerate(daily["time"]):
        tmax = float(daily["temperature_2m_max"][i] or 0)
        tmin = float(daily["temperature_2m_min"][i] or 0)
        pp = float(daily["precipitation_probability_max"][i] or 0)
        rs = float(daily["precipitation_sum"][i] or 0)
        h = float(hum.get(date, 60))
        days.append({"date": date, "dt": datetime.strptime(date, "%Y-%m-%d"), "tmax": tmax,
                     "tmin": tmin, "rain_prob": pp, "rain_mm": rs, "hum": h,
                     "score": fungal_score(tmax, tmin, h, rs)})
    return days


def weather_card(d):
    icon = "🌧️" if d["rain_prob"] >= 60 else "⛅" if d["rain_prob"] >= 30 else "☀️"
    lab, key = FUNGAL_LABELS[d["score"]]
    spray = "✅ Spray OK" if spray_ok(d) else "⛔ Avoid spray"
    return (f"<div class='wx-card'><div class='wx-day'>{d['dt']:%a %d}</div><div class='wx-icon'>{icon}</div>"
            f"<div class='wx-temp'>{d['tmax']:.0f}° / {d['tmin']:.0f}°</div>"
            f"<div class='wx-meta'>💧 {d['hum']:.0f}% · ☔ {d['rain_prob']:.0f}%</div>"
            f"<span class='badge risk-{key}'>{lab}</span><div class='wx-meta'>{spray}</div></div>")


def weather_alert():
    geo = st.session_state.get("geo")
    if not geo:
        return None
    try:
        days = build_days(fetch_forecast(geo["lat"], geo["lon"]))
    except Exception:
        return None
    worst = max(days[:3], key=lambda x: x["score"])
    if worst["score"] >= 3:
        return {"title": f"🌧️ High fungal risk in {geo['name']}",
                "message": f"Humid / wet weather around {worst['dt']:%a %d %b}. Scout leaves and plan a protective spray on a dry day."}
    if worst["score"] == 2:
        return {"title": f"🌦️ Moderate disease pressure in {geo['name']}",
                "message": "Keep an eye on lower leaves over the next 3 days."}
    return None


def month_alert():
    month = datetime.now().month
    if month in (11, 12):
        return {"title": "🍂 High Fungal Risk Season!",
                "message": "Monitor lower canopy and consider a preventative copper spray."}
    if month == 5:
        return {"title": "☀️ Pest Monitoring Alert!", "message": "Hot dry conditions increase spider-mite risk."}
    return None


# ---------------- HISTORY / SCANS ----------------
def save_history(record):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    items = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    except Exception:
        pass
    return items


def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    shutil.rmtree(SCANS_DIR, ignore_errors=True)


def save_scan(img, cls, conf, source, plot, notes):
    os.makedirs(SCANS_DIR, exist_ok=True)
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.jpg"
    th = img.convert("RGB").copy()
    th.thumbnail((320, 320))
    th.save(os.path.join(SCANS_DIR, name), "JPEG", quality=85)
    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "disease": cls,
              "confidence": float(conf), "risk": risk_level(cls)[0], "source": source,
              "plot": (plot or "My Field").strip() or "My Field", "notes": notes or "", "thumb": name}
    save_history(record)
    return record


def history_frame(items):
    raw = pd.DataFrame(items)
    for col, default in (("plot", "My Field"), ("notes", ""), ("source", "unknown"), ("thumb", None),
                         ("risk", None), ("confidence", 0.0)):
        if col not in raw:
            raw[col] = default
    raw["plot"] = raw["plot"].fillna("My Field")
    raw["notes"] = raw["notes"].fillna("")
    raw["risk"] = [r if isinstance(r, str) else risk_level(str(d))[0] for r, d in zip(raw["risk"], raw["disease"])]
    raw["score"] = [RISK_SCORE[risk_level(str(d))[1]] for d in raw["disease"]]
    raw["name"] = raw["disease"].map(lambda x: pretty_name(str(x)))
    return raw


# ---------------- HTML BUILDERS ----------------
def gauge_svg(value, color):
    value = max(0.0, min(100.0, float(value)))
    filled = math.pi * 54 * value / 100
    return ("<div class='gauge'><svg viewBox='0 0 140 88' width='170' height='108'>"
            "<path d='M16 74 A54 54 0 0 1 124 74' fill='none' stroke='rgba(18,48,31,.10)' stroke-width='12' stroke-linecap='round'/>"
            f"<path d='M16 74 A54 54 0 0 1 124 74' fill='none' stroke='{color}' stroke-width='12' stroke-linecap='round' stroke-dasharray='{filled:.1f} 400'/>"
            f"<text x='70' y='66' text-anchor='middle' font-size='24' font-weight='700' fill='#12301f'>{value:.0f}%</text>"
            "<text x='70' y='84' text-anchor='middle' font-size='9' fill='#587263'>confidence</text></svg></div>")


def diagnosis_box(cls, conf, risk_label, risk_key):
    color = "#2ecc71" if conf >= 80 else "#f1c40f" if conf >= 60 else "#e74c3c"
    return ("<div class='primary-diagnosis-box'><div class='diag-flex'><div class='diag-text'>"
            f"<h2>✅ {pretty_name(cls)}</h2><p>Crop: <strong>{crop_of(cls)}</strong></p>"
            f"<span class='badge risk-{risk_key}'>{RISK_ICON[risk_key]} Risk: {risk_label}</span></div>"
            f"{gauge_svg(conf, color)}</div></div>")


def is_real_medicine(m):
    m = m.strip().lower()
    return bool(m) and not m.startswith(("no ", "none", "not "))


def shop_links(query):
    q = quote_plus(query + " fungicide pesticide")
    return (f'<a href="https://www.flipkart.com/search?q={q}" target="_blank" rel="noopener">🛒 {query} on Flipkart</a>'
            f' &nbsp;|&nbsp; <a href="https://www.amazon.in/s?k={q}" target="_blank" rel="noopener">Amazon.in</a>')


def knowledge_html(cls):
    info = disease_treatments.get(cls)
    if not info:
        return None
    meds = [m.strip() for m in info["medicines"].split(",") if is_real_medicine(m)]
    med_html = "".join(f"<div class='med-row'>{shop_links(m)}</div>" for m in meds) \
        or "<div class='med-row'>No purchase needed for this result.</div>"
    t = TRANSLATIONS[st.session_state.get("lang", "en")]
    return ("<div class='solution-box'>"
            f"<h3>💊 {t['medicines']}</h3><p>{info['medicines']}</p>{med_html}"
            f"<h3>🛠️ {t['treatment']}</h3><p>{info['treatment']}</p>"
            f"<h3>💡 {t['suggestions']}</h3><p>{info['suggestions']}</p>"
            f"<h3>🌱 {t['nutrients']}</h3><p>{info['nutrients']}</p></div>")


def organic_html(cls):
    li = "".join(f"<li>{x}</li>" for x in ORGANIC[disease_category(cls)])
    pv = "".join(f"<li>{x}</li>" for x in PREVENTION)
    return ("<div class='solution-box'><h3>🌿 Organic &amp; natural options</h3>"
            f"<ul>{li}</ul><h3>🛡️ Prevention checklist</h3><ul>{pv}</ul>"
            "<p style='opacity:.7;font-size:.85rem'>Organic options support recovery. For severe infections "
            "use the recommended treatment as well.</p></div>")


def whatsapp_url(cls, conf):
    info = disease_treatments.get(cls, {})
    risk_label, _ = risk_level(cls)
    text = (f"🌿 AI Plant Doctor\nCrop: {crop_of(cls)}\nDiagnosis: {pretty_name(cls)} ({conf:.0f}%)\n"
            f"Risk: {risk_label}\nTreatment: {info.get('treatment', '-')}\n"
            f"Medicines: {info.get('medicines', '-')}")
    return "https://wa.me/?text=" + quote(text)


# ---------------- PDF REPORT ----------------
def generate_pdf_report(cls, confidence, record, predictions, image, heat_img, mode, plot):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    info = disease_treatments.get(cls, {})
    risk_label, _ = risk_level(cls)
    plan = care_plan(cls)

    def header():
        c.setFillColor(HexColor("#004d40"))
        c.rect(0, height - 45, width, 45, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, height - 29, "AI Plant Doctor - Diagnosis Report")
        c.setFillColor(black)

    def footer():
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(HexColor("#555555"))
        c.drawString(40, 28, "Informational only. Consult a local agricultural expert before applying any treatment.")
        c.setFillColor(black)

    header()
    y = height - 80

    def ensure(space):
        nonlocal y
        if y - space < 50:
            footer()
            c.showPage()
            header()
            y = height - 80

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"Predicted: {pretty_name(cls)}")
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Confidence: {confidence:.2f}%     Risk level: {risk_label}     Field: {plot}")
    y -= 16
    c.setFont("Helvetica", 9)
    eng = "Trained AI model" if mode == "model" else "DEMO mode (simulated prediction)"
    c.drawString(40, y, f"Time: {record['time']}     Source: {record['source']}     Engine: {eng}")
    y -= 25

    img_h = 170
    ensure(img_h + 30)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Leaf image" + ("  /  AI focus map" if heat_img else ""))
    y -= 10
    c.drawImage(ImageReader(image), 40, y - img_h, width=img_h, height=img_h,
                preserveAspectRatio=True, anchor="nw")
    if heat_img:
        c.drawImage(ImageReader(heat_img), 40 + img_h + 20, y - img_h, width=img_h, height=img_h,
                    preserveAspectRatio=True, anchor="nw")
    y -= img_h + 25

    def section(title, text):
        nonlocal y
        lines = textwrap.wrap(re.sub(r"<[^>]+>", "", text), 95) or [""]
        ensure(20 + 14 * len(lines))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, title)
        y -= 15
        c.setFont("Helvetica", 10)
        for ln in lines:
            c.drawString(52, y, ln)
            y -= 14
        y -= 8

    section("Top predictions", "; ".join(f"{pretty_name(p['class'])} ({p['confidence']:.1f}%)" for p in predictions))
    section("Recommended medicines", info.get("medicines", "None"))
    section("Suggested treatment", info.get("treatment", "No treatment info available."))
    section("Additional suggestions", info.get("suggestions", "No suggestions available."))
    section("Key nutrient focus", info.get("nutrients", "Consult local expert."))
    section("Organic / natural options", "; ".join(ORGANIC[disease_category(cls)]))

    ensure(30 + 14 * len(plan))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Care plan")
    y -= 15
    c.setFont("Helvetica", 10)
    for d, task in plan:
        when = "Today" if d == 0 else f"Day {d}"
        date = (datetime.now() + timedelta(days=d)).strftime("%d %b %Y")
        c.drawString(52, y, f"[ ] {when} ({date}): {task}")
        y -= 14
    footer()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def speak_text(text):
    if not TTS_AVAILABLE:
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


# ---------------- CHATBOT ----------------
STOP = {"leaf", "spot", "the", "and", "for", "how", "can", "treat", "cure", "my", "of", "in", "is", "to", "a"}


def kb_lookup(prompt):
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    best, best_score = None, 0
    for cls in disease_treatments:
        crop_tokens = set(re.findall(r"[a-z]+", crop_of(cls).lower()))
        parts = re.split(r"_{2,}", cls, maxsplit=1)
        dis_tokens = set(re.findall(r"[a-z]+", (parts[1] if len(parts) > 1 else "").lower())) - STOP
        crop_hit = bool(words & crop_tokens)
        dis_hit = len(words & dis_tokens)
        if (crop_hit and dis_hit >= 1) or dis_hit >= 2:
            score = dis_hit + (1 if crop_hit else 0)
            if score > best_score:
                best, best_score = cls, score
    return best


def chatbot_response(prompt):
    p = prompt.lower()
    cls = kb_lookup(prompt)
    if cls is None and re.search(r"\b(my|last|this)\b.*\b(diagnosis|scan|result|plant|disease|leaf)\b", p):
        res = st.session_state.get("result")
        if res:
            cls = res["cls"]
    if cls:
        info = disease_treatments[cls]
        if re.search(r"\b(organic|natural|homemade|home remedy)\b", p):
            return (f"**Organic options for {pretty_name(cls)}**\n\n"
                    + "\n".join(f"- {re.sub('<[^>]+>', '', x)}" for x in ORGANIC[disease_category(cls)]))
        if re.search(r"\b(prevent|prevention|avoid)\b", p):
            return f"**Preventing {pretty_name(cls)}**\n\n" + "\n".join(f"- {x}" for x in PREVENTION)
        return (f"**{pretty_name(cls)}**\n\n- 💊 **Medicines:** {info['medicines']}\n"
                f"- 🛠️ **Treatment:** {info['treatment']}\n- 💡 **Tips:** {info['suggestions']}\n"
                f"- 🌱 **Nutrients:** {info['nutrients']}")
    for crop in {crop_of(c).lower(): crop_of(c) for c in disease_treatments}.values():
        if crop.lower() in p:
            diseases = [pretty_name(c).split(" - ", 1)[-1] for c in disease_treatments
                        if crop_of(c) == crop and "healthy" not in c.lower()]
            if diseases:
                return f"I know these **{crop}** problems: " + ", ".join(diseases) + ". Ask me about any of them!"
    if "weather" in p or "rain" in p:
        return "Open the **🌦️ Weather Advisor** page for a 7-day fungal-risk forecast and safe spray windows."
    if re.search(r"\b(help|support|expert)\b", p):
        return ("I can explain diseases, medicines, organic options and prevention. "
                "For urgent field problems call the Kisan Call Centre: **1800-180-1551**.")
    if re.search(r"\b(hello|hi|hey)\b", p):
        return "Hello! 👋 I'm your AI Crop Assistant. Ask me about a crop disease, medicines or organic options."
    if re.search(r"\b(fertili[sz]er|nutrient)\b", p):
        return "Tell me the crop and disease and I'll suggest a nutrient focus, e.g. *'nutrients for potato late blight'*."
    return "I'm still learning! Try *'tomato early blight cure'* or *'organic remedy for grape black rot'*."


# ---------------- CSS (light theme) ----------------
# Tokens (__A1__ etc.) are filled in by inject_css() from the chosen theme.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&family=Noto+Sans+Kannada:wght@400;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');
:root {
  --acc: __A1__; --acc2: __A2__; --acc-rgb: __A1RGB__; --acc2-rgb: __A2RGB__;
  --bg1: __B1__; --bg2: __B2__; --bg3: __B3__;
  --ink: #10281c; --muted: #56705f; --card: rgba(255,255,255,.86);
  --line: rgba(var(--acc2-rgb), .13);
  --shadow: 0 1px 2px rgba(var(--acc2-rgb), .06), 0 8px 24px rgba(var(--acc2-rgb), .07);
  --shadow-lg: 0 2px 4px rgba(var(--acc2-rgb), .06), 0 22px 50px rgba(var(--acc2-rgb), .16);
  --font: 'Figtree', 'Noto Sans Kannada', 'Noto Sans Devanagari', system-ui, sans-serif;
  --display: 'Bricolage Grotesque', 'Figtree', 'Noto Sans Kannada', 'Noto Sans Devanagari', system-ui, sans-serif;
}
/* fonts only on text elements, so Streamlit icon fonts keep working */
.stApp, .stApp p, .stApp label, .stApp li, .stApp a, .stApp button, .stApp input, .stApp textarea { font-family: var(--font); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--display); }

/* ---------- Page: soft aurora over a pale tint ---------- */
.stApp {
  background:
    radial-gradient(700px 420px at 92% -4%, rgba(var(--acc-rgb), .22), transparent 65%),
    radial-gradient(620px 380px at -6% 38%, rgba(var(--acc2-rgb), .10), transparent 68%),
    radial-gradient(520px 320px at 78% 100%, rgba(var(--acc-rgb), .14), transparent 70%),
    linear-gradient(170deg, var(--bg1), var(--bg2) 58%, var(--bg3));
  background-attachment: fixed; color: var(--ink);
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.6rem 2rem 4rem 2rem !important; max-width: 1150px !important; }

.stApp p, .stApp li, .stApp label { color: var(--ink); }
.stApp h2, .stApp h3, .stApp h4 { color: var(--ink) !important; font-weight: 700 !important; letter-spacing: -.3px; }
hr { border-color: var(--line) !important; }
.stApp [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

/* ---------- Hero: the one big moment ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 32px; padding: 2.6rem 2.8rem; margin-bottom: 1.8rem;
  background:
    radial-gradient(520px 260px at 100% 0%, rgba(var(--acc-rgb), .30), transparent 70%),
    radial-gradient(420px 240px at 0% 100%, rgba(var(--acc-rgb), .14), transparent 70%),
    linear-gradient(135deg, #ffffff, rgba(255,255,255,.72));
  border: 1px solid rgba(255,255,255,.9);
  box-shadow: var(--shadow-lg), inset 0 0 0 1px var(--line);
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  animation: rise .7s ease both;
}
@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
.hero::after { /* thin accent rule along the bottom edge */
  content: ""; position: absolute; left: 2.8rem; right: 2.8rem; bottom: 0; height: 4px; border-radius: 4px 4px 0 0;
  background: linear-gradient(90deg, var(--acc2), var(--acc), transparent);
}
.hero-text { position: relative; z-index: 1; max-width: 660px; }
.stApp .hero h1 {
  font-family: var(--display); color: var(--acc2) !important; -webkit-text-fill-color: var(--acc2);
  font-weight: 800 !important; font-size: clamp(2.1rem, 5.6vw, 3.7rem) !important; line-height: 1.04 !important;
  letter-spacing: -1.6px !important; margin: 0 0 .7rem 0 !important; padding: 0 !important;
}
.stApp .hero .hero-sub { color: var(--ink) !important; font-size: clamp(1.02rem, 2.4vw, 1.25rem); font-weight: 500; margin: 0 0 .55rem 0; max-width: 34em; }
.stApp .hero .hero-note { color: var(--muted) !important; font-size: .9rem; margin: 0; }
.hero-leaf { position: absolute; right: -14px; top: -34px; height: 160%; width: auto; pointer-events: none; }
.hero-leaf path:first-child { fill: rgba(var(--acc-rgb), .20); }
.hero-leaf path:last-child { stroke: rgba(var(--acc2-rgb), .38); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(var(--acc-rgb), .10)) !important;
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { font-size: .9rem !important; }
.brand { display: flex; align-items: center; gap: .75rem; padding: .3rem 0 1.1rem 0; }
.brand-logo { width: 46px; height: 46px; border-radius: 15px; display: grid; place-items: center;
  background: linear-gradient(140deg, var(--acc2), var(--acc)); box-shadow: 0 8px 18px rgba(var(--acc2-rgb), .30); }
.brand-name { font-family: var(--display); font-weight: 800; font-size: 1.2rem; color: var(--ink); line-height: 1.1; letter-spacing: -.3px; }
.brand-tag { font-size: .76rem; color: var(--muted); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  border-radius: 14px; padding: .5rem .8rem; margin: 3px 0; transition: background .2s ease, transform .2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(var(--acc-rgb), .12); transform: translateX(2px); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: #ffffff; box-shadow: var(--shadow), inset 4px 0 0 var(--acc);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { font-weight: 700; color: var(--acc2); }

/* ---------- Inputs ---------- */
.stTextInput input, .stNumberInput input, .stChatInput textarea {
  background: rgba(255,255,255,.95) !important; color: var(--ink) !important; border: 1px solid var(--line) !important;
  border-radius: 14px !important;
}
.stTextInput input:focus { border-color: var(--acc) !important; box-shadow: 0 0 0 4px rgba(var(--acc-rgb), .18) !important; }
div[data-baseweb="select"] > div { background: rgba(255,255,255,.95) !important; border: 1px solid var(--line) !important;
  border-radius: 14px !important; color: var(--ink) !important; }
div[data-testid="stChatInput"] { background: #ffffff; border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); }

div[data-testid="stFileUploader"] section, div[data-testid="stCameraInput"] {
  background: rgba(255,255,255,.85) !important; border: 2px dashed rgba(var(--acc-rgb), .55) !important; border-radius: 22px !important;
}
div[data-testid="stFileUploader"] section:hover { border-color: var(--acc2) !important; background: rgba(var(--acc-rgb), .07) !important; }
div[data-testid="stFileUploader"] section *, div[data-testid="stCameraInput"] label { color: var(--ink) !important; }
div[data-testid="stImage"] img { border-radius: 20px; border: 1px solid var(--line); box-shadow: var(--shadow); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, var(--acc2), var(--acc)); color: #ffffff !important; font-weight: 600; border: none;
  border-radius: 999px; padding: .68rem 1.6rem;
  box-shadow: 0 8px 20px rgba(var(--acc2-rgb), .28), inset 0 1px 0 rgba(255,255,255,.35);
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button p, .stDownloadButton > button p { color: #ffffff !important; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(var(--acc2-rgb), .36), inset 0 1px 0 rgba(255,255,255,.35); filter: brightness(1.05); }
.stButton > button:active { transform: scale(.98); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid rgba(var(--acc-rgb), .55); outline-offset: 3px; }

/* ---------- Tabs: pill segmented bar ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; background: rgba(255,255,255,.7); border: 1px solid var(--line); border-radius: 999px; padding: 5px; width: fit-content; max-width: 100%; }
.stTabs [data-baseweb="tab"] { background: transparent; border: none; border-radius: 999px; padding: .5rem 1.1rem; height: auto; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--acc2), var(--acc)) !important; box-shadow: 0 6px 14px rgba(var(--acc2-rgb), .26); }
.stTabs [aria-selected="true"] p { color: #ffffff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Progress, metrics, misc widgets ---------- */
.stProgress > div > div > div > div { background-image: linear-gradient(90deg, var(--acc2), var(--acc)); }
div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: .9rem 1.1rem; box-shadow: var(--shadow); backdrop-filter: blur(6px); }
div[data-testid="stMetricValue"] { color: var(--acc2) !important; font-family: var(--display); font-weight: 700; }
div[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
div[data-testid="stAlert"] { border-radius: 16px; }
div[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); }
.stChatMessage { background: var(--card) !important; border: 1px solid var(--line); border-radius: 20px; }
div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid var(--line); }
::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-thumb { background: rgba(var(--acc2-rgb), .28); border-radius: 8px; }

/* ---------- Diagnosis: the second big moment ---------- */
.primary-diagnosis-box {
  position: relative; overflow: hidden;
  background: radial-gradient(420px 200px at 100% 0%, rgba(var(--acc-rgb), .22), transparent 70%), #ffffff;
  border: 1px solid var(--line); border-radius: 26px;
  padding: 1.3rem 1.8rem 1.3rem 2.1rem; margin: 1rem 0; box-shadow: var(--shadow-lg);
}
.primary-diagnosis-box::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 9px; background: linear-gradient(180deg, var(--acc), var(--acc2)); }
.diag-flex { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .6rem; }
.diag-text h2 { margin: 0 0 .25rem 0; font-size: clamp(1.25rem, 4vw, 2rem); letter-spacing: -.6px; }
.diag-text p { margin: 0; }
.gauge { margin: 0 auto; }

.badge { display: inline-block; padding: .28rem .85rem; border-radius: 999px; font-weight: 600; font-size: .85rem; margin-top: .55rem; }
.risk-healthy  { background: #e3f8ea; color: #14803f; border: 1px solid #7fd4a0; }
.risk-moderate { background: #fff6d6; color: #8a6a00; border: 1px solid #f0cf5a; }
.risk-high     { background: #ffe9d6; color: #b0540b; border: 1px solid #f3a766; }
.risk-critical { background: #fde4e1; color: #b3261e; border: 1px solid #f0948d; }

.demo-banner { background: #fff8e1; border: 1px solid #f5d77a; color: #7a5b00; border-radius: 14px; padding: .55rem 1rem; font-size: .88rem; margin-bottom: 1rem; }
.help-card { background: rgba(var(--acc-rgb), .10); border: 1px dashed rgba(var(--acc2-rgb), .38); border-radius: 16px; padding: .85rem 1.1rem; margin-top: 1rem; font-size: .92rem; }
.share-btn { display: inline-block; background: #25D366; color: #06210f !important; font-weight: 700; padding: .62rem 1.4rem; border-radius: 999px; text-decoration: none !important; margin: .4rem .4rem .4rem 0; box-shadow: 0 8px 18px rgba(37,211,102,.30), inset 0 1px 0 rgba(255,255,255,.4); transition: transform .2s ease; }
.share-btn:hover { transform: translateY(-2px); filter: brightness(1.05); }

/* ---------- Content cards (quiet) ---------- */
.solution-box { background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 1.3rem 1.6rem; margin-top: .6rem; box-shadow: var(--shadow); }
.solution-box h3 { font-size: 1.05rem; margin: 1.1rem 0 .25rem 0; color: var(--acc2) !important; }
.solution-box h3:first-child { margin-top: 0; }
.solution-box a { color: var(--acc2) !important; text-decoration: none; font-weight: 600; border-bottom: 1px solid rgba(var(--acc2-rgb), .3); }
.solution-box a:hover { border-bottom-color: var(--acc2); }
.med-row { margin: .3rem 0; font-size: .92rem; }

.timeline { border-left: 3px solid rgba(var(--acc-rgb), .5); margin: .5rem 0 1rem .6rem; padding-left: 1.2rem; }
.tl-item { position: relative; margin-bottom: 1rem; }
.tl-item::before { content: ""; position: absolute; left: -1.72rem; top: .35rem; width: 13px; height: 13px; border-radius: 50%; background: var(--acc); box-shadow: 0 0 0 4px rgba(var(--acc-rgb), .22); }
.tl-day { font-weight: 700; color: var(--acc2); font-size: .9rem; }

.step-grid, .stat-grid, .wx-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0 1.4rem; }
.wx-grid { grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: .8rem; }
.step-card, .stat-card, .wx-card { background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 1.2rem; text-align: center; box-shadow: var(--shadow); }
.step-ico { font-size: 2rem; } .step-title { font-family: var(--display); font-weight: 700; color: var(--acc2); margin: .3rem 0; } .step-text { font-size: .88rem; color: var(--muted); }
.stat-num { font-family: var(--display); font-size: 2.1rem; font-weight: 800; color: var(--acc2); letter-spacing: -.5px; } .stat-lbl { font-size: .85rem; color: var(--muted); }
.wx-day { font-weight: 700; } .wx-icon { font-size: 1.8rem; } .wx-temp { font-weight: 700; color: var(--acc2); } .wx-meta { font-size: .78rem; color: var(--muted); margin-top: .25rem; }

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
  .block-container { padding: 1rem 1rem 3rem 1rem !important; }
  .hero { padding: 1.7rem 1.4rem; border-radius: 24px; }
  .hero::after { left: 1.4rem; right: 1.4rem; }
  .hero-leaf { opacity: .5; }
  .primary-diagnosis-box, .solution-box { padding: 1rem 1rem 1rem 1.4rem; border-radius: 18px; }
  .stButton > button, .stDownloadButton > button { width: 100%; }
  .diag-flex { justify-content: center; text-align: center; }
  .stTabs [data-baseweb="tab-list"] { border-radius: 20px; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""
LARGE_TEXT_CSS = ("<style>.stMarkdown p, .stMarkdown li, .stApp label p, .stButton button p, "
                  ".solution-box, .wx-meta, .step-text { font-size: 1.15rem !important; }</style>")


def inject_css(theme_name, large):
    th = THEMES[theme_name]
    tokens = {"__A1RGB__": hex_to_rgb(th["acc"]), "__A2RGB__": hex_to_rgb(th["acc2"]),
              "__A1__": th["acc"], "__A2__": th["acc2"],
              "__B1__": th["bg"][0], "__B2__": th["bg"][1], "__B3__": th["bg"][2]}
    css = CSS
    for key, val in tokens.items():
        css = css.replace(key, val)
    st.markdown(css + (LARGE_TEXT_CSS if large else ""), unsafe_allow_html=True)


# ---------------- RESULT RENDERER ----------------
def render_result(res):
    cls, conf = res["cls"], res["confidence"]
    risk_label, risk_key = risk_level(cls)
    t = TRANSLATIONS[st.session_state.get("lang", "en")]

    if res["mode"] == "demo":
        st.markdown("<div class='demo-banner'>🧪 DEMO result: no trained model found, prediction is simulated.</div>",
                    unsafe_allow_html=True)
    st.markdown(diagnosis_box(cls, conf, risk_label, risk_key), unsafe_allow_html=True)
    if conf < st.session_state.get("min_conf", 70):
        st.warning(t["low_confidence"])

    b1, b2 = st.columns([1, 1])
    with b1:
        st.download_button("🖨️ Download PDF Report", data=res["pdf"],
                           file_name=f"plant_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                           mime="application/pdf", key="pdf_dl", **STRETCH)
    with b2:
        st.markdown(f"<a class='share-btn' href='{whatsapp_url(cls, conf)}' target='_blank' rel='noopener'>💬 Share on WhatsApp</a>",
                    unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["🩺 Diagnosis", "💊 Cure & Medicines", "🌿 Organic & Prevention",
                                  "🔥 AI Focus Map", "📅 Care Plan"])
    with t1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Crop", crop_of(cls))
        m2.metric("Field", res.get("plot", "My Field"))
        m3.metric("Photo quality", f"{res['quality']}/100")
        if len(res["predictions"]) > 1:
            st.markdown(f"**{t['top_predictions']}**")
            for r in res["predictions"][1:]:
                st.progress(min(max(r["confidence"] / 100, 0.0), 1.0),
                            text=f"{pretty_name(r['class'])} — {r['confidence']:.1f}%")
        for icon, msg in res["issues"]:
            st.write(f"{icon} {msg}")
        st.markdown(EXPERT_HTML, unsafe_allow_html=True)
    with t2:
        html = knowledge_html(cls)
        if html:
            st.markdown(html, unsafe_allow_html=True)
            st.caption("Always read the product label and follow local agricultural guidelines.")
        else:
            st.info("No detailed treatment information available for this class.")
    with t3:
        st.markdown(organic_html(cls), unsafe_allow_html=True)
    with t4:
        if res.get("heat_img") is not None:
            c1, c2 = st.columns(2)
            c1.image(res["image"], caption="Your leaf", **STRETCH)
            c2.image(res["heat_img"], caption="Red = areas that drove the prediction", **STRETCH)
            st.caption("Computed by hiding parts of the leaf and measuring how much the confidence drops.")
        elif res["mode"] == "demo":
            st.info("The focus map needs a trained model.")
        else:
            st.info("Enable “Generate AI focus map” in Model settings and analyze again.")
    with t5:
        plan = care_plan(cls)
        items = ""
        for d, task in plan:
            when = "Today" if d == 0 else f"Day {d}"
            date = (datetime.now() + timedelta(days=d)).strftime("%d %b")
            items += f"<div class='tl-item'><div class='tl-day'>{when} · {date}</div><div>{task}</div></div>"
        st.markdown(f"<div class='timeline'>{items}</div>", unsafe_allow_html=True)
        st.download_button("📆 Add reminders to my calendar (.ics)", data=build_ics(cls, plan),
                           file_name="plant_care_plan.ics", mime="text/calendar", key="ics_dl")
    st.caption("⚠️ For information only. Confirm with a local agricultural expert before treating crops.")


def read_image(f):
    return ImageOps.exif_transpose(Image.open(f)).convert("RGB")


# =====================================================================
#                              UI STARTS
# =====================================================================
PAGES = {"home": "🏠 Home", "batch": "🗂️ Batch Scan", "weather": "🌦️ Weather Advisor",
         "library": "📚 Disease Library", "chat": "🤖 Chatbot", "history": "📜 History", "about": "ℹ️ About"}

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    f"<div class='brand'><div class='brand-logo'>{LEAF_LOGO}</div>"
    "<div><div class='brand-name'>Plant Doctor</div><div class='brand-tag'>Leaf health, simplified</div></div></div>",
    unsafe_allow_html=True)
lang_choice = st.sidebar.selectbox("Language / ಭಾಷೆ", list(LANG_NAMES), format_func=LANG_NAMES.get)
st.session_state["lang"] = lang_choice
txt = TRANSLATIONS[lang_choice]

theme_name = st.sidebar.selectbox("🎨 Theme", list(THEMES))
large_text = st.sidebar.toggle("🔠 Large text mode", value=False)
page = st.sidebar.radio("Go to / ತೆರೆಯಿರಿ", list(PAGES), format_func=PAGES.get, key="main_nav")

st.sidebar.markdown(f"### {txt['alert_title']}")
_alert = weather_alert() or month_alert()
if _alert:
    st.sidebar.warning(f"**{_alert['title']}**\n\n{_alert['message']}")
else:
    st.sidebar.info("No major alerts currently active.")

with st.sidebar.expander("⚙️ Model settings"):
    normalize = st.checkbox("Scale pixels to 0–1 (÷255)", value=True,
                            help="Turn OFF if your model was trained on raw 0–255 pixels or has its own rescaling layer.")
    show_heat = st.checkbox("Generate AI focus map", value=True,
                            help="Highlights leaf regions that influenced the prediction (a few extra seconds).")
    voice_on = st.checkbox("Speak result (local PC only)", value=False, disabled=not TTS_AVAILABLE)
    min_conf = st.slider("Low-confidence warning below (%)", 30, 95, 70)
    st.session_state["min_conf"] = min_conf

_model, _model_path, _model_err = get_model()
if _model is not None:
    _w, _h = model_input_size(_model)
    st.sidebar.success(f"🧠 {os.path.basename(_model_path)}\n\nInput {_w}×{_h} · {len(class_names)} classes")
else:
    st.sidebar.warning("🧪 Demo mode: no trained model loaded.")
    if _model_err:
        st.sidebar.caption(f"Reason: {_model_err[:200]}")

if st.sidebar.button(txt["clear_history"], key="clear_hist_btn"):
    clear_history()
    st.sidebar.success("History cleared.")

inject_css(theme_name, large_text)

# ---------------- HERO ----------------
st.markdown(
    "<div class='hero'><div class='hero-text'>"
    f"<h1>{txt['title']}</h1>"
    f"<p class='hero-sub'>{txt['subtitle']}</p>"
    f"<p class='hero-note'>{txt['langs_line']}</p>"
    f"</div>{LEAF_HERO}</div>",
    unsafe_allow_html=True)

# ======================= HOME =======================
if page == "home":
    c_plot, c_notes = st.columns([1, 2])
    plot_name = c_plot.text_input("🌾 Field / plot name", value="My Field", key="plot_name")
    notes = c_notes.text_input("📝 Notes (optional)", placeholder="e.g. lower leaves, after rain", key="scan_notes")

    st.markdown("### 📷 Select Image Source")
    input_method = st.radio("Input Method", ["Camera", "Upload"], horizontal=True, label_visibility="collapsed")
    image_obj = None
    if input_method == "Camera":
        cam = st.camera_input("Take a clear close-up picture of the leaf")
        if cam:
            image_obj = read_image(cam)
    else:
        up = st.file_uploader("Upload leaf image (jpg/png)", type=["jpg", "jpeg", "png"])
        if up:
            image_obj = read_image(up)

    if not image_obj:
        steps = [("📷", "Capture or upload", "Take a close, well-lit photo of a single leaf."),
                 ("🧠", "AI analysis", "Disease, confidence and risk level in seconds."),
                 ("💊", "Treat & track", "Cure plan, medicines, calendar and PDF report.")]
        st.markdown("<div class='step-grid'>" + "".join(
            f"<div class='step-card'><div class='step-ico'>{i}</div><div class='step-title'>{t}</div>"
            f"<div class='step-text'>{d}</div></div>" for i, t, d in steps) + "</div>", unsafe_allow_html=True)
        hist = load_history()
        if hist:
            hf = history_frame(hist)
            healthy_pct = (hf["score"] == 0).mean() * 100
            st.markdown("<div class='stat-grid'>"
                        f"<div class='stat-card'><div class='stat-num'>{len(hf)}</div><div class='stat-lbl'>Total scans</div></div>"
                        f"<div class='stat-card'><div class='stat-num'>{healthy_pct:.0f}%</div><div class='stat-lbl'>Healthy leaves</div></div>"
                        f"<div class='stat-card'><div class='stat-num'>{hf['plot'].nunique()}</div><div class='stat-lbl'>Fields tracked</div></div>"
                        "</div>", unsafe_allow_html=True)
        with st.expander("📸 Tips for a perfect photo"):
            st.markdown("- Use daylight; avoid flash and harsh shadows\n- Fill the frame with **one** leaf on a plain background\n"
                        "- Keep the camera steady and in focus\n- Photograph the affected area, not just a healthy part\n"
                        "- Take 2–3 photos of different leaves for confidence")
    else:
        img_id = hashlib.md5(image_obj.resize((64, 64)).tobytes()).hexdigest()
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(image_obj, caption="Leaf image", **STRETCH)
        with col_b:
            q_score, q_issues = photo_quality(image_obj)
            st.markdown("#### 📸 Photo check")
            st.progress(q_score / 100, text=f"Quality score: {q_score}/100")
            if q_issues:
                for icon, msg in q_issues:
                    st.write(f"{icon} {msg}")
            else:
                st.write("🟢 Looks good. Sharp and well lit.")
            analyze = st.button(txt["analyze"], **STRETCH)

        if analyze:
            with st.status("Analyzing your leaf...", expanded=True) as status:
                st.write("🧠 Running disease detection")
                preds, mode = predict_disease(image_obj, top_n=3, normalize=normalize)
                cls, conf = preds[0]["class"], preds[0]["confidence"]
                heat_img = None
                if mode == "model" and show_heat:
                    st.write("🔥 Building AI focus map")
                    try:
                        heat = occlusion_heatmap(image_obj, cls, normalize=normalize)
                        if heat is not None:
                            heat_img = overlay_heatmap(image_obj, heat)
                    except Exception:
                        heat_img = None
                st.write("🗂️ Saving to your field diary")
                record = save_scan(image_obj, cls, conf, "camera" if input_method == "Camera" else "upload",
                                   plot_name, notes)
                st.write("🖨️ Preparing PDF report")
                pdf = generate_pdf_report(cls, conf, record, preds, image_obj, heat_img, mode, record["plot"])
                st.session_state["result"] = {"img_id": img_id, "cls": cls, "confidence": conf, "predictions": preds,
                                              "mode": mode, "pdf": pdf, "heat_img": heat_img, "image": image_obj,
                                              "quality": q_score, "issues": q_issues, "plot": record["plot"]}
                status.update(label="Analysis complete ✅", state="complete", expanded=False)
            if voice_on:
                speak_text(f"{pretty_name(cls)} detected. Risk level {record['risk']}.")
            if risk_level(cls)[1] == "healthy" and conf >= 85:
                st.balloons()
            st.toast(f"Detected: {pretty_name(cls)}", icon="🌿")

        res = st.session_state.get("result")
        if res and res.get("img_id") == img_id:
            render_result(res)

# ======================= BATCH SCAN =======================
elif page == "batch":
    st.markdown("## 🗂️ Batch Scan: check a whole field at once")
    st.caption("Upload up to 12 leaf photos. You get a field health score and the most urgent problem.")
    bplot = st.text_input("🌾 Field / plot name", value="My Field", key="batch_plot")
    files = st.file_uploader("Upload leaf photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_files")
    files = (files or [])[:12]
    if files and st.button(f"🔍 Analyze {len(files)} photo(s)", **STRETCH):
        rows, prog = [], st.progress(0.0, text="Starting...")
        for i, f in enumerate(files):
            img = read_image(f)
            preds, mode = predict_disease(img, top_n=1, normalize=normalize)
            cls, conf = preds[0]["class"], preds[0]["confidence"]
            rec = save_scan(img, cls, conf, "batch", bplot, "")
            rows.append({"file": f.name, "cls": cls, "conf": conf, "risk": rec["risk"], "thumb": rec["thumb"], "mode": mode})
            prog.progress((i + 1) / len(files), text=f"Analyzed {i + 1}/{len(files)}")
        prog.empty()
        st.session_state["batch"] = rows
    rows = st.session_state.get("batch")
    if rows and files:
        if rows[0]["mode"] == "demo":
            st.markdown("<div class='demo-banner'>🧪 DEMO results: no trained model found.</div>", unsafe_allow_html=True)
        keys = [risk_level(r["cls"])[1] for r in rows]
        health = sum(HEALTH_POINTS[k] for k in keys) / len(keys)
        m1, m2, m3 = st.columns(3)
        m1.metric("Field health score", f"{health:.0f}/100")
        m2.metric("Healthy leaves", f"{sum(k == 'healthy' for k in keys)}/{len(keys)}")
        m3.metric("Need action", sum(k in ('high', 'critical') for k in keys))
        st.progress(health / 100, text="Overall field health")
        worst = max(rows, key=lambda r: RISK_SCORE[risk_level(r["cls"])[1]])
        if risk_level(worst["cls"])[1] != "healthy":
            st.markdown(f"### 🚨 Most urgent: {pretty_name(worst['cls'])}")
            st.markdown(knowledge_html(worst["cls"]) or "", unsafe_allow_html=True)
        st.markdown("#### Results")
        cols = st.columns(4)
        for i, r in enumerate(rows):
            with cols[i % 4]:
                p = os.path.join(SCANS_DIR, r["thumb"])
                if os.path.exists(p):
                    st.image(p, **STRETCH)
                st.markdown(f"{RISK_ICON[risk_level(r['cls'])[1]]} **{pretty_name(r['cls'])}**  \n{r['conf']:.0f}%")
        st.bar_chart(pd.Series([pretty_name(r["cls"]) for r in rows]).value_counts())
        out = pd.DataFrame([{"file": r["file"], "disease": pretty_name(r["cls"]), "confidence %": round(r["conf"], 1),
                             "risk": r["risk"]} for r in rows])
        st.dataframe(out, **STRETCH)
        st.download_button("📥 Download batch results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           "batch_results.csv", "text/csv")

# ======================= WEATHER ADVISOR =======================
elif page == "weather":
    st.markdown("## 🌦️ Weather Advisor")
    st.caption("7-day fungal-disease pressure and the safest days to spray (forecast by Open-Meteo). Needs internet.")
    city = st.text_input("📍 Your city / village", value=(st.session_state.get("geo") or {}).get("name", ""),
                         placeholder="e.g. Bengaluru")
    crops = sorted({crop_of(c) for c in disease_treatments})
    crop_sel = st.selectbox("🌱 Your crop (optional)", ["Any"] + crops)
    if city.strip():
        try:
            places = geocode(city.strip())
        except Exception as e:
            places = None
            st.error(f"Couldn't reach the weather service ({e.__class__.__name__}). Check your internet and try again.")
        if places is not None and not places:
            st.warning("Location not found. Try a nearby town or a different spelling.")
        elif places:
            labels = [f"{p['name']}, {p.get('admin1', '')}, {p.get('country', '')}".replace(" ,", ",") for p in places]
            pick = st.selectbox("Select the right place", range(len(places)), format_func=lambda i: labels[i]) \
                if len(places) > 1 else 0
            geo = places[pick]
            st.session_state["geo"] = {"name": geo["name"], "lat": geo["latitude"], "lon": geo["longitude"]}
            try:
                days = build_days(fetch_forecast(geo["latitude"], geo["longitude"]))
            except Exception as e:
                days = None
                st.error(f"Couldn't load the forecast ({e.__class__.__name__}).")
            if days:
                st.markdown("<div class='wx-grid'>" + "".join(weather_card(d) for d in days) + "</div>",
                            unsafe_allow_html=True)
                windows = [f"{d['dt']:%a %d %b}" for d in days if spray_ok(d)]
                worst = max(days, key=lambda x: x["score"])
                if windows:
                    st.success("✅ **Best spray days:** " + ", ".join(windows) +
                               " (low rain chance, so the spray is less likely to wash off).")
                else:
                    st.warning("⛔ No dry spray window in the next 7 days. Check again in a day or two.")
                if worst["score"] >= 3:
                    st.error(f"High fungal pressure around {worst['dt']:%a %d %b}. Scout leaves and consider a protective spray before it.")
                elif worst["score"] == 2:
                    st.info("Moderate disease pressure ahead. Keep scouting the lower leaves.")
                else:
                    st.success("Low disease pressure this week.")
                if crop_sel != "Any" and worst["score"] >= 2:
                    probs = [pretty_name(c).split(" - ", 1)[-1] for c in disease_treatments
                             if crop_of(c) == crop_sel and "healthy" not in c.lower()]
                    st.markdown(f"**Watch {crop_sel} for:** " + ", ".join(probs))
                st.markdown("#### Humidity & rain chance")
                st.line_chart(pd.DataFrame({"Humidity %": [d["hum"] for d in days],
                                            "Rain chance %": [d["rain_prob"] for d in days]},
                                           index=[f"{d['dt']:%a %d}" for d in days]))
                st.caption("Risk score = humidity + temperature + rain. It is a general estimate, not a diagnosis.")
    else:
        st.info("Enter your location to see the forecast. Your sidebar alerts will then use live weather too.")

# ======================= DISEASE LIBRARY =======================
elif page == "library":
    st.markdown("## 📚 Disease Library")
    q = st.text_input("🔎 Search disease, crop or medicine", placeholder="e.g. blight, copper, grape")
    f1, f2 = st.columns(2)
    crop_f = f1.multiselect("Crop", sorted({crop_of(c) for c in disease_treatments}))
    risk_f = f2.multiselect("Risk level", ["Healthy", "Moderate", "High", "Critical"])
    matches = []
    for cls, info in disease_treatments.items():
        blob = (pretty_name(cls) + " " + info["medicines"]).lower()
        if q and q.lower() not in blob:
            continue
        if crop_f and crop_of(cls) not in crop_f:
            continue
        if risk_f and risk_level(cls)[0] not in risk_f:
            continue
        matches.append(cls)
    st.caption(f"{len(matches)} result(s)")
    for cls in matches:
        label, key = risk_level(cls)
        with st.expander(f"{RISK_ICON[key]} {pretty_name(cls)} · {label}"):
            st.markdown(knowledge_html(cls), unsafe_allow_html=True)
            st.markdown(organic_html(cls), unsafe_allow_html=True)

# ======================= CHATBOT =======================
elif page == "chat":
    st.markdown("## 🤖 AI Crop Assistant")
    st.markdown("Ask about any disease, medicine, organic option or prevention. Say *“my last diagnosis”* to talk about your latest scan.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    chips = ["Tomato early blight cure", "Organic remedy for potato late blight",
             "How to prevent grape black rot", "What diseases affect apple?"]
    chip_cols = st.columns(len(chips))
    chip = None
    for i, (col, label) in enumerate(zip(chip_cols, chips)):
        if col.button(label, key=f"chip_{i}", **STRETCH):
            chip = label
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍🌾" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
    prompt = st.chat_input("How can I treat tomato early blight?") or chip
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🌾"):
            st.markdown(prompt)
        response = chatbot_response(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ======================= HISTORY =======================
elif page == "history":
    st.markdown("## 📜 Field Diary & Analytics")
    items = load_history()
    if not items:
        st.info("No scans yet. Analyze a leaf on the Home page!")
    else:
        raw = history_frame(items)
        sel = st.selectbox("🌾 Field / plot", ["All"] + sorted(raw["plot"].unique()))
        df = raw if sel == "All" else raw[raw["plot"] == sel]
        total = len(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total scans", total)
        c2.metric("Healthy leaves", f"{(df['score'] == 0).mean() * 100:.0f}%")
        issues = df[df["score"] > 0]["name"].value_counts()
        c3.metric("Most common issue", issues.index[0] if len(issues) else "None")

        tab_t, tab_g, tab_tr = st.tabs(["📋 Table", "🖼️ Gallery", "📈 Trends"])
        with tab_t:
            table = pd.DataFrame({"time": df["time"], "field": df["plot"], "disease": df["name"],
                                  "confidence": df["confidence"].map(lambda c: f"{c:.1f}%"),
                                  "risk": df["risk"], "source": df["source"], "notes": df["notes"]}).iloc[::-1]
            st.dataframe(table, **STRETCH)
            st.download_button("📥 " + txt["download"], table.to_csv(index=False).encode("utf-8"),
                               "plant_doctor_history.csv", "text/csv")
        with tab_g:
            shown = df[df["thumb"].notna()].iloc[::-1].head(24)
            if shown.empty:
                st.info("Photos are saved from now on. Scan a leaf to start your gallery.")
            cols = st.columns(4)
            for i, (_, r) in enumerate(shown.iterrows()):
                p = os.path.join(SCANS_DIR, str(r["thumb"]))
                with cols[i % 4]:
                    if os.path.exists(p):
                        st.image(p, **STRETCH)
                    st.caption(f"{RISK_ICON[risk_level(str(r['disease']))[1]]} {r['name']}  \n{r['time']} · {r['plot']}")
        with tab_tr:
            tr = df.assign(time=pd.to_datetime(df["time"], errors="coerce")).dropna(subset=["time"])
            if len(tr) >= 2:
                st.markdown("**Disease severity over time** (0 = healthy · 3 = critical)")
                st.line_chart(tr.pivot_table(index="time", columns="plot", values="score", aggfunc="mean"))
                if sel != "All":
                    first, last = tr.iloc[0]["score"], tr.iloc[-1]["score"]
                    if last < first:
                        st.success(f"📈 {sel} is improving: severity dropped from {first} to {last}.")
                    elif last > first:
                        st.error(f"📉 {sel} is getting worse: severity rose from {first} to {last}.")
                    else:
                        st.info(f"➡️ {sel} is stable since the first scan.")
            else:
                st.info("Scan at least two leaves to see trends.")
            st.markdown("**Disease frequency**")
            st.bar_chart(df["name"].value_counts())

# ======================= ABOUT =======================
elif page == "about":
    st.markdown("## ℹ️ About AI Plant Doctor")
    st.markdown("""
Identify leaf diseases from a photo, then get a cure plan, medicine links and a printable report.

**Features**
- 📷 Camera capture or upload · 🧠 trained Keras model (auto-detected) with demo fallback
- 💊 Treatment, nutrients, Flipkart / Amazon links · 🌿 organic options and prevention
- 🖨️ PDF report · 💬 WhatsApp share · 📆 calendar reminders
- 📸 Photo-quality check · 🎯 confidence gauge · ⚠️ risk level · 🔥 AI focus map
- 🗂️ Batch scan with field health score · 🌦️ weather advisor and spray windows
- 📚 Disease library · 🤖 smart chatbot · 📜 field diary with gallery, trends and recovery tracker
- 🎨 4 light themes · 🔠 large-text mode · English / Kannada / Hindi

**Using your own model**
1. Put your `.keras` or `.h5` file next to `app.py` (or in a `model/` folder).
2. Optional: add `class_names.json` with class names in training order.
3. If predictions look wrong, toggle *Scale pixels to 0–1* in Model settings.

**Disclaimer:** For informational purposes only. Always consult local agricultural experts for final diagnosis and treatment.
""")
    st.markdown(EXPERT_HTML, unsafe_allow_html=True)
