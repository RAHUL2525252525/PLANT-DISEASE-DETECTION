"""
AI Plant Doctor 🌿  (v6 - simple, green, mobile friendly)
=========================================================
Files:  app.py  +  knowledge.py  (+ your trained .keras / .h5 model in ./model)

What it does
  • Big round camera button  +  upload from gallery
  • AI diagnosis with confidence gauge, risk level and top-3 possibilities
  • Cure & treatment suggestions, nutrients, medicines with Flipkart buy links
  • Organic options, prevention checklist, care plan (+ calendar reminders)
  • Photo-quality check, WhatsApp share, recent scans, PDF diagnosis report
Works on laptop and mobile. Needs Streamlit >= 1.39 for the round camera button.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import re
import io
import json
import uuid
import base64
import random
import hashlib
import textwrap
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus, quote

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black, white, HexColor
from reportlab.lib.utils import ImageReader

from knowledge import disease_treatments

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Plant Doctor", page_icon="🌿", layout="wide",
                   initial_sidebar_state="collapsed")

# Newer Streamlit replaced use_container_width with width="stretch"
_m = re.match(r"(\d+)\.(\d+)", st.__version__)
_ST_VER = (int(_m.group(1)), int(_m.group(2))) if _m else (1, 0)
STRETCH = {"width": "stretch"} if _ST_VER >= (1, 50) else {"use_container_width": True}

RISK_ICON = {"healthy": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}
EXPERT_HTML = ("<div class='note'>📞 <b>Need an expert?</b> Kisan Call Centre (India, toll-free): "
               "<b>1800-180-1551</b> or visit your nearest Krishi Vigyan Kendra (KVK).</div>")

# ---------------- Knowledge ----------------
ORGANIC = {
    "fungal": [
        "Neem oil spray on both sides of the leaves (follow the label for dilution)",
        "Bio-fungicides with Trichoderma or Bacillus subtilis",
        "Baking soda + mild soap spray for mildew-type problems (test on a few leaves first)",
        "Remove and destroy infected leaves; water at the base, in the morning",
    ],
    "bacterial": [
        "Remove infected leaves and never handle plants when they are wet",
        "Copper-based sprays are accepted in organic farming; use exactly as labelled",
        "Use drip irrigation instead of overhead watering",
        "Start with certified disease-free seed or seedlings",
    ],
    "viral": [
        "No spray cures a virus. Remove and destroy infected plants early",
        "Control aphids and whiteflies with yellow sticky traps and neem sprays",
        "Disinfect tools and hands between plants",
        "Plant resistant varieties and certified virus-free seed",
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
        "Check leaves weekly so problems are caught early",
        "Water at the base, in the morning",
    ],
}
PREVENTION = [
    "Choose resistant varieties where available",
    "Space plants for good airflow and prune dense growth",
    "Clean tools, pots and hands regularly",
    "Rotate crops and do not leave infected residue in the field",
    "Check the underside of leaves once a week",
]
SPRAY_SAFETY = [
    "Spray early morning or late evening, never in strong sun or wind",
    "Skip spraying if rain is expected within 6 hours",
    "Wear gloves, mask and full sleeves; wash up afterwards",
    "Follow the label dose exactly. More is not better",
    "Keep chemicals away from children, pets and water sources",
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


def pretty_name(cls):
    return cls.replace("___", " - ").replace("__", " - ").replace("_", " ").strip()


def crop_of(cls):
    return re.split(r"_{2,}", cls)[0].replace("_", " ").strip()


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


# ---------------- Class names & model ----------------
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
        return None, None, "No model file (.keras / .h5) found"
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


def predict_disease(img, top_n=3, normalize=True):
    """Returns (results, mode) where mode is 'model' or 'demo'."""
    model, _, _ = get_model()
    if model is None:
        names = list(disease_treatments.keys())
        picks = random.sample(names, min(top_n, len(names)))
        res = [{"class": c, "confidence": random.uniform(70, 99)} for c in picks]
        return sorted(res, key=lambda x: x["confidence"], reverse=True), "demo"
    arr = np.asarray(img.convert("RGB").resize(model_input_size(model)), dtype=np.float32)
    x = (arr / 255.0 if normalize else arr)[None]
    probs = _to_probs(model.predict(x, verbose=0))[0]
    names = class_names if len(class_names) == len(probs) else [f"Class_{i}" for i in range(len(probs))]
    idx = np.argsort(probs)[::-1][:top_n]
    return [{"class": names[i], "confidence": float(probs[i] * 100)} for i in idx], "model"


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
        issues.append(("🟡", "Very bright. Move out of direct sun."))
        score -= 30
    return max(score, 5), issues


# ---------------- Care plan ----------------
def care_plan(cls):
    _, key = risk_level(cls)
    if key == "healthy":
        return [(14, "Routine leaf check"), (30, "Fertilise as per the nutrient plan")]
    if key == "critical":
        return [(0, "Remove / isolate infected plants immediately"), (0, "Disinfect tools and hands"),
                (3, "Check neighbouring plants for symptoms"), (7, "Re-check the whole field or garden")]
    if key == "high":
        return [(0, "Start the recommended treatment"), (0, "Remove and destroy infected leaves"),
                (5, "Check for spread"), (10, "Second application if symptoms persist"),
                (21, "Final check and note the results")]
    return [(0, "Apply the recommended treatment"), (7, "Check the plant for new spots"),
            (14, "Repeat the application if needed"), (28, "Final check")]


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


# ---------------- HTML builders ----------------
def gauge_svg(value, color):
    value = max(0.0, min(100.0, float(value)))
    filled = 3.14159 * 54 * value / 100
    return ("<div class='gauge'><svg viewBox='0 0 140 88' width='170' height='108'>"
            "<path d='M16 74 A54 54 0 0 1 124 74' fill='none' stroke='#dcebe1' stroke-width='12' stroke-linecap='round'/>"
            f"<path d='M16 74 A54 54 0 0 1 124 74' fill='none' stroke='{color}' stroke-width='12' stroke-linecap='round' stroke-dasharray='{filled:.1f} 400'/>"
            f"<text x='70' y='66' text-anchor='middle' font-size='24' font-weight='700' fill='#12301f'>{value:.0f}%</text>"
            "<text x='70' y='84' text-anchor='middle' font-size='9' fill='#587263'>confidence</text></svg></div>")


def diagnosis_box(cls, conf, risk_label, risk_key):
    color = "#22a860" if conf >= 80 else "#e6b422" if conf >= 60 else "#e0533d"
    return ("<div class='diag'><div class='diag-flex'><div>"
            f"<div class='diag-label'>Diagnosis</div><h2>{pretty_name(cls)}</h2>"
            f"<p>Crop: <b>{crop_of(cls)}</b></p>"
            f"<span class='badge risk-{risk_key}'>{RISK_ICON[risk_key]} Risk: {risk_label}</span></div>"
            f"{gauge_svg(conf, color)}</div></div>")


def is_real_medicine(m):
    m = m.strip().lower()
    return bool(m) and not m.startswith(("no ", "none", "not "))


def flipkart_url(name):
    return "https://www.flipkart.com/search?q=" + quote_plus(name + " pesticide")


def amazon_url(name):
    return "https://www.amazon.in/s?k=" + quote_plus(name + " pesticide")


def medicines_html(cls):
    info = disease_treatments.get(cls, {})
    meds = [m.strip() for m in info.get("medicines", "").split(",") if is_real_medicine(m)]
    if not meds:
        return ("<div class='card'><b>No purchase needed.</b> "
                f"{info.get('medicines', 'Nothing to buy for this result.')}</div>")
    rows = "".join(
        f"<div class='med'><div class='med-name'>💊 {m}</div><div class='med-links'>"
        f"<a class='buy fk' href='{flipkart_url(m)}' target='_blank' rel='noopener'>🛒 Buy on Flipkart</a>"
        f"<a class='buy am' href='{amazon_url(m)}' target='_blank' rel='noopener'>Amazon.in</a></div></div>"
        for m in meds)
    return f"<div class='med-grid'>{rows}</div>"


def bullets(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def whatsapp_url(cls, conf):
    info = disease_treatments.get(cls, {})
    risk_label, _ = risk_level(cls)
    text = (f"🌿 AI Plant Doctor\nCrop: {crop_of(cls)}\nDiagnosis: {pretty_name(cls)} ({conf:.0f}%)\n"
            f"Risk: {risk_label}\nTreatment: {info.get('treatment', '-')}\n"
            f"Medicines: {info.get('medicines', '-')}")
    return "https://wa.me/?text=" + quote(text)


def thumb_b64(img):
    t = img.convert("RGB").copy()
    t.thumbnail((140, 140))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------- PDF report ----------------
def generate_pdf_report(cls, confidence, predictions, image, mode):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    info = disease_treatments.get(cls, {})
    risk_label, _ = risk_level(cls)
    plan = care_plan(cls)

    def header():
        c.setFillColor(HexColor("#0f6b3a"))
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
    c.drawString(40, y, f"Diagnosis: {pretty_name(cls)}")
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Confidence: {confidence:.1f}%     Risk level: {risk_label}     Crop: {crop_of(cls)}")
    y -= 16
    c.setFont("Helvetica", 9)
    eng = "Trained AI model" if mode == "model" else "DEMO mode (simulated prediction)"
    c.drawString(40, y, f"Date: {datetime.now():%d %b %Y, %H:%M}     Engine: {eng}")
    y -= 22

    img_h = 160
    ensure(img_h + 30)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Leaf image")
    y -= 10
    c.drawImage(ImageReader(image), 40, y - img_h, width=img_h, height=img_h,
                preserveAspectRatio=True, anchor="nw")
    y -= img_h + 22

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
    section("Key nutrient focus", info.get("nutrients", "Consult a local expert."))
    section("Organic / natural options", "; ".join(ORGANIC[disease_category(cls)]))
    section("Prevention checklist", "; ".join(PREVENTION))

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


# ---------------- Styling ----------------
LEAF_HERO = ("<svg class='hero-leaf' viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'>"
             "<path d='M100 8C176 40 192 114 100 192C8 114 24 40 100 8Z' fill='rgba(255,255,255,.16)'/>"
             "<path d='M100 30V190M100 72L62 50M100 104L142 78M100 136L58 112M100 164L136 142' "
             "stroke='rgba(255,255,255,.55)' stroke-width='4' fill='none' stroke-linecap='round'/></svg>")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Nunito+Sans:wght@400;600;700&display=swap');
:root {
  --g1: #0f6b3a; --g2: #22a860; --g3: #7fd49f; --mint: #eef8f1; --ink: #12301f; --muted: #587263;
  --line: rgba(15,107,58,.14); --shadow: 0 2px 4px rgba(15,107,58,.05), 0 10px 28px rgba(15,107,58,.09);
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp a, .stApp button, .stApp input { font-family: 'Nunito Sans', system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Outfit', 'Nunito Sans', system-ui, sans-serif; color: var(--ink) !important; letter-spacing: -.3px; }

/* greenery background: soft light + faint leaf pattern */
.stApp {
  background-color: #f3fbf5;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'%3E%3Cg fill='none' stroke='%2322a860' stroke-opacity='.10' stroke-width='2' stroke-linecap='round'%3E%3Cpath d='M40 40C70 52 78 84 40 118C2 84 10 52 40 40Z'/%3E%3Cpath d='M40 52V112'/%3E%3Cpath d='M150 120C176 130 182 158 150 188C118 158 124 130 150 120Z'/%3E%3Cpath d='M150 130V184'/%3E%3C/g%3E%3C/svg%3E"),
    radial-gradient(800px 420px at 100% 0%, rgba(34,168,96,.18), transparent 65%),
    radial-gradient(700px 400px at 0% 100%, rgba(127,212,159,.25), transparent 65%);
  background-attachment: fixed; color: var(--ink);
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 980px !important; padding: 1.4rem 1.5rem 4rem 1.5rem !important; }
.stApp p, .stApp li, .stApp label { color: var(--ink); }
.stApp [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

/* ---------- Hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 30px; padding: 2.3rem 2.5rem; margin-bottom: 1.5rem;
  background: linear-gradient(120deg, var(--g1), var(--g2)); box-shadow: 0 18px 44px rgba(15,107,58,.28);
  animation: rise .6s ease both;
}
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.hero-text { position: relative; z-index: 1; max-width: 600px; }
.stApp .hero h1 { color: #fff !important; font-weight: 800; font-size: clamp(1.9rem, 6vw, 3.2rem); line-height: 1.08; letter-spacing: -1px; margin: 0 0 .5rem 0; padding: 0; }
.hero-sub { color: rgba(255,255,255,.95) !important; font-size: clamp(1rem, 2.6vw, 1.2rem); margin: 0 0 1rem 0; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chips span { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); color: #fff; padding: .28rem .8rem; border-radius: 999px; font-size: .82rem; font-weight: 600; }
.hero-leaf { position: absolute; right: -12px; top: -30px; height: 155%; width: auto; pointer-events: none; }

/* ---------- Scan card ---------- */
.scan-head { text-align: center; margin: .2rem 0 .2rem 0; }
.scan-head h3 { margin: 0; font-size: 1.35rem; }
.scan-head p { color: var(--muted) !important; margin: .2rem 0 0 0; font-size: .95rem; }
.cam-hint { text-align: center; color: var(--muted); font-size: .9rem; margin: .5rem 0 1rem 0; font-weight: 600; }
.or-line { display: flex; align-items: center; gap: .8rem; color: var(--muted); font-size: .85rem; margin: .4rem 0 .8rem 0; }
.or-line::before, .or-line::after { content: ""; flex: 1; height: 1px; background: var(--line); }

/* round camera button (needs Streamlit 1.39+ key classes) */
.st-key-cam_btn { display: flex; justify-content: center; margin-top: .6rem; }
.st-key-cam_btn button {
  width: 104px !important; height: 104px !important; min-height: 104px; border-radius: 50% !important; padding: 0 !important;
  background: radial-gradient(circle at 30% 25%, var(--g3), var(--g2) 45%, var(--g1)) !important; border: 4px solid #fff !important;
  box-shadow: 0 0 0 8px rgba(34,168,96,.20), 0 14px 30px rgba(15,107,58,.35) !important;
  transition: transform .2s ease, box-shadow .2s ease; animation: pulse 2.4s ease-in-out infinite;
}
.st-key-cam_btn button p { font-size: 2.6rem !important; line-height: 1 !important; color: #fff !important; }
.st-key-cam_btn button:hover { transform: scale(1.06); box-shadow: 0 0 0 11px rgba(34,168,96,.24), 0 18px 36px rgba(15,107,58,.4) !important; }
.st-key-cam_btn button:active { transform: scale(.95); }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 8px rgba(34,168,96,.20), 0 14px 30px rgba(15,107,58,.35); } 50% { box-shadow: 0 0 0 15px rgba(34,168,96,.10), 0 14px 30px rgba(15,107,58,.35); } }

div[data-testid="stCameraInput"], div[data-testid="stFileUploader"] section {
  background: #fff !important; border: 2px dashed rgba(34,168,96,.55) !important; border-radius: 22px !important;
}
div[data-testid="stFileUploader"] section:hover { border-color: var(--g1) !important; background: #f4fcf6 !important; }
div[data-testid="stFileUploader"] section *, div[data-testid="stCameraInput"] label { color: var(--ink) !important; }
div[data-testid="stCameraInput"] button { border-radius: 999px !important; }
div[data-testid="stImage"] img { border-radius: 20px; border: 1px solid var(--line); box-shadow: var(--shadow); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, var(--g1), var(--g2)); color: #fff !important; font-weight: 700; border: none;
  border-radius: 999px; padding: .7rem 1.5rem; box-shadow: 0 8px 20px rgba(15,107,58,.26);
  transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button p, .stDownloadButton > button p { color: #fff !important; }
.stDownloadButton > button:hover, .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(15,107,58,.34); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid rgba(34,168,96,.5); outline-offset: 3px; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; background: rgba(255,255,255,.8); border: 1px solid var(--line); border-radius: 999px; padding: 5px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 999px; padding: .45rem 1rem; height: auto; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 700; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--g1), var(--g2)) !important; }
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.stProgress > div > div > div > div { background-image: linear-gradient(90deg, var(--g1), var(--g2)); }
div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: .8rem 1rem; box-shadow: var(--shadow); }
div[data-testid="stMetricValue"] { color: var(--g1) !important; font-family: 'Outfit', sans-serif; }
div[data-testid="stExpander"] { background: #fff; border: 1px solid var(--line); border-radius: 16px; }
div[data-testid="stAlert"] { border-radius: 14px; }

/* ---------- Diagnosis ---------- */
.diag { position: relative; overflow: hidden; background: #fff; border: 1px solid var(--line); border-radius: 24px; padding: 1.2rem 1.6rem 1.2rem 2rem; margin: 1rem 0; box-shadow: var(--shadow); }
.diag::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 9px; background: linear-gradient(180deg, var(--g2), var(--g1)); }
.diag-flex { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .6rem; }
.diag-label { color: var(--muted); font-size: .85rem; font-weight: 700; }
.diag h2 { margin: .1rem 0 .2rem 0; font-size: clamp(1.3rem, 4.5vw, 2rem); }
.diag p { margin: 0; }
.gauge { margin: 0 auto; }
.badge { display: inline-block; padding: .25rem .8rem; border-radius: 999px; font-weight: 700; font-size: .85rem; margin-top: .5rem; }
.risk-healthy  { background: #e3f8ea; color: #14803f; border: 1px solid #7fd4a0; }
.risk-moderate { background: #fff6d6; color: #8a6a00; border: 1px solid #f0cf5a; }
.risk-high     { background: #ffe9d6; color: #b0540b; border: 1px solid #f3a766; }
.risk-critical { background: #fde4e1; color: #b3261e; border: 1px solid #f0948d; }

/* ---------- Cards ---------- */
.card { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.4rem; margin: .6rem 0; box-shadow: var(--shadow); }
.card h4 { margin: 0 0 .3rem 0; color: var(--g1) !important; font-size: 1.05rem; }
.card p { margin: 0; }
.card ul { margin: .2rem 0 0 0; padding-left: 1.2rem; }
.card li { margin: .25rem 0; }
.note { background: rgba(34,168,96,.09); border: 1px dashed rgba(15,107,58,.35); border-radius: 14px; padding: .8rem 1rem; margin-top: 1rem; font-size: .92rem; }
.demo-banner { background: #fff8e1; border: 1px solid #f5d77a; color: #7a5b00; border-radius: 12px; padding: .5rem .9rem; font-size: .88rem; margin-bottom: .8rem; }

.med-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .9rem; margin: .6rem 0; }
.med { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.1rem; box-shadow: var(--shadow); }
.med-name { font-weight: 700; margin-bottom: .6rem; }
.med-links { display: flex; flex-wrap: wrap; gap: .5rem; }
.buy { display: inline-block; padding: .45rem 1rem; border-radius: 999px; font-weight: 700; font-size: .85rem; text-decoration: none !important; }
.buy.fk { background: #2874f0; color: #fff !important; box-shadow: 0 6px 14px rgba(40,116,240,.28); }
.buy.am { background: #fff; color: var(--g1) !important; border: 1px solid var(--line); }
.buy:hover { filter: brightness(1.06); }

.share-btn { display: inline-block; width: 100%; text-align: center; box-sizing: border-box; background: #25D366; color: #06210f !important; font-weight: 700; padding: .7rem 1.2rem; border-radius: 999px; text-decoration: none !important; box-shadow: 0 8px 18px rgba(37,211,102,.28); }

.timeline { border-left: 3px solid rgba(34,168,96,.45); margin: .5rem 0 1rem .6rem; padding-left: 1.2rem; }
.tl-item { position: relative; margin-bottom: 1rem; }
.tl-item::before { content: ""; position: absolute; left: -1.7rem; top: .35rem; width: 13px; height: 13px; border-radius: 50%; background: var(--g2); box-shadow: 0 0 0 4px rgba(34,168,96,.2); }
.tl-day { font-weight: 700; color: var(--g1); font-size: .9rem; }

.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: .9rem; margin: 1.2rem 0; }
.step { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 1rem; text-align: center; box-shadow: var(--shadow); }
.step .ico { font-size: 1.9rem; } .step b { display: block; color: var(--g1); margin: .25rem 0; } .step span { color: var(--muted); font-size: .88rem; }

.recent { display: flex; gap: .8rem; overflow-x: auto; padding: .3rem .2rem 1rem .2rem; scroll-snap-type: x proximity; }
.rc { flex: 0 0 130px; scroll-snap-align: start; background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: .5rem; box-shadow: var(--shadow); }
.rc img { width: 100%; height: 90px; object-fit: cover; border-radius: 12px; }
.rc b { display: block; font-size: .78rem; margin-top: .35rem; line-height: 1.2; }
.rc span { font-size: .72rem; color: var(--muted); }

.disclaimer { text-align: center; color: var(--muted); font-size: .82rem; margin-top: 1.6rem; }

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
  .block-container { padding: 1rem .9rem 3rem .9rem !important; }
  .hero { padding: 1.5rem 1.3rem; border-radius: 22px; }
  .hero-leaf { opacity: .45; }
  .diag { padding: 1rem 1rem 1rem 1.4rem; border-radius: 18px; }
  .diag-flex { justify-content: center; text-align: center; }
  .stButton > button, .stDownloadButton > button { width: 100%; }
  .st-key-cam_btn button { width: 92px !important; height: 92px !important; min-height: 92px; }
  .stTabs [data-baseweb="tab-list"] { border-radius: 20px; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""


# ---------------- Helpers ----------------
def read_image(f):
    return ImageOps.exif_transpose(Image.open(f)).convert("RGB")


def analyze(image, normalize):
    """Runs the model, builds the PDF, returns a result dict."""
    preds, mode = predict_disease(image, top_n=3, normalize=normalize)
    cls, conf = preds[0]["class"], preds[0]["confidence"]
    q_score, q_issues = photo_quality(image)
    return {"cls": cls, "confidence": conf, "predictions": preds, "mode": mode, "image": image,
            "quality": q_score, "issues": q_issues,
            "pdf": generate_pdf_report(cls, conf, preds, image, mode)}


def render_result(res, min_conf):
    cls, conf = res["cls"], res["confidence"]
    info = disease_treatments.get(cls, {})
    risk_label, risk_key = risk_level(cls)

    if res["mode"] == "demo":
        st.markdown("<div class='demo-banner'>🧪 DEMO result: no trained model found, so this prediction is simulated.</div>",
                    unsafe_allow_html=True)
    st.markdown(diagnosis_box(cls, conf, risk_label, risk_key), unsafe_allow_html=True)
    if conf < min_conf:
        st.warning("⚠️ Low confidence. Please check the leaf yourself or retake a clearer photo.")

    b1, b2 = st.columns(2)
    with b1:
        st.download_button("📄 Download PDF report", data=res["pdf"],
                           file_name=f"plant_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                           mime="application/pdf", key="pdf_dl", **STRETCH)
    with b2:
        st.markdown(f"<a class='share-btn' href='{whatsapp_url(cls, conf)}' target='_blank' rel='noopener'>💬 Share on WhatsApp</a>",
                    unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["🛠️ Cure & tips", "💊 Medicines", "🌿 Organic & prevention", "📅 Care plan"])
    with t1:
        if info:
            st.markdown(f"<div class='card'><h4>🛠️ How to treat it</h4><p>{info['treatment']}</p></div>"
                        f"<div class='card'><h4>💡 Suggestions</h4><p>{info['suggestions']}</p></div>"
                        f"<div class='card'><h4>🌱 Nutrient focus</h4><p>{info['nutrients']}</p></div>",
                        unsafe_allow_html=True)
        else:
            st.info("No detailed treatment information is available for this class yet.")
        if len(res["predictions"]) > 1:
            st.markdown("**Other possibilities**")
            for r in res["predictions"][1:]:
                st.progress(min(max(r["confidence"] / 100, 0.0), 1.0),
                            text=f"{pretty_name(r['class'])}: {r['confidence']:.1f}%")
        for icon, msg in res["issues"]:
            st.write(f"{icon} {msg}")
        st.markdown(EXPERT_HTML, unsafe_allow_html=True)
    with t2:
        st.markdown(f"<div class='card'><h4>Recommended medicines</h4><p>{info.get('medicines', '-')}</p></div>",
                    unsafe_allow_html=True)
        st.markdown(medicines_html(cls), unsafe_allow_html=True)
        st.markdown(f"<div class='card'><h4>🧤 Spray safely</h4>{bullets(SPRAY_SAFETY)}</div>", unsafe_allow_html=True)
        st.caption("Links open a Flipkart search for the medicine. Always read the product label and follow local guidelines.")
    with t3:
        st.markdown(f"<div class='card'><h4>🌿 Organic and natural options</h4>{bullets(ORGANIC[disease_category(cls)])}</div>"
                    f"<div class='card'><h4>🛡️ Prevention checklist</h4>{bullets(PREVENTION)}</div>",
                    unsafe_allow_html=True)
    with t4:
        plan = care_plan(cls)
        items = ""
        for d, task in plan:
            when = "Today" if d == 0 else f"Day {d}"
            date = (datetime.now() + timedelta(days=d)).strftime("%d %b")
            items += f"<div class='tl-item'><div class='tl-day'>{when} · {date}</div><div>{task}</div></div>"
        st.markdown(f"<div class='timeline'>{items}</div>", unsafe_allow_html=True)
        st.download_button("📆 Add reminders to my calendar", data=build_ics(cls, plan),
                           file_name="plant_care_plan.ics", mime="text/calendar", key="ics_dl")


# =====================================================================
#                                UI
# =====================================================================
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    "<div class='hero'><div class='hero-text'>"
    "<h1>🌿 AI Plant Doctor</h1>"
    "<p class='hero-sub'>Snap a leaf. Get the diagnosis, cure and medicines in seconds.</p>"
    "<div class='chips'><span>📷 Camera</span><span>🧠 AI diagnosis</span><span>💊 Medicines</span><span>📄 PDF report</span></div>"
    f"</div>{LEAF_HERO}</div>",
    unsafe_allow_html=True)

# ---- state ----
st.session_state.setdefault("cam_open", False)
st.session_state.setdefault("result", None)
st.session_state.setdefault("recent", [])

# ---- scan area ----
st.markdown("<div class='scan-head'><h3>Scan a leaf</h3>"
            "<p>Tap the camera, or upload a photo from your gallery.</p></div>", unsafe_allow_html=True)

if st.button("✕" if st.session_state["cam_open"] else "📷", key="cam_btn"):
    st.session_state["cam_open"] = not st.session_state["cam_open"]
    st.rerun()
st.markdown(f"<div class='cam-hint'>{'Tap ✕ to close the camera' if st.session_state['cam_open'] else 'Tap to open the camera'}</div>",
            unsafe_allow_html=True)

image_obj = None
if st.session_state["cam_open"]:
    cam = st.camera_input("Take a clear close-up of one leaf", label_visibility="collapsed")
    if cam:
        image_obj = read_image(cam)

st.markdown("<div class='or-line'>or upload from gallery</div>", unsafe_allow_html=True)
up = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if up and image_obj is None:
    image_obj = read_image(up)

with st.expander("⚙️ Advanced settings"):
    normalize = st.checkbox("Scale pixels to 0–1 (÷255)", value=True,
                            help="Turn OFF if your model was trained on raw 0–255 pixels or has its own rescaling layer.")
    min_conf = st.slider("Low-confidence warning below (%)", 30, 95, 70)
    _model, _model_path, _model_err = get_model()
    if _model is not None:
        w_, h_ = model_input_size(_model)
        st.caption(f"🧠 Model: {os.path.basename(_model_path)} · input {w_}×{h_} · {len(class_names)} classes")
    else:
        st.caption("🧪 Demo mode: no trained model loaded." + (f" Reason: {_model_err[:160]}" if _model_err else ""))

# ---- analyse (automatically, once per new photo) ----
if image_obj is None:
    st.markdown(
        "<div class='steps'>"
        "<div class='step'><div class='ico'>📷</div><b>1. Take a photo</b><span>Close-up of one leaf in daylight.</span></div>"
        "<div class='step'><div class='ico'>🧠</div><b>2. Get the diagnosis</b><span>Disease, confidence and risk level.</span></div>"
        "<div class='step'><div class='ico'>💊</div><b>3. Treat it</b><span>Cure tips, Flipkart links and a PDF report.</span></div>"
        "</div>", unsafe_allow_html=True)
    with st.expander("📸 Tips for a perfect photo"):
        st.markdown("- Use daylight; avoid flash and harsh shadows\n- Fill the frame with **one** leaf on a plain background\n"
                    "- Keep the camera steady and in focus\n- Photograph the affected area, not just a healthy part\n"
                    "- Take 2–3 photos of different leaves for more confidence")
else:
    img_id = hashlib.md5(image_obj.resize((64, 64)).tobytes()).hexdigest()
    cached = st.session_state["result"]
    if not cached or cached.get("img_id") != img_id:
        with st.spinner("🔍 Analysing your leaf..."):
            res = analyze(image_obj, normalize)
        res["img_id"] = img_id
        st.session_state["result"] = res
        st.session_state["recent"] = ([{"name": pretty_name(res["cls"]), "conf": res["confidence"],
                                        "risk": risk_level(res["cls"])[1], "thumb": thumb_b64(image_obj),
                                        "time": datetime.now().strftime("%d %b, %H:%M")}]
                                      + st.session_state["recent"])[:8]
        if risk_level(res["cls"])[1] == "healthy" and res["confidence"] >= 85:
            st.balloons()
        cached = res

    col_img, col_q = st.columns([1, 1])
    with col_img:
        st.image(image_obj, caption="Your leaf", **STRETCH)
    with col_q:
        st.markdown("#### 📸 Photo check")
        st.progress(cached["quality"] / 100, text=f"Quality score: {cached['quality']}/100")
        if not cached["issues"]:
            st.write("🟢 Looks good. Sharp and well lit.")
    render_result(cached, min_conf)

# ---- recent scans ----
recent = st.session_state["recent"]
if recent:
    st.markdown("### 🕘 Recent scans")
    cards = "".join(
        f"<div class='rc'><img src='data:image/jpeg;base64,{r['thumb']}' alt='leaf'/>"
        f"<b>{RISK_ICON[r['risk']]} {r['name']}</b><span>{r['conf']:.0f}% · {r['time']}</span></div>"
        for r in recent)
    st.markdown(f"<div class='recent'>{cards}</div>", unsafe_allow_html=True)
    if st.button("Clear recent scans", key="clear_recent"):
        st.session_state["recent"] = []
        st.session_state["result"] = None
        st.rerun()

st.markdown("<div class='disclaimer'>⚠️ For information only. Confirm with a local agricultural expert before treating crops.</div>",
            unsafe_allow_html=True)
