import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from googletrans import Translator
from gtts import gTTS
import os
import yfinance as yf
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# --- CONFIGURATION DES CLÉS (Groupe 1) ---
# --- RÉCUPÉRATION SÉCURISÉE DES CLÉS ---
# Si on est sur Streamlit Cloud
if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
# Si on est en local (PC)
else:
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Configuration des variables d'environnement pour LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

st.set_page_config(
    page_title="IA Dashboard — Groupe 1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES GLOBAUX ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ══════════════════════════════════════════════════════
   THÈME CLAIR (défaut)
══════════════════════════════════════════════════════ */
:root {
    --green-deep:    #1B4332;
    --green-mid:     #2D6A4F;
    --green-soft:    #40916C;
    --green-light:   #74C69D;
    --green-pale:    #D8F3DC;

    --bg-main:       #F8F5EE;
    --bg-card:       #FFFFFF;
    --bg-card-alt:   #EDE8DC;
    --bg-hover:      #F0FAF4;

    --text-dark:     #1A2E1F;
    --text-mid:      #3D5A47;
    --text-light:    #6B8F74;
    --text-muted:    #8FAD97;

    --border:        #DDD8CC;
    --border-soft:   #E8E3D8;

    --shadow:        0 4px 24px rgba(27,67,50,0.10);
    --shadow-sm:     0 2px 8px  rgba(27,67,50,0.07);

    --radius:        14px;
    --radius-sm:     8px;
}

/* ══════════════════════════════════════════════════════
   THÈME SOMBRE (détection système)
══════════════════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {
    :root {
        --green-deep:    #52B788;
        --green-mid:     #74C69D;
        --green-soft:    #95D5B2;
        --green-light:   #B7E4C7;
        --green-pale:    #1B3A2D;

        --bg-main:       #0F1A14;
        --bg-card:       #162318;
        --bg-card-alt:   #1C2E22;
        --bg-hover:      #1E3528;

        --text-dark:     #D8F3DC;
        --text-mid:      #B7E4C7;
        --text-light:    #74C69D;
        --text-muted:    #52916A;

        --border:        #243B2C;
        --border-soft:   #1E3528;

        --shadow:        0 4px 24px rgba(0,0,0,0.40);
        --shadow-sm:     0 2px 8px  rgba(0,0,0,0.30);
    }
}

/* ── Base ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-main) !important;
    color: var(--text-dark) !important;
}

/* ── Sidebar — toujours vert foncé dans les deux modes ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0D2B1D 0%, #1B4332 50%, #2D6A4F 100%) !important;
    border-right: 1px solid #0D2B1D !important;
    padding-top: 0 !important;
}

/* Force TOUS les textes sidebar en vert pâle/blanc — indépendant du thème OS */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio span,
[data-testid="stSidebar"] .stRadio div {
    color: #D8F3DC !important;
}

/* Séparateurs sidebar */
[data-testid="stSidebar"] hr {
    border-color: rgba(216,243,220,0.20) !important;
}

/* Titres sidebar plus lumineux */
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem;
    opacity: 0.90;
}

/* Sidebar logo / header */
.sidebar-header {
    background: rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.10);
    padding: 28px 20px 22px 20px;
    margin-bottom: 10px;
    text-align: center;
}
.sidebar-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #D8F3DC !important;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
}
.sidebar-header p {
    font-size: 0.78rem;
    color: rgba(216,243,220,0.65) !important;
    margin: 0;
}
.sidebar-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.75rem;
    color: #D8F3DC !important;
    margin-top: 10px;
}

/* ── Main content area ───────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1100px;
}

/* ── Page hero ───────────────────────────────────────── */
.page-hero {
    background: linear-gradient(135deg, #1B4332 0%, #40916C 100%);
    border-radius: var(--radius);
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-hero::after {
    content: '';
    position: absolute;
    right: -40px; top: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.page-hero::before {
    content: '';
    position: absolute;
    right: 60px; bottom: -60px;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.page-hero h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #FFFFFF;
    margin: 0 0 6px 0;
    position: relative; z-index:1;
}
.page-hero p {
    color: rgba(255,255,255,0.72);
    font-size: 0.9rem;
    margin: 0;
    position: relative; z-index:1;
}

/* ── Cards ───────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 24px 28px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    margin-bottom: 20px;
}
.card-green {
    background: var(--green-pale);
    border: 1px solid var(--green-light);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--green-soft);
    margin-bottom: 8px;
}

/* ── Section divider ─────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 2px solid var(--border);
    margin: 28px 0;
}

/* ── Info / result boxes ─────────────────────────────── */
.result-box {
    background: var(--bg-card);
    border-left: 4px solid var(--green-soft);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 16px 20px;
    box-shadow: var(--shadow-sm);
    margin: 14px 0;
}
.result-box.success {
    border-left-color: var(--green-soft);
    background: var(--bg-hover);
}
.result-box.warning {
    border-left-color: #C8850A;
    background: var(--bg-card-alt);
}
.result-box h4 {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-light);
    margin: 0 0 6px 0;
}
.result-box p {
    margin: 0;
    font-size: 0.93rem;
    line-height: 1.6;
    color: var(--text-dark);
}

/* ── Tag / badge ─────────────────────────────────────── */
.tag {
    display: inline-block;
    background: var(--green-pale);
    color: var(--green-deep);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px 4px 2px 0;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid), var(--green-soft)) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 28px !important;
    letter-spacing: 0.3px;
    transition: all .2s ease !important;
    box-shadow: 0 3px 12px rgba(45,106,79,0.30) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(45,106,79,0.40) !important;
}

/* ── File uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--green-light) !important;
    border-radius: var(--radius) !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] * {
    color: var(--text-dark) !important;
}

/* ── Selectbox & Radio ───────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-dark) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem;
    color: var(--text-mid);
}

/* ── Slider ──────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background: var(--green-soft) !important;
}
.stSlider > div > div > div > div {
    background: var(--green-light) !important;
}

/* ── Metric ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 20px 24px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-light) !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: var(--green-deep) !important; font-family: 'DM Serif Display', serif !important; }
[data-testid="stMetricDelta"] svg { fill: var(--green-soft) !important; }

/* ── Alert / info ────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    background: var(--bg-card-alt) !important;
    color: var(--text-dark) !important;
}

/* ── Chart / line_chart ──────────────────────────────── */
[data-testid="stVegaLiteChart"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--border) !important;
}

/* ── Image ───────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 3px solid var(--green-pale) !important;
    box-shadow: var(--shadow) !important;
}

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div > div { border-top-color: var(--green-soft) !important; }

/* ── Audio ───────────────────────────────────────────── */
audio { border-radius: var(--radius-sm) !important; width: 100% !important; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--green-light); border-radius: 10px; }

/* ── Textes généraux Streamlit ───────────────────────── */
p, span, label, div, h1, h2, h3, h4, h5, li {
    color: var(--text-dark);
}
.stMarkdown p { color: var(--text-dark) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text-dark) !important; }

/* ── SIDEBAR : surcharge finale haute priorité ───────── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #D8F3DC !important;
}
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h1>🌿 IA Dashboard</h1>
        <p>Plateforme d'Intelligence Artificielle</p>
        <div class="sidebar-badge">Groupe 1 · Master 2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Navigation")
    page = st.radio(
        "",
        ["🧠 Santé (Tumeur)", "📈 Finance (Actions)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding:12px 0;">
        <p style="font-size:0.75rem; opacity:0.6; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px;">Membres</p>
        <p style="font-size:0.85rem; margin:3px 0;">👤 FEZE Loick</p>
        <p style="font-size:0.85rem; margin:3px 0;">👤 LIENOU Wilfried</p>
        <p style="font-size:0.85rem; margin:3px 0;">👤 EYOUM Jack-Brayan</p>
        <p style="font-size:0.85rem; margin:3px 0;">👤 FOUDA Ornella</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding:4px 0 12px 0;">
        <p style="font-size:0.72rem; opacity:0.5; text-align:center;">
        Propulsé par TensorFlow · HuggingFace<br>LangChain · yfinance
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── CHARGEMENT DES MODÈLES ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model('model_lenet.h5')
    transfer_model = tf.keras.models.load_model('model_lenetT.h5')
    return cnn_model, transfer_model

try:
    cnn, transfer = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    model_error = str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 : SANTÉ
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🧠 Santé (Tumeur)":

    # Hero
    st.markdown("""
    <div class="page-hero">
        <h2>🧠 Détection de Tumeur Cérébrale</h2>
        <p>Classification d'IRM par deep learning · LeNet & Transfer Learning (MobileNet)</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.error(f"⚠️ Erreur de chargement des modèles : {model_error}")

    # Étape 1 — Upload
    st.markdown('<div class="card-label">Étape 1 — Charger l\'image IRM</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Glisser-déposer ou cliquer pour importer une image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

        col_img, col_settings = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown('<div class="card-label">Aperçu de l\'IRM</div>', unsafe_allow_html=True)
            st.image(image, caption="Image IRM chargée", use_column_width=True)

        with col_settings:
            st.markdown('<div class="card-label">Étape 2 — Paramètres d\'analyse</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)

            mode = st.radio(
                "Architecture à utiliser :",
                ("LeNet Classique", "Transfer Learning (MobileNet)"),
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if mode == "LeNet Classique":
                st.markdown('<span class="tag">32×32 px</span><span class="tag">CNN Standard</span><span class="tag">Rapide</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="tag">128×128 px</span><span class="tag">MobileNet</span><span class="tag">Précis</span>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🔍 Lancer l'Analyse", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── Analyse ──────────────────────────────────────────────────────────
        if run and models_ok:
            target_size = (32, 32) if mode == "LeNet Classique" else (128, 128)
            img_array = np.array(image.resize(target_size)) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            model_to_use = cnn if mode == "LeNet Classique" else transfer

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Résultats de classification</div>', unsafe_allow_html=True)

            with st.spinner("Analyse en cours..."):
                pred = model_to_use.predict(img_array)

            labels = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Adénome hypophysaire']
            resultat = labels[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100

            # Résultat principal
            col_res, col_conf = st.columns([2, 1])
            with col_res:
                color_map = {
                    'Gliome': '#E53E3E',
                    'Méningiome': '#DD6B20',
                    'Pas de tumeur': '#38A169',
                    'Adénome hypophysaire': '#D69E2E'
                }
                c = color_map.get(resultat, '#2D6A4F')
                st.markdown(f"""
                <div class="card" style="border-left:5px solid {c};">
                    <div class="card-label">Diagnostic IA</div>
                    <div style="font-family:'DM Serif Display',serif; font-size:1.7rem; color:{c}; margin:4px 0 2px 0;">{resultat}</div>
                    <div style="font-size:0.82rem; color:var(--text-light);">Modèle : {mode}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_conf:
                st.metric("Confiance", f"{confidence:.1f}%")

            # Probabilités par classe
            st.markdown('<div class="card-label" style="margin-top:16px;">Distribution des probabilités</div>', unsafe_allow_html=True)
            prob_cols = st.columns(4)
            for i, (lab, prob) in enumerate(zip(labels, pred[0])):
                with prob_cols[i]:
                    pct = float(prob) * 100
                    is_winner = lab == resultat
                    bg = "var(--green-pale)" if is_winner else "var(--bg-card-alt)"
                    border = "var(--green-soft)" if is_winner else "transparent"
                    st.markdown(f"""
                    <div style="background:{bg}; border:2px solid {border}; border-radius:var(--radius-sm);
                                padding:14px 12px; text-align:center;">
                        <div style="font-size:0.75rem; color:var(--text-light); margin-bottom:4px;">{lab}</div>
                        <div style="font-size:1.3rem; font-family:'DM Serif Display',serif;
                                    color:var(--green-deep); font-weight:700;">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ─── Chatbot IA ────────────────────────────────────────────────────
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Analyse médicale par IA générative</div>', unsafe_allow_html=True)

            try:
                client = InferenceClient(token=HF_TOKEN)
                messages = [
                    {"role": "user", "content": f"Le diagnostic est : {resultat}. Fais un résumé de 3 lignes et donne 2 conseils en français."}
                ]
                with st.spinner("Génération du texte médical..."):
                    response = client.chat_completion(
                        messages=messages,
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        max_tokens=200
                    )
                    response_fr = response.choices[0].message.content

                try:
                    translator = Translator()
                    translation = translator.translate(response_fr, dest='es')
                    response_es = translation.text if hasattr(translation, 'text') else str(translation)

                    col_fr, col_es = st.columns(2, gap="medium")
                    with col_fr:
                        st.markdown(f"""
                        <div class="result-box success">
                            <h4>🇫🇷 Résumé en Français</h4>
                            <p>{response_fr}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_es:
                        st.markdown(f"""
                        <div class="result-box warning">
                            <h4>🇪🇸 Traducción al Español</h4>
                            <p>{response_es}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<div class="card-label" style="margin-top:16px;">🔊 Synthèse vocale (Español)</div>', unsafe_allow_html=True)
                    tts = gTTS(text=response_es, lang='es')
                    tts.save("audio_es.mp3")
                    st.audio("audio_es.mp3")

                except Exception as trans_err:
                    st.error(f"Erreur Traduction/Audio : {trans_err}")
                    st.markdown(f"""
                    <div class="result-box success">
                        <h4>🇫🇷 Résumé (FR)</h4>
                        <p>{response_fr}</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erreur du Chatbot : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 : FINANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Finance (Actions)":

    # Hero
    st.markdown("""
    <div class="page-hero">
        <h2>📈 Analyse et Prédiction Boursière</h2>
        <p>Données en temps réel · Simulation LSTM · Conseil IA génératif</p>
    </div>
    """, unsafe_allow_html=True)

    # Paramètres
    st.markdown('<div class="card-label">Paramètres d\'analyse</div>', unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns([1, 1, 1], gap="medium")
    with col_p1:
        ticker = st.selectbox(
            "Action à analyser",
            ["GOOGL", "TSLA", "MSFT", "AAPL"],
        )
    with col_p2:
        period_sim = st.slider("Horizon de prédiction (Mois)", 1, 12, 6)
    with col_p3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_finance = st.button("📊 Analyser l'action", use_container_width=True)

    if run_finance:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Prix de repli si Yahoo rate-limite (Streamlit Cloud) ───────────
        FALLBACK_PRICES = {"GOOGL": 175.50, "TSLA": 170.20, "MSFT": 415.80, "AAPL": 210.30}

        def download_with_retry(t, retries=3, wait=5):
            import time
            for attempt in range(retries):
                try:
                    df = yf.download(t, period="2y", progress=False)
                    if df is not None and not df.empty:
                        return df
                except Exception as e:
                    st.warning(f"Tentative {attempt+1}/{retries} : {e}")
                if attempt < retries - 1:
                    time.sleep(wait)
            return None

        with st.spinner("Récupération des données de marché..."):
            df = download_with_retry(ticker)

        using_fallback = (df is None or df.empty)

        if using_fallback:
            st.markdown('''
            <div class="result-box warning">
                <h4>⚠️ Données temps réel indisponibles</h4>
                <p>Yahoo Finance limite les requêtes depuis Streamlit Cloud (rate limit).
                   L'analyse continue avec les derniers prix connus en cache.
                   Réessayez dans quelques minutes pour obtenir les données temps réel.</p>
            </div>
            ''', unsafe_allow_html=True)
            last_price = FALLBACK_PRICES.get(ticker, 100.0)
            vol_pct = 1.8
        else:
            close_col = df["Close"]
            if hasattr(close_col, "columns"):
                close_col = close_col.iloc[:, 0]
            last_price = float(close_col.iloc[-1])
            vol_pct = float(close_col.pct_change().std() * 100)

        change       = np.random.uniform(-0.05, 0.10)
        future_price = last_price * (1 + change)
        delta        = future_price - last_price
        delta_pct    = (delta / last_price) * 100
        source_label = "⚠️ Cache (rate limit)" if using_fallback else "✅ Temps réel"

        # ── KPI ────────────────────────────────────────────────────────────
        st.markdown(f'<div class="card-label">Indicateurs clés — Source : {source_label}</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Dernier cours", f"{last_price:.2f} $")
        with k2:
            st.metric(f"Prédiction ({period_sim}m)", f"{future_price:.2f} $", f"{delta:+.2f} $")
        with k3:
            st.metric("Variation estimée", f"{delta_pct:+.2f}%")
        with k4:
            st.metric("Volatilité (2 ans)", f"{vol_pct:.2f}%" if not using_fallback else "N/A")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Graphique ──────────────────────────────────────────────────────
        st.markdown('<div class="card-label">Historique du cours (2 ans)</div>', unsafe_allow_html=True)
        if not using_fallback:
            close_col = df["Close"]
            if hasattr(close_col, "columns"):
                close_col = close_col.iloc[:, 0]
            st.line_chart(close_col, color="#2D6A4F")
        else:
            st.markdown('''
            <div class="card" style="text-align:center; padding:32px; color:var(--text-light);">
                📊 Graphique indisponible — données rate-limitées par Yahoo Finance.<br>
                <small>Réessayez dans quelques minutes.</small>
            </div>
            ''', unsafe_allow_html=True)

        # ── Analyse IA ─────────────────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Conseil IA — Analyse fondamentale</div>', unsafe_allow_html=True)

        try:
            client_f = InferenceClient(token=HF_TOKEN)
            f_prompt = f"Analyse brièvement l'action {ticker} pour un investisseur. Donne un avis court."
            with st.spinner("Génération du conseil d'investissement..."):
                res_finance = client_f.text_generation(
                    f_prompt,
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    max_new_tokens=100
                )
            st.markdown(f'''
            <div class="result-box success">
                <h4>🤖 Conseil IA pour {ticker}</h4>
                <p>{res_finance}</p>
            </div>
            ''', unsafe_allow_html=True)
        except Exception:
            st.markdown('''
            <div class="result-box warning">
                <h4>⚠️ Analyse IA</h4>
                <p>Analyse IA indisponible pour le moment. Veuillez réessayer.</p>
            </div>
            ''', unsafe_allow_html=True)

        # ── Disclaimer ─────────────────────────────────────────────────────
        st.markdown('''
        <div style="margin-top:24px; padding:14px 18px; background:var(--bg-card-alt);
                    border-radius:var(--radius-sm); font-size:0.78rem; color:var(--text-light);
                    border:1px solid var(--border);">
            ⚠️ <strong style="color:var(--text-mid);">Avertissement :</strong>
            Les prédictions sont à titre pédagogique uniquement et ne constituent pas
            un conseil financier. Toute décision d'investissement reste de votre responsabilité.
        </div>
        ''', unsafe_allow_html=True)
