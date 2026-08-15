"""
Streamlit front-end for the Fake News Detection & Fact Verification Agent.

Run with:
    streamlit run app.py
"""
import os
import sys
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.agent import FakeNewsAgent

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Fact Desk",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#11151C"
PANEL = "#1A2029"
PAPER = "#E8E6DE"
MUTED = "#7C8595"
WIRE_TEAL = "#3FA796"
ALARM = "#E4572E"
GOLD = "#D8A93B"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Serif', serif;
    color: {PAPER};
}}

.wire-header {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-size: 13px;
    color: {MUTED};
    border-top: 1px solid {MUTED};
    border-bottom: 1px solid {MUTED};
    padding: 8px 0;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
}}

.masthead {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 42px;
    letter-spacing: 1px;
    color: {PAPER};
    margin-bottom: 0px;
    line-height: 1.1;
}}

.masthead-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: {GOLD};
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
}}

.stamp-box {{
    font-family: 'IBM Plex Mono', monospace;
    display: inline-block;
    border: 4px solid var(--stamp-color);
    color: var(--stamp-color);
    padding: 14px 28px;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 4px;
    transform: rotate(-3deg);
    border-radius: 4px;
    margin: 10px 0 4px 0;
}}

.telex {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: {MUTED};
    letter-spacing: 1px;
}}

.model-row {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 10px 14px;
    background: {PANEL};
    border-left: 3px solid var(--row-color);
    margin-bottom: 8px;
    border-radius: 2px;
}}

.claim-card {{
    background: {PANEL};
    border-left: 3px solid {GOLD};
    padding: 10px 16px;
    margin-bottom: 8px;
    font-size: 15px;
    font-style: italic;
    color: {PAPER};
}}

.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {GOLD};
    border-bottom: 1px solid #3A4150;
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 14px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header / masthead
# ---------------------------------------------------------------------------
now = datetime.now().strftime("%a %d %b %Y — %H:%M")
st.markdown(
    f"""<div class="wire-header"><span>AGENT STATUS: {{status}}</span><span>{now}</span></div>""".replace(
        "{status}", "ONLINE"
    ),
    unsafe_allow_html=True,
)
st.markdown('<div class="masthead">AI FACT DESK</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="masthead-sub">Fake News Detection &amp; Fact Verification Agent · BERT · RoBERTa · DeBERTa</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load the agent once (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Booting the fact desk (loading models)...")
def load_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FakeNewsAgent(device=device)


agent = load_agent()

with st.sidebar:
    st.markdown('<div class="section-label">Desk Status</div>', unsafe_allow_html=True)
    if agent.available_models:
        for key in agent.available_models:
            st.markdown(f"🟢 `{config.MODEL_REGISTRY[key]['display_name']}` — loaded")
    else:
        st.warning(
            "No fine-tuned models found in `models/`. Running on a lightweight "
            "heuristic fallback so the UI stays usable — train the models "
            "(`src/train.py`) for real predictions.",
            icon="⚠️",
        )

    st.markdown('<div class="section-label">Ensemble Weights</div>', unsafe_allow_html=True)
    weights = {}
    for key in agent.available_models or []:
        weights[key] = st.slider(
            config.MODEL_REGISTRY[key]["display_name"].split(" (")[0], 0.0, 2.0, 1.0, 0.1
        )

    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.caption(
        "This agent runs a submitted article through three independently "
        "fine-tuned transformer classifiers, combines their votes into a "
        "weighted verdict, flags sensational language, and surfaces the "
        "sentences that most influenced the call."
    )

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Submit Article</div>', unsafe_allow_html=True)

tab_paste, tab_file = st.tabs(["Paste text", "Upload .txt"])

article_text = ""
with tab_paste:
    article_text = st.text_area(
        "Article title + body",
        height=220,
        placeholder="Paste the headline and full article text here...",
    )
with tab_file:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None:
        article_text = uploaded.read().decode("utf-8", errors="ignore")
        st.text_area("Preview", article_text, height=200, disabled=True)

analyze = st.button("▶ Run Verification", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Run + render results
# ---------------------------------------------------------------------------
if analyze:
    if not article_text or len(article_text.split()) < 8:
        st.error("Paste at least a couple of sentences of article text to analyze.")
    else:
        with st.spinner("Cross-checking with the desk..."):
            verdict = agent.predict(article_text, weights=weights or None)

        stamp_color = WIRE_TEAL if verdict.final_label == "REAL" else ALARM
        st.markdown('<div class="section-label">Verdict</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stamp-box" style="--stamp-color:{stamp_color}">
                {"VERIFIED" if verdict.final_label == "REAL" else "FLAGGED · LIKELY FAKE"}
            </div>
            <div class="telex">CONFIDENCE {verdict.final_confidence:.1%} &nbsp;·&nbsp;
            {verdict.models_used if verdict.models_used else 1} SOURCE MODEL(S) CONSULTED</div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.info(verdict.explanation, icon="🧾")

        # --- Per-model breakdown -------------------------------------------------
        st.markdown('<div class="section-label">Model Breakdown</div>', unsafe_allow_html=True)
        col_chart, col_rows = st.columns([1.2, 1])

        with col_chart:
            fig = go.Figure()
            names = [v.display_name.split(" (")[0] for v in verdict.per_model]
            reals = [v.prob_real * 100 for v in verdict.per_model]
            fakes = [v.prob_fake * 100 for v in verdict.per_model]
            fig.add_trace(go.Bar(y=names, x=reals, name="REAL", orientation="h", marker_color=WIRE_TEAL))
            fig.add_trace(go.Bar(y=names, x=fakes, name="FAKE", orientation="h", marker_color=ALARM))
            fig.update_layout(
                barmode="stack",
                paper_bgcolor=INK,
                plot_bgcolor=INK,
                font=dict(color=PAPER, family="IBM Plex Mono"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=140 + 40 * len(names),
                legend=dict(orientation="h", y=-0.2),
                xaxis=dict(ticksuffix="%", gridcolor="#2A3140"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_rows:
            for v in verdict.per_model:
                row_color = WIRE_TEAL if v.label == "REAL" else ALARM
                st.markdown(
                    f"""<div class="model-row" style="--row-color:{row_color}">
                    {v.display_name}<br>→ <b>{v.label}</b> ({v.confidence:.1%} confidence)
                    </div>""",
                    unsafe_allow_html=True,
                )

        # --- Explainability --------------------------------------------------
        st.markdown('<div class="section-label">Why the Desk Flagged This</div>', unsafe_allow_html=True)
        if verdict.key_sentences:
            for s in verdict.key_sentences:
                st.markdown(f'<div class="claim-card">"{s}"</div>', unsafe_allow_html=True)
        else:
            st.caption("No standout sentences identified.")

        if verdict.flagged_phrases:
            st.markdown(
                f"**Sensational language detected:** `{'`, `'.join(verdict.flagged_phrases)}`"
            )

        # --- Optional live cross-reference -----------------------------------
        if config.NEWSAPI_KEY:
            st.markdown('<div class="section-label">Live Cross-Reference</div>', unsafe_allow_html=True)
            with st.spinner("Checking recent headlines..."):
                query = " ".join(article_text.split()[:12])
                ref = agent.cross_reference(query)
            if ref.get("articles"):
                for a in ref["articles"]:
                    st.markdown(f"- [{a['title']}]({a['url']}) — *{a['source']}*")
            else:
                st.caption("No related live coverage found.")
        else:
            st.caption(
                "Tip: set the `NEWSAPI_KEY` environment variable to enable live "
                "cross-referencing against current headlines."
            )

st.markdown("---")
st.caption(
    "Educational project — verdicts are model outputs, not a substitute for "
    "professional fact-checking. Dataset: Kaggle 'Fake and Real News Dataset' "
    "(Bisaillon)."
)
