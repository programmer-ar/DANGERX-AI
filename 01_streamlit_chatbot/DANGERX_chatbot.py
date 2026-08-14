# Import the libraries we need
import os
import time
import random
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

# Load our environment variables from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

AI_MODELS = {
    "gpt-oss-120b": {
        "name": "OpenAI GPT OSS 120B",
        "description": "Production-grade general model, 120B params, ~3000 tok/s",
        "cost": "Medium"
    },
    "gemma-4-31b": {
        "name": "Gemma 4 31B",
        "description": "Fast multimodal model for everyday chat and agentic use (preview)",
        "cost": "Low"
    },
    "zai-glm-4.7": {
        "name": "Z.ai GLM 4.7",
        "description": "Largest option at 355B params for complex reasoning (preview)",
        "cost": "High"
    }
}

THINKING_MESSAGES = [
    "Parsing your request...",
    "Consulting the neural net...",
    "Assembling a sharp answer...",
    "Crunching tokens...",
    "Almost there...",
]

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DANGERX AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemma-4-31b"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are DANGERX AI, a helpful, sharp, and friendly AI assistant."}
    ]

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
def get_theme_css(dark_mode: bool = True) -> str:
    if dark_mode:
        bg = "radial-gradient(circle at 20% -10%, #10224f 0%, #060b1a 45%, #020408 100%)"
        panel_bg = "rgba(59, 130, 246, 0.06)"
        panel_border = "rgba(59, 130, 246, 0.28)"
        text_color = "#eaf1ff"
        muted = "#8fa2c9"
        user_bubble = "linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)"
        ai_bubble = "rgba(255, 255, 255, 0.05)"
        ai_border = "rgba(59, 130, 246, 0.35)"
        accent = "#3b82f6"
        accent2 = "#22d3ee"
        input_bg = "rgba(255, 255, 255, 0.05)"
    else:
        bg = "radial-gradient(circle at 20% -10%, #dbeafe 0%, #eff6ff 45%, #ffffff 100%)"
        panel_bg = "rgba(255, 255, 255, 0.7)"
        panel_border = "rgba(37, 99, 235, 0.2)"
        text_color = "#0b1734"
        muted = "#5c6c8c"
        user_bubble = "linear-gradient(135deg, #2563eb 0%, #0891b2 100%)"
        ai_bubble = "rgba(255, 255, 255, 0.9)"
        ai_border = "rgba(37, 99, 235, 0.2)"
        accent = "#2563eb"
        accent2 = "#0891b2"
        input_bg = "rgba(255, 255, 255, 0.9)"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }}

    /* Restore Streamlit's icon font so icons (sidebar arrow, etc.) render as
       glyphs instead of their raw ligature name like "keyboard_double_arrow_left" */
    [data-testid="stIconMaterial"],
    span[class*="material-symbols"],
    .material-icons,
    .material-symbols-outlined {{
        font-family: 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Icons' !important;
    }}

    .stApp {{
        background: {bg};
        color: {text_color};
        overflow-x: hidden;
    }}

    /* ================= Animated 3D theme-cycling background ================= */
    .dx-scene {{
        position: fixed;
        inset: 0;
        z-index: 0;
        overflow: hidden;
        perspective: 1200px;
        pointer-events: none;
    }}

    /* rotating conic gradient sweeping through a blue palette gallery */
    .dx-gallery {{
        position: absolute;
        inset: -20%;
        background: conic-gradient(
            from 0deg,
            #1d4ed8, #2563eb, #0ea5e9, #06b6d4,
            #6366f1, #3b82f6, #0284c7, #1d4ed8
        );
        opacity: {"0.18" if dark_mode else "0.10"};
        filter: blur(90px) saturate(140%);
        animation: dx-spin 40s linear infinite, dx-hue 24s linear infinite;
        transform-style: preserve-3d;
    }}

    @keyframes dx-spin {{
        from {{ transform: rotate(0deg) scale(1.3); }}
        to   {{ transform: rotate(360deg) scale(1.3); }}
    }}
    @keyframes dx-hue {{
        0%   {{ filter: blur(90px) saturate(140%) hue-rotate(0deg); }}
        100% {{ filter: blur(90px) saturate(140%) hue-rotate(360deg); }}
    }}

    /* floating 3D glass orbs, each drifting on its own axis */
    .dx-orb {{
        position: absolute;
        border-radius: 50%;
        filter: blur(2px);
        transform-style: preserve-3d;
        will-change: transform;
    }}
    .dx-orb.o1 {{
        width: 340px; height: 340px; top: 8%; left: 6%;
        background: radial-gradient(circle at 35% 30%, {accent} 0%, transparent 70%);
        opacity: 0.5;
        animation: dx-float-a 16s ease-in-out infinite;
    }}
    .dx-orb.o2 {{
        width: 260px; height: 260px; top: 55%; left: 78%;
        background: radial-gradient(circle at 40% 35%, {accent2} 0%, transparent 70%);
        opacity: 0.45;
        animation: dx-float-b 20s ease-in-out infinite;
    }}
    .dx-orb.o3 {{
        width: 200px; height: 200px; top: 75%; left: 20%;
        background: radial-gradient(circle at 40% 35%, #0ea5e9 0%, transparent 70%);
        opacity: 0.35;
        animation: dx-float-c 24s ease-in-out infinite;
    }}
    .dx-orb.o4 {{
        width: 180px; height: 180px; top: 15%; left: 65%;
        background: radial-gradient(circle at 40% 35%, #6366f1 0%, transparent 70%);
        opacity: 0.3;
        animation: dx-float-a 22s ease-in-out infinite reverse;
    }}

    @keyframes dx-float-a {{
        0%, 100% {{ transform: translate3d(0, 0, 0) rotateX(0deg) rotateY(0deg); }}
        25%      {{ transform: translate3d(40px, -60px, 80px) rotateX(25deg) rotateY(15deg); }}
        50%      {{ transform: translate3d(-30px, 30px, -40px) rotateX(-15deg) rotateY(30deg); }}
        75%      {{ transform: translate3d(60px, 50px, 40px) rotateX(10deg) rotateY(-20deg); }}
    }}
    @keyframes dx-float-b {{
        0%, 100% {{ transform: translate3d(0, 0, 0) rotateZ(0deg); }}
        33%      {{ transform: translate3d(-50px, -40px, 60px) rotateZ(20deg); }}
        66%      {{ transform: translate3d(30px, 60px, -30px) rotateZ(-15deg); }}
    }}
    @keyframes dx-float-c {{
        0%, 100% {{ transform: translate3d(0, 0, 0) scale(1); }}
        50%      {{ transform: translate3d(-40px, -50px, 50px) scale(1.15); }}
    }}

    /* subtle 3D perspective grid floor for depth */
    .dx-grid {{
        position: absolute;
        left: -50%; right: -50%; bottom: -10%;
        height: 50%;
        background-image:
            linear-gradient({panel_border} 1px, transparent 1px),
            linear-gradient(90deg, {panel_border} 1px, transparent 1px);
        background-size: 60px 60px;
        opacity: {"0.2" if dark_mode else "0.12"};
        transform: rotateX(75deg);
        transform-origin: bottom;
        animation: dx-grid-move 8s linear infinite;
    }}
    @keyframes dx-grid-move {{
        from {{ background-position: 0 0, 0 0; }}
        to   {{ background-position: 0 60px, 0 0; }}
    }}

    /* keep actual app content above the scene */
    section[data-testid="stSidebar"], .main .block-container {{
        position: relative;
        z-index: 1;
    }}

    /* Hide default streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background: transparent !important;}}

    /* Title */
    .dx-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 8px 0 4px 0;
    }}
    .dx-logo {{
        font-size: 2.1rem;
        filter: drop-shadow(0 0 12px {accent});
        animation: pulse-glow 2.4s ease-in-out infinite;
    }}
    .dx-title {{
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(90deg, {accent} 0%, {accent2} 55%, {accent} 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shine 5s linear infinite;
    }}
    .dx-subtitle {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 300;
        font-size: 0.95rem;
        color: {muted};
        letter-spacing: 0.5px;
        margin-top: -6px;
    }}

    @keyframes shine {{
        to {{ background-position: 200% center; }}
    }}
    @keyframes pulse-glow {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.12); opacity: 0.75; }}
    }}

    /* Chat bubbles */
    .dx-chat-wrap {{
        display: flex;
        flex-direction: column;
        gap: 14px;
        padding: 8px 2px 18px 2px;
    }}
    .dx-row {{
        display: flex;
        width: 100%;
        animation: fadeInUp 0.35s ease both;
    }}
    .dx-row.user {{ justify-content: flex-end; }}
    .dx-row.ai {{ justify-content: flex-start; }}

    .dx-bubble {{
        max-width: 72%;
        padding: 13px 18px;
        border-radius: 18px;
        line-height: 1.5;
        font-size: 0.98rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
    }}
    .dx-bubble.user {{
        background: {user_bubble};
        color: white;
        border-bottom-right-radius: 4px;
    }}
    .dx-bubble.ai {{
        background: {ai_bubble};
        border: 1px solid {ai_border};
        color: {text_color};
        border-bottom-left-radius: 4px;
        backdrop-filter: blur(6px);
    }}
    .dx-label {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.6px;
        opacity: 0.6;
        margin-bottom: 4px;
        text-transform: uppercase;
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Thinking / typing indicator */
    .dx-thinking {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 18px;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        background: {ai_bubble};
        border: 1px solid {ai_border};
        width: fit-content;
        font-size: 0.9rem;
        color: {muted};
    }}
    .dx-dots {{
        display: flex;
        gap: 4px;
    }}
    .dx-dots span {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {accent};
        animation: bounce 1.2s infinite ease-in-out;
    }}
    .dx-dots span:nth-child(2) {{ animation-delay: 0.15s; }}
    .dx-dots span:nth-child(3) {{ animation-delay: 0.3s; }}

    @keyframes bounce {{
        0%, 80%, 100% {{ transform: scale(0.6); opacity: 0.4; }}
        40% {{ transform: scale(1); opacity: 1; }}
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {panel_bg};
        border-right: 1px solid {panel_border};
        backdrop-filter: blur(10px);
    }}
    section[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not(i) {{
        color: {text_color} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}

    .dx-model-card {{
        background: {panel_bg};
        border: 1px solid {panel_border};
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.82rem;
        color: {muted} !important;
    }}

    /* Input area */
    div[data-testid="stForm"] {{
        background: {input_bg};
        border: 1px solid {panel_border};
        border-radius: 16px;
        padding: 10px 14px;
        backdrop-filter: blur(8px);
    }}

    .stTextInput input {{
        background: transparent !important;
        color: {text_color} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}

    .stButton button, .stFormSubmitButton button {{
        background: linear-gradient(135deg, {accent} 0%, {accent2} 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }}
    .stButton button:hover, .stFormSubmitButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
    }}

    hr {{
        border-color: {panel_border} !important;
    }}
    </style>
    """


st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Animated 3D background scene (gallery of blues sweeping behind the UI)
st.markdown(
    """
    <div class="dx-scene">
        <div class="dx-gallery"></div>
        <div class="dx-grid"></div>
        <div class="dx-orb o1"></div>
        <div class="dx-orb o2"></div>
        <div class="dx-orb o3"></div>
        <div class="dx-orb o4"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ DANGERX AI")
    st.caption("Control panel")

    st.markdown("---")
    st.markdown("**🎨 Appearance**")
    dark_mode_toggle = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        st.rerun()

    st.markdown("---")
    st.markdown("**🧠 AI Model**")

    model_options = list(AI_MODELS.keys())
    model_labels = [f"{AI_MODELS[m]['name']}" for m in model_options]

    selected_index = st.selectbox(
        "Choose a model",
        range(len(model_options)),
        format_func=lambda x: model_labels[x],
        index=model_options.index(st.session_state.selected_model),
        label_visibility="collapsed",
    )
    st.session_state.selected_model = model_options[selected_index]

    m = AI_MODELS[st.session_state.selected_model]
    st.markdown(
        f"""<div class="dx-model-card">
        <b>{m['name']}</b><br>
        {m['description']}<br>
        <span style="opacity:0.7;">Cost tier: {m['cost']}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**🛠️ Tools**")

    # --- Quick prompts -----------------------------------------------------
    st.caption("Quick prompts")
    quick_prompts = {
        "💡 Explain": "Explain this in simple terms: ",
        "✍️ Rewrite": "Rewrite this to sound more professional: ",
        "🐛 Debug": "Help me debug this code: ",
        "📋 Summarize": "Summarize the following: ",
    }
    qp_cols = st.columns(2)
    for i, (label, prefix) in enumerate(quick_prompts.items()):
        with qp_cols[i % 2]:
            if st.button(label, use_container_width=True, key=f"qp_{i}"):
                st.session_state.prefill = prefix
                st.rerun()

    st.markdown("")

    # --- Conversation stats --------------------------------------------------
    st.caption("Conversation stats")
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    ai_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    total_words = sum(len(m["content"].split()) for m in user_msgs + ai_msgs)
    s1, s2, s3 = st.columns(3)
    s1.metric("You", len(user_msgs))
    s2.metric("AI", len(ai_msgs))
    s3.metric("Words", total_words)

    st.markdown("")

    # --- Regenerate last response --------------------------------------------
    if ai_msgs and st.button("🔁 Regenerate last response", use_container_width=True):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
        st.session_state.regenerate = True
        st.rerun()

    # --- Export conversation ---------------------------------------------
    if len(st.session_state.messages) > 1:
        transcript_lines = []
        for m in st.session_state.messages:
            if m["role"] == "system":
                continue
            who = "You" if m["role"] == "user" else "DANGERX AI"
            transcript_lines.append(f"{who}: {m['content']}")
        transcript = "\n\n".join(transcript_lines)
        st.download_button(
            "⬇️ Export chat (.txt)",
            data=transcript,
            file_name=f"dangerx_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": "You are DANGERX AI, a helpful, sharp, and friendly AI assistant."}
        ]
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dx-header">
        <div class="dx-logo">⚡</div>
        <div>
            <div class="dx-title">DANGERX AI</div>
            <div class="dx-subtitle">Fast. Sharp. Always on.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API key / base URL checks
# ---------------------------------------------------------------------------
if not api_key:
    st.error("⚠️ Please add your API key to the .env file!")
    st.stop()
if not base_url:
    st.error("⚠️ Please add your BASE_URL to the .env file!")
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Helper: get AI response
# ---------------------------------------------------------------------------
def get_ai_response():
    response = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=st.session_state.messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Conversation display
# ---------------------------------------------------------------------------
chat_container = st.container()

def render_messages():
    with chat_container:
        st.markdown('<div class="dx-chat-wrap">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            role = "user" if message["role"] == "user" else "ai"
            label = "You" if role == "user" else "DANGERX AI"
            content = message["content"].replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="dx-row {role}">
                    <div class="dx-bubble {role}">
                        <div class="dx-label">{label}</div>
                        {content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

render_messages()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
prefill_value = st.session_state.pop("prefill", "")

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "You:",
            value=prefill_value,
            placeholder="Ask DANGERX AI anything...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ⚡", use_container_width=True)

# ---------------------------------------------------------------------------
# Handle new message with a smooth "thinking" effect
# ---------------------------------------------------------------------------
trigger_response = False

if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    trigger_response = True

if st.session_state.pop("regenerate", False):
    trigger_response = True

if trigger_response:
    # Re-render so the latest user message appears immediately
    chat_container.empty()
    render_messages()

    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        f"""
        <div class="dx-row ai">
            <div class="dx-thinking">
                <span id="dx-think-text">{random.choice(THINKING_MESSAGES)}</span>
                <div class="dx-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # cycle a couple of thinking messages for a smoother perceived-loading effect
    for msg in random.sample(THINKING_MESSAGES, k=2):
        thinking_placeholder.markdown(
            f"""
            <div class="dx-row ai">
                <div class="dx-thinking">
                    <span>{msg}</span>
                    <div class="dx-dots"><span></span><span></span><span></span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.35)

    try:
        ai_response = get_ai_response()
    except Exception as e:
        ai_response = f"⚠️ Something went wrong talking to the model: {e}"

    thinking_placeholder.empty()

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.rerun()