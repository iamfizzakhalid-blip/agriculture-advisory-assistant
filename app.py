import sys
import os
sys.path.append(os.path.abspath("."))
import streamlit as st
import time

from scripts.load_db import (
    get_chroma_client,
    get_or_create_collection,
    load_embedding_model,
    CHROMA_DIR,
)

from scripts.rag_pipeline import rag_pipeline

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Agri Assistant 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Main area ---------- */
    .stApp {
        background: linear-gradient(135deg, #faf8f4 0%, #f5f0e8 50%, #faf7f2 100%);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #252535 50%, #1e1e2e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #d4cfc6 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e8c87a !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(212, 160, 67, 0.2) !important;
    }

    /* ---------- Header ---------- */
    .main-header {
        background: linear-gradient(135deg, #7a5a2e 0%, #c48a2c 50%, #d4a043 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(140, 100, 30, 0.25);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.88);
        font-size: 1rem;
        margin: 0;
    }

    /* ---------- Crop chips in sidebar ---------- */
    .crop-chip {
        display: inline-block;
        background: rgba(212, 160, 67, 0.1);
        border: 1px solid rgba(212, 160, 67, 0.25);
        border-radius: 20px;
        padding: 0.35rem 0.85rem;
        margin: 0.2rem 0.15rem;
        font-size: 0.85rem;
        transition: background 0.2s;
    }
    .crop-chip:hover {
        background: rgba(212, 160, 67, 0.22);
    }

    /* ---------- Welcome card ---------- */
    .welcome-card {
        background: #ffffff;
        border: 1px solid rgba(196, 138, 44, 0.15);
        border-left: 4px solid #c48a2c;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .welcome-card h3 {
        color: #7a5a2e;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    .welcome-card p {
        color: #555;
        margin: 0;
        line-height: 1.6;
    }
    .welcome-card .suggestions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    .welcome-card .suggestion {
        background: #fdf6eb;
        border: 1px solid #e8d5b0;
        border-radius: 8px;
        padding: 0.4rem 0.9rem;
        font-size: 0.85rem;
        color: #7a5a2e;
    }

    /* ---------- Chat bubbles ---------- */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
    }

    /* ---------- Chat input ---------- */
    .stChatInput > div {
        border-radius: 12px !important;
        border: 1px solid rgba(196, 138, 44, 0.25) !important;
    }
    .stChatInput > div:focus-within {
        border-color: #c48a2c !important;
        box-shadow: 0 0 0 2px rgba(196, 138, 44, 0.15) !important;
    }

    /* ---------- Source expander ---------- */
    .source-item {
        background: #fdf8f0;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid #c48a2c;
        font-size: 0.88rem;
        color: #5c4220;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #999;
        font-size: 0.82rem;
    }
    .footer strong {
        color: #8a6a30;
    }

    /* ---------- Clear-chat button ---------- */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(200, 85, 85, 0.12) !important;
        color: #e8a0a0 !important;
        border: 1px solid rgba(200, 85, 85, 0.28) !important;
        border-radius: 8px !important;
        width: 100%;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(200, 85, 85, 0.25) !important;
    }

    /* ---------- About tech stack badges ---------- */
    .tech-badge {
        display: inline-block;
        background: rgba(212, 160, 67, 0.1);
        border: 1px solid rgba(212, 160, 67, 0.2);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        margin: 0.15rem 0;
        font-size: 0.78rem;
    }

    /* ---------- Response time badge ---------- */
    .response-time {
        display: inline-block;
        background: #fdf6eb;
        border: 1px solid #e8d5b0;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        color: #8a6a30;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.markdown("## 🌾 Agri Assistant")
    st.caption("Your AI Crop Advisor")

    st.markdown("---")

    st.markdown("#### Supported Crops")
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 0.3rem;">
        <span class="crop-chip">🌾 Wheat</span>
        <span class="crop-chip">🌽 Maize</span>
        <span class="crop-chip">🌾 Rice</span>
        <span class="crop-chip">🧵 Cotton</span>
        <span class="crop-chip">🎋 Sugarcane</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### Tech Stack")
    st.markdown("""
    <div>
        <div class="tech-badge">🗄 ChromaDB</div>
        <div class="tech-badge">🤖 Sentence Transformers</div>
        <div class="tech-badge">🔗 RAG Pipeline</div>
        <div class="tech-badge">⚡ Groq LLM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; font-size:0.75rem; opacity:0.5;">
        v1.0 • AI-Powered Advisory
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# MAIN HEADER
# -------------------------------
st.markdown("""
<div class="main-header">
    <h1>🌾 Agriculture Advisory Assistant</h1>
    <p>AI-powered crop advisory system using RAG + LLM — ask anything about Wheat, Rice, Cotton, Sugarcane, or Maize.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL + DB (ONLY ONCE)
# -------------------------------
@st.cache_resource
def load_resources():
    model = load_embedding_model()
    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    return model, collection


model, collection = load_resources()

# -------------------------------
# SESSION STATE (CHAT HISTORY)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Welcome!</h3>
        <p>I'm your Agriculture Advisory Assistant. Ask me anything about crop management, sowing, harvesting, pest control, and more.</p>
        <div class="suggestions">
            <span class="suggestion">🌾 When to sow wheat?</span>
            <span class="suggestion">🌽 Maize fertilizer schedule</span>
            <span class="suggestion">🧵 Cotton pest control</span>
            <span class="suggestion">🎋 Sugarcane irrigation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# USER INPUT
# -------------------------------
user_input = st.chat_input("Ask about crops (e.g., wheat sowing time)...")

if user_input:
    # -------------------------------
    # SHOW USER MESSAGE
    # -------------------------------
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------
    # GENERATE RESPONSE
    # -------------------------------
    with st.spinner("Thinking... 🌱"):
        start = time.time()

        try:
            results, answer = rag_pipeline(
                question=user_input,
                collection=collection,
                model=model,
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

        end = time.time()
        response_time = end - start

        if results is None:
            bot_reply = answer
        else:
            bot_reply = answer

    # -------------------------------
    # SHOW BOT RESPONSE + SOURCES
    # -------------------------------
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        st.markdown(
            f'<span class="response-time">⏱ {response_time:.2f}s</span>',
            unsafe_allow_html=True,
        )

        # 🔍 Retrieved Sources (clean + no duplicates)
        if results:
            seen = set()

            with st.expander("🔍 Retrieved Sources"):
                for item in results:
                    filename = item['metadata'].get('filename', '').lower()

                    if "wheat" in filename:
                        source_name = "🌾 Wheat Guide (PMD)"
                    elif "maize" in filename:
                        source_name = "🌽 Maize Post Harvest Guide"
                    elif "rice" in filename:
                        source_name = "🌾 Rice Cultivation Guide"
                    elif "cotton" in filename:
                        source_name = "🧵 Cotton Production Manual"
                    elif "sugarcane" in filename:
                        source_name = "🎋 Sugarcane Farming Guide"
                    else:
                        source_name = "📄 Agriculture Source"

                    if source_name not in seen:
                        st.markdown(
                            f'<div class="source-item">{source_name}</div>',
                            unsafe_allow_html=True,
                        )
                        seen.add(source_name)

    # -------------------------------
    # SAVE BOT RESPONSE
    # -------------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
    Made with ❤️ using <strong>Streamlit</strong> · <strong>ChromaDB</strong> · <strong>Sentence Transformers</strong> · <strong>Groq</strong>
</div>
""", unsafe_allow_html=True)
