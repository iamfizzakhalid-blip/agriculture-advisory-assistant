import sys
import os
sys.path.append(os.path.abspath("."))
import streamlit as st
import time
from datetime import datetime

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
# GLOBAL STYLES
# Palette pulled straight from .streamlit/config.toml so custom
# markup and native Streamlit widgets always match:
#   background        #fbfbf8
#   secondaryBg        #f0f3ec
#   primary (green)    #3f7d3a
#   text               #26311f
# -------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary: #6fcf6a;
            --primary-dark: #8fdb8b;
            --text: #e8ede4;
            --text-muted: #9fab97;
            --bg: #0e1210;
            --bg-secondary: #1a2119;
            --border: #2e3a2b;
        }

        #MainMenu, footer {visibility: hidden;}

        /* ---- Hero header ---- */
        .agri-hero {
            padding: 1.3rem 1.6rem;
            border-radius: 14px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            margin-bottom: 1.2rem;
        }
        .agri-hero h1 {
            margin: 0;
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--primary-dark);
        }
        .agri-hero p {
            margin: 0.35rem 0 0 0;
            font-size: 0.92rem;
            color: var(--text-muted);
        }

        /* ---- Crop chips ---- */
        .crop-chip {
            display: inline-block;
            padding: 4px 12px;
            margin: 3px 6px 3px 0;
            border-radius: 999px;
            background: var(--bg);
            border: 1px solid var(--border);
            font-size: 0.83rem;
            color: var(--primary-dark);
            font-weight: 600;
        }

        /* ---- Source cards ---- */
        .source-item {
            padding: 6px 10px;
            margin: 4px 0;
            border-left: 3px solid var(--primary);
            background: var(--bg-secondary);
            border-radius: 0 8px 8px 0;
            font-size: 0.88rem;
            color: var(--text);
        }

        /* ---- Response time badge ---- */
        .resp-badge {
            display: inline-block;
            background: var(--bg-secondary);
            color: var(--primary-dark);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-top: 6px;
        }

        /* ---- Welcome card ---- */
        .welcome-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.4rem;
            text-align: center;
            color: var(--text);
        }
        .welcome-card .emoji-row {
            font-size: 1.5rem;
            margin-bottom: 0.4rem;
        }
        .welcome-card span.hint {
            color: var(--text-muted);
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.markdown("## 🌾 Agri Assistant")
    st.caption("Your AI partner for smarter farming decisions")

    st.divider()
    st.markdown("#### 🌱 Supported Crops")
    st.markdown(
        """
        <div>
            <span class="crop-chip">🌾 Wheat</span>
            <span class="crop-chip">🌽 Maize</span>
            <span class="crop-chip">🌾 Rice</span>
            <span class="crop-chip">🧵 Cotton</span>
            <span class="crop-chip">🎋 Sugarcane</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### ℹ️ About")
    st.write(
        """
This application uses:

- 🗂️ ChromaDB for vector search
- 🧠 Sentence Transformers for embeddings
- 🔎 Retrieval-Augmented Generation (RAG)
- ⚡ Groq LLM for fast responses

Developed as an AI-powered Agriculture Advisory System.
"""
    )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response_time = None
        st.rerun()

    st.caption(f"🕒 {datetime.now().strftime('%A, %d %b %Y')}")

# -------------------------------
# HERO / HEADER
# -------------------------------
st.markdown(
    """
    <div class="agri-hero">
        <h1>🌾 Agriculture Advisory Assistant</h1>
        <p>AI-powered crop advisory system using RAG + LLM — ask about sowing times, pest control, irrigation, fertilizers and more.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# LOAD MODEL + DB (ONLY ONCE)
# -------------------------------
@st.cache_resource
def load_resources():
    model = load_embedding_model()
    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    return model, collection


with st.spinner("Loading knowledge base... 📚"):
    model, collection = load_resources()

# -------------------------------
# SESSION STATE (CHAT HISTORY)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="emoji-row">🌾 🌽 🌾 🧵 🎋</div>
            <b>Welcome!</b> Ask me anything about Wheat, Rice, Cotton, Sugarcane or Maize.<br>
            <span class="hint">Try: "When should I sow wheat in Punjab?" or "How to control pests in cotton?"</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
for msg in st.session_state.messages:
    avatar = "🧑‍🌾" if msg["role"] == "user" else "🌾"
    with st.chat_message(msg["role"], avatar=avatar):
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

    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_input)

    # -------------------------------
    # GENERATE RESPONSE
    # -------------------------------
    with st.chat_message("assistant", avatar="🌾"):
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
            st.session_state.last_response_time = response_time

            bot_reply = answer

        # -------------------------------
        # SHOW BOT RESPONSE + SOURCES
        # -------------------------------
        st.markdown(bot_reply)
        st.markdown(
            f'<span class="resp-badge">⏱ {response_time:.2f}s</span>',
            unsafe_allow_html=True,
        )

        # 🔍 Retrieved Sources (clean + no duplicates)
        if results:
            seen = set()
            source_lines = []

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
                    source_lines.append(source_name)
                    seen.add(source_name)

            with st.expander(f"🔍 Retrieved Sources ({len(source_lines)})"):
                for line in source_lines:
                    st.markdown(
                        f'<div class="source-item">{line}</div>',
                        unsafe_allow_html=True,
                    )

    # -------------------------------
    # SAVE BOT RESPONSE
    # -------------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

st.divider()
st.caption(
    "Made with ❤️ using Streamlit, ChromaDB, Sentence Transformers and Groq"
)