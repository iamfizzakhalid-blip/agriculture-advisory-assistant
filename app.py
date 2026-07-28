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
# -------------------------------
st.markdown(
    """
    <style>
        :root {
            --agri-green: #2e7d32;
            --agri-green-light: #66bb6a;
            --agri-cream: #f6f8f1;
        }

        /* App background */
        .stApp {
            background: linear-gradient(180deg, #f6f8f1 0%, #ffffff 250px);
        }

        /* Hide default Streamlit chrome */
        #MainMenu, footer {visibility: hidden;}

        /* Header block */
        .agri-hero {
            padding: 1.4rem 1.8rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #2e7d32, #66bb6a);
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 18px rgba(46, 125, 50, 0.25);
        }
        .agri-hero h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
        }
        .agri-hero p {
            margin: 0.3rem 0 0 0;
            font-size: 0.95rem;
            opacity: 0.92;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f0f4ec;
            border-right: 1px solid #dde5d3;
        }
        .crop-chip {
            display: inline-block;
            padding: 5px 12px;
            margin: 3px 4px 3px 0;
            border-radius: 999px;
            background: white;
            border: 1px solid #cfe0c4;
            font-size: 0.85rem;
            color: #2e7d32;
            font-weight: 600;
        }
        .sidebar-stat {
            background: white;
            border: 1px solid #dde5d3;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        .sidebar-stat .label {
            font-size: 0.75rem;
            color: #6b7a5e;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .sidebar-stat .value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #2e7d32;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 4px 6px;
            margin-bottom: 6px;
        }

        /* Source expander */
        .source-item {
            padding: 6px 10px;
            margin: 4px 0;
            border-left: 3px solid #66bb6a;
            background: #f6f8f1;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
        }

        /* Response time badge */
        .resp-badge {
            display: inline-block;
            background: #eef5e9;
            color: #2e7d32;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 4px;
        }

        /* Welcome card */
        .welcome-card {
            background: white;
            border: 1px dashed #b8d1a8;
            border-radius: 14px;
            padding: 1.4rem;
            text-align: center;
            color: #3a4a30;
        }
        .welcome-card .emoji-row {
            font-size: 1.6rem;
            margin-bottom: 0.4rem;
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

    st.markdown("---")
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

    st.markdown("---")
    st.markdown("#### 📊 Session Stats")

    total_msgs = len(st.session_state.get("messages", []))
    questions_asked = sum(
        1 for m in st.session_state.get("messages", []) if m["role"] == "user"
    )
    last_resp_time = st.session_state.get("last_response_time")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""<div class="sidebar-stat">
                    <div class="label">Questions</div>
                    <div class="value">{questions_asked}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""<div class="sidebar-stat">
                    <div class="label">Last reply</div>
                    <div class="value">{f"{last_resp_time:.1f}s" if last_resp_time else "—"}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
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

    st.markdown("---")
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
        <p>AI-powered crop advisory system using RAG + LLM 🌱 — ask about sowing times, pest control, irrigation, fertilizers and more.</p>
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
            <span style="opacity:0.75;">Try: "When should I sow wheat in Punjab?" or "How to control pests in cotton?"</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

st.markdown("---")
st.caption(
    "Made with ❤️ using Streamlit, ChromaDB, Sentence Transformers and Groq"
)