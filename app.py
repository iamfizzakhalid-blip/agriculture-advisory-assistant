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
from scripts.chat_db import (
    create_chat as db_create_chat,
    save_chat as db_save_chat,
    load_chat as db_load_chat,
    delete_chat as db_delete_chat,
    list_chats as db_list_chats,
)

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Agri Assistant 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SQLite-backed chat storage wrappers


def save_chat(chat_id: str, title: str, messages: list):
    try:
        db_save_chat(chat_id, title, messages)
    except Exception:
        pass


def load_chat(chat_id: str):
    try:
        return db_load_chat(chat_id)
    except Exception:
        return None


def delete_chat(chat_id: str):
    try:
        db_delete_chat(chat_id)
    except Exception:
        pass


def list_chats():
    try:
        return db_list_chats()
    except Exception:
        return []


def create_new_chat(title: str = "New Chat") -> str:
    return db_create_chat(title=title)


def make_title(messages: list, chat_id: str = "") -> str:
    """Derive a short chat title from the user-provided topic or the first user message."""
    # If a topic was stored for this chat, always use it
    stored_topics = st.session_state.get("chat_topics", {})
    if chat_id and chat_id in stored_topics:
        return stored_topics[chat_id]
    for m in messages:
        if m["role"] == "user":
            text = m["content"].strip().replace("\n", " ")
            return text[:40] + ("…" if len(text) > 40 else "")
    return "New Chat"


def switch_to_chat(chat_id: str):
    data = load_chat(chat_id)
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = data["messages"] if data else []


# =============================================
# BACKGROUND IMAGE CONFIG
# Change the URL below to use a local file or
# different image. For local: url('images/wheat.jpg')
# =============================================
BG_IMAGE_URL = "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1920&q=80"

# =============================================
# CUSTOM CSS — Agriculture Glassmorphism Theme
# =============================================
st.markdown(f"""
<style>
    /* Responsive adjustments: tablet and mobile */
    @media (max-width: 900px) {{
      .hero-header {{ padding: 1.2rem 1.4rem; }}
      .hero-header h1 {{ font-size: 1.6rem; }}
      .hero-header .hero-emoji {{ font-size: 2rem; }}
      .hero-header p {{ font-size: 0.95rem; }}
      .welcome-section {{ padding: 1rem 1.2rem; }}
      .welcome-section .suggestion {{ padding: 0.4rem 0.8rem; font-size: 0.82rem; }}
      section[data-testid="stSidebar"] .stButton > button {{ font-size: 0.9rem; padding: 6px 8px !important; }}
      .crop-card .crop-name {{ font-size: 0.82rem; }}
      .source-card .source-name {{ font-size: 0.85rem; }}
      .stChatMessage {{ padding: 0.75rem !important; }}
      .response-time {{ font-size: 0.7rem; }}
    }}

    @media (max-width: 600px) {{
      .hero-header {{ padding: 0.8rem 1rem; border-radius: 12px; }}
      .hero-header h1 {{ font-size: 1.2rem; }}
      .hero-header .hero-emoji {{ font-size: 1.6rem; }}
      .hero-header p {{ font-size: 0.88rem; }}
      .welcome-section {{ padding: 0.8rem 1rem; margin: 0.6rem 0; }}
      .welcome-section .suggestion {{ padding: 0.35rem 0.6rem; font-size: 0.78rem; }}
      .crop-card {{ padding: 0.4rem 0.6rem; }}
      .crop-card .crop-name {{ font-size: 0.78rem; }}
      section[data-testid="stSidebar"] {{ padding: 8px !important; }}
      .stChatMessage {{ font-size: 0.95rem !important; padding: 0.6rem !important; margin-bottom: 0.5rem !important; }}
      .stChatInput textarea, .stChatInput input {{ font-size: 0.95rem !important; }}
      .stChatInput button {{ padding: 6px 8px !important; }}
      .source-card {{ padding: 0.5rem 0.8rem; }}
      .footer {{ display: none !important; }}
      div[style*="max-height:48vh"] {{ max-height: 60vh !important; }}
      .chat-list-row {{ margin-bottom: 3px !important; }}
      section[data-testid="stSidebar"] .stButton > button {{ min-height: 34px !important; }}
    }}
</style>
<style>
    /* ============================================= */
    /* 1. FONTS                                       */
    /* ============================================= */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    /* ============================================= */
    /* 2. FULL-SCREEN WHEAT BACKGROUND                */
    /* ============================================= */
    .stApp {{
        background: url('{BG_IMAGE_URL}') center/cover no-repeat fixed;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: rgba(22, 16, 8, 0.68);
        backdrop-filter: blur(2px);
        -webkit-backdrop-filter: blur(2px);
        z-index: 0;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    /* ============================================= */
    /* 3. KILL ALL DEFAULT STREAMLIT BLUE              */
    /* ============================================= */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .stDeployButton {{
        display: none !important;
    }}
    .st-emotion-cache-1avcm0n {{
        background: transparent !important;
    }}
    *:focus-visible {{
        outline-color: #D4A017 !important;
    }}
    .stApp a {{
        color: #E8B960 !important;
    }}
    .stApp a:hover {{
        color: #FFE082 !important;
    }}

    /* ============================================= */
    /* 4. SIDEBAR                                     */
    /* ============================================= */
    section[data-testid="stSidebar"] {{
        background: rgba(22, 16, 8, 0.93) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(212, 160, 23, 0.12);
        overflow-y: auto !important;
    }}
    section[data-testid="stSidebar"] > div {{
        overflow-y: auto !important;
        max-height: 100vh;
    }}
    section[data-testid="stSidebar"] * {{
        color: #F5ECD5 !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {{
        color: #FFE082 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(212, 160, 23, 0.15) !important;
    }}

    /* ============================================= */
    /* 5. HERO HEADER                                 */
    /* ============================================= */
    .hero-header {{
        background: linear-gradient(135deg, rgba(120, 80, 10, 0.75) 0%, rgba(180, 130, 20, 0.55) 100%);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(212, 160, 23, 0.25);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }}
    .hero-header .hero-emoji {{
        font-size: 3rem;
        display: block;
        margin-bottom: 0.5rem;
    }}
    .hero-header h1 {{
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.4);
    }}
    .hero-header p {{
        color: rgba(255, 245, 220, 0.85);
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
    }}

    /* ============================================= */
    /* 6. CROP CARDS (Sidebar)                        */
    /* ============================================= */
    .crop-card {{
        background: rgba(212, 160, 23, 0.07);
        border: 1px solid rgba(212, 160, 23, 0.18);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        margin: 0.35rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        transition: all 0.25s ease;
        cursor: default;
    }}
    .crop-card:hover {{
        background: rgba(212, 160, 23, 0.18);
        transform: translateX(4px);
        border-color: rgba(212, 160, 23, 0.4);
    }}
    .crop-card .crop-icon {{
        font-size: 1.3rem;
        width: 2rem;
        text-align: center;
    }}
    .crop-card .crop-name {{
        font-size: 0.88rem;
        font-weight: 500;
        color: #FFF8E7 !important;
    }}

    /* ============================================= */
    /* 7. WELCOME SECTION                             */
    /* ============================================= */
    .welcome-section {{
        background: rgba(40, 30, 15, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(212, 160, 23, 0.15);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin: 1rem 0;
        text-align: center;
    }}
    .welcome-section h3 {{
        color: #FFF8E7;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
        font-size: 1.3rem;
    }}
    .welcome-section p {{
        color: #F5ECD5;
        margin: 0 0 1.2rem 0;
        line-height: 1.6;
        font-size: 0.95rem;
        opacity: 0.8;
    }}
    .welcome-section .suggestions {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        justify-content: center;
    }}
    .welcome-section .suggestion {{
        background: rgba(212, 160, 23, 0.12);
        border: 1px solid rgba(212, 160, 23, 0.3);
        border-radius: 25px;
        padding: 0.5rem 1.1rem;
        font-size: 0.85rem;
        color: #FFE082;
        transition: all 0.25s ease;
        cursor: default;
    }}
    .welcome-section .suggestion:hover {{
        background: rgba(212, 160, 23, 0.25);
        transform: translateY(-2px);
    }}

    /* ============================================= */
    /* 8. CHAT MESSAGES                               */
    /* ============================================= */
    .stChatMessage {{
        background: rgba(30, 22, 12, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(212, 160, 23, 0.1) !important;
        border-radius: 14px !important;
        margin-bottom: 0.75rem !important;
        padding: 1rem !important;
    }}
    .stChatMessage p, .stChatMessage li, .stChatMessage span {{
        color: #FFF8E7 !important;
    }}

    /* ============================================= */
    /* 9. CHAT INPUT — Force override Streamlit blue  */
    /* ============================================= */
    .stChatInput,
    .stChatInput > div,
    .stChatInput > div > div,
    .stChatInput [data-testid],
    .stChatInput div[class*="emotion"],
    .stChatInput div[class*="css"] {{
        background: rgba(22, 16, 8, 0.9) !important;
        background-color: rgba(22, 16, 8, 0.9) !important;
        border-radius: 24px !important;
    }}
    .stChatInput * {{
        border-radius: 24px !important;
    }}
    .stChatInput > div {{
        border-radius: 24px !important;
        border: 1px solid rgba(212, 160, 23, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }}
    .stChatInput > div:focus-within {{
        border-color: #D4A017 !important;
        box-shadow: 0 0 0 2px rgba(212, 160, 23, 0.2) !important;
    }}
    .stChatInput textarea,
    .stChatInput input {{
        color: #FFF8E7 !important;
        background: transparent !important;
        background-color: transparent !important;
    }}
    .stChatInput textarea::placeholder,
    .stChatInput input::placeholder {{
        color: #E8B960 !important;
        opacity: 0.5 !important;
    }}
    /* Chat send button */
    .stChatInput button {{
        background: rgba(212, 160, 23, 0.2) !important;
        color: #FFE082 !important;
        border: none !important;
    }}
    .stChatInput button:hover {{
        background: rgba(212, 160, 23, 0.35) !important;
    }}

    /* ============================================= */
    /* 10. SOURCE CARDS                               */
    /* ============================================= */
    .source-card {{
        background: rgba(40, 30, 15, 0.35);
        border: 1px solid rgba(212, 160, 23, 0.15);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        transition: all 0.25s ease;
    }}
    .source-card:hover {{
        background: rgba(212, 160, 23, 0.12);
        transform: translateX(3px);
        border-color: rgba(212, 160, 23, 0.3);
    }}
    .source-card .source-icon {{
        font-size: 1.2rem;
        width: 1.8rem;
        text-align: center;
    }}
    .source-card .source-name {{
        font-size: 0.88rem;
        color: #F5ECD5;
        font-weight: 500;
    }}

    /* ============================================= */
    /* 11. RESPONSE TIME BADGE                        */
    /* ============================================= */
    .response-time {{
        display: inline-block;
        background: rgba(212, 160, 23, 0.15);
        border: 1px solid rgba(212, 160, 23, 0.3);
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        color: #FFE082;
        margin-top: 0.5rem;
    }}

    /* ============================================= */
    /* 12. TECH BADGES (Sidebar)                      */
    /* ============================================= */
    .tech-badge {{
        display: inline-block;
        background: rgba(212, 160, 23, 0.07);
        border: 1px solid rgba(212, 160, 23, 0.18);
        border-radius: 8px;
        padding: 0.25rem 0.65rem;
        margin: 0.15rem 0;
        font-size: 0.78rem;
        transition: all 0.2s ease;
    }}
    .tech-badge:hover {{
        background: rgba(212, 160, 23, 0.18);
    }}

    /* ============================================= */
    /* 13. SIDEBAR BUTTONS                            */
    /* Default = neutral gold "list item" style.      */
    /* Destructive (delete / clear) = red style.       */
    /* Active chat / New Chat = solid gold style.      */
    /* Targeting relies on Streamlit's "st-key-*"      */
    /* wrapper class that is added when a `key=` is    */
    /* passed to a widget (Streamlit >= 1.35).         */
    /* ============================================= */
    section[data-testid="stSidebar"] .stButton > button {{
        background: rgba(212, 160, 23, 0.07) !important;
        color: #FFF8E7 !important;
        border: 1px solid rgba(212, 160, 23, 0.18) !important;
        border-radius: 10px !important;
        width: 100%;
        text-align: left;
        font-weight: 500;
        transition: all 0.25s ease;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(212, 160, 23, 0.18) !important;
        border-color: rgba(212, 160, 23, 0.4) !important;
        transform: translateX(2px);
    }}

    /* New Chat button — solid gold accent */
    div[class*="st-key-new_chat_btn"] button {{
        background: rgba(212, 160, 23, 0.28) !important;
        border: 1px solid rgba(212, 160, 23, 0.55) !important;
        color: #FFE082 !important;
        font-weight: 600 !important;
    }}
    div[class*="st-key-new_chat_btn"] button:hover {{
        background: rgba(212, 160, 23, 0.4) !important;
    }}

    /* Active chat in the conversation list */
    div[class*="st-key-active_chat_"] button {{
        background: rgba(212, 160, 23, 0.22) !important;
        border-color: rgba(212, 160, 23, 0.5) !important;
        color: #FFE082 !important;
        font-weight: 600 !important;
    }}

    /* Delete + Clear buttons — red destructive style */
    div[class*="st-key-del_"] button,
    div[class*="st-key-clear_current_chat"] button {{
        background: rgba(198, 40, 40, 0.12) !important;
        color: #EF9A9A !important;
        border: 1px solid rgba(198, 40, 40, 0.25) !important;
    }}
    div[class*="st-key-del_"] button:hover,
    div[class*="st-key-clear_current_chat"] button:hover {{
        background: rgba(198, 40, 40, 0.25) !important;
        transform: translateY(-1px);
    }}

    /* ============================================= */
    /* 14. EXPANDER                                   */
    /* ============================================= */
    .streamlit-expanderHeader {{
        background: rgba(40, 30, 15, 0.3) !important;
        border-radius: 10px !important;
        color: #E8B960 !important;
    }}
    details summary {{
        color: #E8B960 !important;
    }}
    details summary:hover {{
        color: #FFE082 !important;
    }}

    /* ============================================= */
    /* 15. FOOTER                                     */
    /* ============================================= */
    .footer {{
        background: rgba(22, 16, 8, 0.4);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(212, 160, 23, 0.1);
        border-radius: 12px;
        text-align: center;
        padding: 1rem 1.5rem;
        margin-top: 2rem;
        color: rgba(245, 236, 213, 0.35);
        font-size: 0.8rem;
    }}
    .footer strong {{
        color: rgba(232, 185, 96, 0.5);
    }}

    /* ============================================= */
    /* 16. SCROLLBAR                                  */
    /* ============================================= */
    ::-webkit-scrollbar {{
        width: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: rgba(0, 0, 0, 0.1);
    }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(212, 160, 23, 0.3);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(212, 160, 23, 0.5);
    }}

    /* ============================================= */
    /* 17. SPINNER                                    */
    /* ============================================= */
    .stSpinner > div {{
        border-top-color: #D4A017 !important;
    }}

    /* ============================================= */
    /* 18. ALERTS                                     */
    /* ============================================= */
    .stAlert {{
        background: rgba(30, 22, 12, 0.5) !important;
        border-radius: 10px !important;
        color: #FFF8E7 !important;
    }}

    /* ============================================= */
    /* 19. GLOBAL TRANSITIONS                         */
    /* ============================================= */
    a, button, .stButton > button {{
        transition: all 0.25s ease !important;
    }}

    /* ============================================= */
    /* 20. BOTTOM BAR / STATUS BAR                    */
    /* ============================================= */
    .stBottom, footer, .stBottom > div {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    /* ============================================= */
    /* 21. CONVERSATION LIST LAYOUT                   */
    /* ============================================= */
</style>
<style>
    .chat-list-row div[data-testid="column"] {{
        padding: 0 2px !important;
    }}
    /* Reduce vertical padding and min-height to tighten chat rows */
    section[data-testid="stSidebar"] .stButton > button {{
        padding: 2px 6px !important;
        min-height: 28px !important;
        line-height: 1 !important;
        border: none !important;
        background: rgba(212,160,23,0.04) !important;
        box-shadow: none !important;
    }}
    div[class*="st-key-del_"] > button {{
        padding: 2px 6px !important;
        min-height: 28px !important;
    }}
    .chat-list-row {{
        margin-bottom: 4px !important;
        padding: 0 !important;
        border-bottom: none !important;
    }}

    /* ============================================= */
    /* 22. NEW TOPIC CARD                            */
    /* ============================================= */
    .new-topic-card {{
        background: rgba(40, 30, 15, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(212, 160, 23, 0.3);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 0.8rem 0;
        animation: topicCardFadeIn 0.25s ease;
    }}
    @keyframes topicCardFadeIn {{
        from {{ opacity: 0; transform: translateY(-6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .new-topic-card .topic-label {{
        color: #FFE082;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
        display: block;
    }}
    /* Confirm button — solid gold accent */
    div[class*="st-key-confirm_topic"] button {{
        background: rgba(212, 160, 23, 0.28) !important;
        border: 1px solid rgba(212, 160, 23, 0.55) !important;
        color: #FFE082 !important;
        font-weight: 600 !important;
    }}
    div[class*="st-key-confirm_topic"] button:hover {{
        background: rgba(212, 160, 23, 0.4) !important;
    }}
    /* Cancel button — subtle style */
    div[class*="st-key-cancel_topic"] button {{
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: rgba(245, 236, 213, 0.6) !important;
        font-weight: 400 !important;
    }}
    div[class*="st-key-cancel_topic"] button:hover {{
        background: rgba(255, 255, 255, 0.1) !important;
        color: #F5ECD5 !important;
    }}
    
</style>
""", unsafe_allow_html=True)

# =============================================
# SESSION STATE (CHAT HISTORY) — must run before sidebar UI
# =============================================
if "current_chat_id" not in st.session_state:
    existing = list_chats()
    if existing:
        st.session_state.current_chat_id = existing[0]["id"]
        st.session_state.messages = existing[0]["messages"]
    else:
        new_id = create_new_chat()
        st.session_state.current_chat_id = new_id
        st.session_state.messages = []

if "show_new_topic_card" not in st.session_state:
    st.session_state.show_new_topic_card = False

if "chat_topics" not in st.session_state:
    st.session_state.chat_topics = {}

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("## 🌾 Agri Assistant")
    st.caption("AI-Powered Crop Advisory")

    st.markdown("---")

    st.markdown("#### 🌿 Supported Crops")
    crops = [
        ("🌾", "Wheat"),
        ("🌽", "Maize"),
        ("🌾", "Rice"),
        ("🧵", "Cotton"),
        ("🎋", "Sugarcane"),
    ]
    # Compact, inline crop badges so conversation list is visible
    crop_html = '<div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center">'
    for icon, name in crops:
        crop_html += (
            f'<div style="padding:6px 10px; border-radius:12px; '
            f'background:rgba(212,160,23,0.04); border:1px solid rgba(212,160,23,0.12); '
            f'font-size:0.9rem;">{icon} {name}</div>'
        )
    crop_html += "</div>"
    st.markdown(crop_html, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================
    # CONVERSATIONS — new chat / switch / delete
    # =========================================
    st.markdown("#### 💬 Conversations")

    # Make the conversations area scrollable if it grows too large
    st.markdown('<div style="max-height:48vh; overflow-y:auto; padding-right:6px;">', unsafe_allow_html=True)

    if st.button("➕ New Topic", use_container_width=True, key="new_chat_btn"):
        st.session_state.show_new_topic_card = not st.session_state.show_new_topic_card

    # ----- New Topic Card -----
    if st.session_state.show_new_topic_card:
        st.markdown('<div class="new-topic-card">', unsafe_allow_html=True)
        st.markdown('<span class="topic-label">📝 Enter a topic name</span>', unsafe_allow_html=True)
        topic_name = st.text_input(
            "Topic",
            value="",
            placeholder="e.g. Wheat sowing schedule",
            label_visibility="collapsed",
            key="new_topic_input",
        )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            confirm_clicked = st.button("✅ Create", use_container_width=True, key="confirm_topic")
        with btn_col2:
            cancel_clicked = st.button("✖ Cancel", use_container_width=True, key="cancel_topic")
        st.markdown('</div>', unsafe_allow_html=True)

        if confirm_clicked:
            # Only create if the current chat isn't already a fresh empty one
            current = load_chat(st.session_state.current_chat_id)
            is_current_empty = current is not None and len(current.get("messages", [])) == 0
            title = topic_name.strip() if topic_name else "New Chat"
            if not is_current_empty:
                new_id = create_new_chat(title=title)
                # Store the user-provided topic so make_title preserves it
                if topic_name.strip():
                    st.session_state.chat_topics[new_id] = title
                switch_to_chat(new_id)
            else:
                # Rename the existing empty chat with the topic
                if topic_name.strip():
                    st.session_state.chat_topics[st.session_state.current_chat_id] = title
                    save_chat(st.session_state.current_chat_id, title, [])
            st.session_state.show_new_topic_card = False
            st.rerun()

        if cancel_clicked:
            st.session_state.show_new_topic_card = False
            st.rerun()

    chats = list_chats()

    if not chats:
        st.caption("No conversations yet.")
    else:
        for chat in chats:
            is_active = chat["id"] == st.session_state.current_chat_id
            title = chat.get("title") or "New Chat"
            label = f"{'💬 ' if is_active else '🕘 '}{title}"
            btn_key = f"active_chat_{chat['id']}" if is_active else f"switch_{chat['id']}"

            st.markdown('<div class="chat-list-row">', unsafe_allow_html=True)
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(label, key=btn_key, use_container_width=True, disabled=is_active):
                    switch_to_chat(chat["id"])
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{chat['id']}"):
                    delete_chat(chat["id"])
                    if chat["id"] == st.session_state.current_chat_id:
                        remaining = list_chats()
                        if remaining:
                            switch_to_chat(remaining[0]["id"])
                        else:
                            switch_to_chat(create_new_chat())
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # close scrollable conversations container (only once)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🗑 Clear Current Chat", use_container_width=True, key="clear_current_chat"):
        st.session_state.messages = []
        save_chat(st.session_state.current_chat_id, "New Chat", [])
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; font-size:0.72rem; opacity:0.4;">
        v1.1 • AI-Powered Advisory
    </div>
    """, unsafe_allow_html=True)

# =============================================
# HERO HEADER
# =============================================
st.markdown("""
<div class="hero-header">
    <span class="hero-emoji">🌾</span>
    <h1>Agriculture Advisory Assistant</h1>
    <p>Your AI-powered crop advisor — ask about sowing, irrigation, pest control, harvesting & more</p>
</div>
""", unsafe_allow_html=True)

# =============================================
# LOAD MODEL + DB (ONLY ONCE)
# =============================================
@st.cache_resource
def load_resources():
    model = load_embedding_model()
    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    return model, collection


model, collection = load_resources()

# =============================================
# WELCOME SECTION (only for an empty conversation)
# =============================================
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-section">
        <h3>👋 Welcome, Farmer!</h3>
        <p>I can help you with crop management, fertilizer schedules, pest control, sowing timelines, and much more. Try one of these:</p>
        <div class="suggestions">
            <span class="suggestion">🌾 When to sow wheat?</span>
            <span class="suggestion">🌽 Maize fertilizer schedule</span>
            <span class="suggestion">🧵 Cotton pest control tips</span>
            <span class="suggestion">🎋 Sugarcane irrigation guide</span>
            <span class="suggestion">🌾 Rice harvesting best practices</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# DISPLAY CHAT HISTORY
# =============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =============================================
# USER INPUT
# =============================================
user_input = st.chat_input("Ask about crops (e.g., wheat sowing time)...")

if user_input:
    # =============================================
    # SHOW USER MESSAGE
    # =============================================
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Persist immediately so the chat title/list update even if generation fails
    save_chat(
        st.session_state.current_chat_id,
        make_title(st.session_state.messages, chat_id=st.session_state.current_chat_id),
        st.session_state.messages,
    )

    # =============================================
    # GENERATE RESPONSE
    # =============================================
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

    # =============================================
    # SHOW BOT RESPONSE + SOURCES
    # =============================================
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
                        source_icon = "🌾"
                        source_name = "Wheat Guide (PMD)"
                    elif "maize" in filename:
                        source_icon = "🌽"
                        source_name = "Maize Post Harvest Guide"
                    elif "rice" in filename:
                        source_icon = "🌾"
                        source_name = "Rice Cultivation Guide"
                    elif "cotton" in filename:
                        source_icon = "🧵"
                        source_name = "Cotton Production Manual"
                    elif "sugarcane" in filename:
                        source_icon = "🎋"
                        source_name = "Sugarcane Farming Guide"
                    else:
                        source_icon = "📄"
                        source_name = "Agriculture Source"

                    if source_name not in seen:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-icon">{source_icon}</span>'
                            f'<span class="source-name">{source_name}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        seen.add(source_name)

    # =============================================
    # SAVE BOT RESPONSE
    # =============================================
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    save_chat(
        st.session_state.current_chat_id,
        make_title(st.session_state.messages, chat_id=st.session_state.current_chat_id),
        st.session_state.messages,
    )

# =============================================
# FOOTER
# =============================================
st.markdown("""
<div class="footer">
    Made with ❤️ using <strong>Streamlit</strong> · <strong>ChromaDB</strong> · <strong>Sentence Transformers</strong> · <strong>Groq</strong>
</div>
""", unsafe_allow_html=True)