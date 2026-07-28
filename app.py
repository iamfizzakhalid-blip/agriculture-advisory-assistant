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

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Agri Assistant 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    /* ============================================= */
    /* 1. FONTS                                       */
    /* ============================================= */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    /* ============================================= */
    /* 2. FULL-SCREEN BACKGROUND                      */
    /* ============================================= */
    .stApp {{
        background: url('{BG_IMAGE_URL}') center/cover no-repeat fixed;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: rgba(15, 25, 10, 0.72);
        backdrop-filter: blur(3px);
        -webkit-backdrop-filter: blur(3px);
        z-index: 0;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    /* ============================================= */
    /* 3. GLASSMORPHISM MIXIN (reused via classes)     */
    /* ============================================= */
    .glass {{
        background: rgba(30, 60, 30, 0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(129, 199, 132, 0.18);
        border-radius: 16px;
    }}

    /* ============================================= */
    /* 4. SIDEBAR                                     */
    /* ============================================= */
    section[data-testid="stSidebar"] {{
        background: rgba(20, 40, 20, 0.85) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(129, 199, 132, 0.12);
    }}
    section[data-testid="stSidebar"] * {{
        color: #c8e6c9 !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {{
        color: #a5d6a7 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(129, 199, 132, 0.15) !important;
    }}

    /* ============================================= */
    /* 5. HERO HEADER                                 */
    /* ============================================= */
    .hero-header {{
        background: linear-gradient(135deg, rgba(27, 94, 32, 0.7) 0%, rgba(46, 125, 50, 0.6) 50%, rgba(56, 142, 60, 0.5) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(129, 199, 132, 0.2);
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
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    .hero-header p {{
        color: rgba(200, 230, 201, 0.9);
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
    }}

    /* ============================================= */
    /* 6. CROP CARDS (Sidebar)                        */
    /* ============================================= */
    .crop-card {{
        background: rgba(129, 199, 132, 0.08);
        border: 1px solid rgba(129, 199, 132, 0.18);
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
        background: rgba(129, 199, 132, 0.18);
        transform: translateX(4px);
        border-color: rgba(129, 199, 132, 0.35);
    }}
    .crop-card .crop-icon {{
        font-size: 1.3rem;
        width: 2rem;
        text-align: center;
    }}
    .crop-card .crop-name {{
        font-size: 0.88rem;
        font-weight: 500;
        color: #c8e6c9 !important;
    }}

    /* ============================================= */
    /* 7. WELCOME SECTION                             */
    /* ============================================= */
    .welcome-section {{
        background: rgba(30, 60, 30, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(129, 199, 132, 0.15);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin: 1rem 0;
        text-align: center;
    }}
    .welcome-section h3 {{
        color: #e8f5e9;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
        font-size: 1.3rem;
    }}
    .welcome-section p {{
        color: #a5d6a7;
        margin: 0 0 1.2rem 0;
        line-height: 1.6;
        font-size: 0.95rem;
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
        color: #ffe082;
        transition: all 0.25s ease;
        cursor: default;
    }}
    .welcome-section .suggestion:hover {{
        background: rgba(212, 160, 23, 0.25);
        transform: translateY(-2px);
    }}

    /* ============================================= */
    /* 8. CHAT AREA                                   */
    /* ============================================= */
    .stChatMessage {{
        background: rgba(30, 60, 30, 0.3) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(129, 199, 132, 0.1) !important;
        border-radius: 14px !important;
        margin-bottom: 0.75rem !important;
        padding: 1rem !important;
    }}

    /* ============================================= */
    /* 9. CHAT INPUT                                  */
    /* ============================================= */
    .stChatInput > div {{
        border-radius: 14px !important;
        border: 1px solid rgba(129, 199, 132, 0.25) !important;
        background: rgba(20, 40, 20, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }}
    .stChatInput > div:focus-within {{
        border-color: #81C784 !important;
        box-shadow: 0 0 0 2px rgba(129, 199, 132, 0.2) !important;
    }}
    .stChatInput textarea {{
        color: #e8f5e9 !important;
    }}
    .stChatInput textarea::placeholder {{
        color: #81C784 !important;
        opacity: 0.6 !important;
    }}

    /* ============================================= */
    /* 10. SOURCE CARDS                               */
    /* ============================================= */
    .source-card {{
        background: rgba(46, 125, 50, 0.12);
        border: 1px solid rgba(129, 199, 132, 0.18);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        transition: all 0.25s ease;
    }}
    .source-card:hover {{
        background: rgba(46, 125, 50, 0.22);
        transform: translateX(3px);
    }}
    .source-card .source-icon {{
        font-size: 1.2rem;
        width: 1.8rem;
        text-align: center;
    }}
    .source-card .source-name {{
        font-size: 0.88rem;
        color: #c8e6c9;
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
        color: #ffe082;
        margin-top: 0.5rem;
    }}

    /* ============================================= */
    /* 12. TECH BADGES (Sidebar)                      */
    /* ============================================= */
    .tech-badge {{
        display: inline-block;
        background: rgba(129, 199, 132, 0.08);
        border: 1px solid rgba(129, 199, 132, 0.18);
        border-radius: 8px;
        padding: 0.25rem 0.65rem;
        margin: 0.15rem 0;
        font-size: 0.78rem;
        transition: all 0.2s ease;
    }}
    .tech-badge:hover {{
        background: rgba(129, 199, 132, 0.18);
    }}

    /* ============================================= */
    /* 13. CLEAR CHAT BUTTON                          */
    /* ============================================= */
    section[data-testid="stSidebar"] .stButton > button {{
        background: rgba(198, 40, 40, 0.12) !important;
        color: #ef9a9a !important;
        border: 1px solid rgba(198, 40, 40, 0.25) !important;
        border-radius: 10px !important;
        width: 100%;
        transition: all 0.25s ease;
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(198, 40, 40, 0.25) !important;
        transform: translateY(-1px);
    }}

    /* ============================================= */
    /* 14. EXPANDER STYLING                           */
    /* ============================================= */
    .streamlit-expanderHeader {{
        background: rgba(30, 60, 30, 0.3) !important;
        border-radius: 10px !important;
        color: #a5d6a7 !important;
    }}

    /* ============================================= */
    /* 15. FOOTER                                     */
    /* ============================================= */
    .footer {{
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: rgba(165, 214, 167, 0.4);
        font-size: 0.8rem;
    }}
    .footer strong {{
        color: rgba(165, 214, 167, 0.6);
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
        background: rgba(129, 199, 132, 0.3);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(129, 199, 132, 0.5);
    }}

    /* ============================================= */
    /* 17. SPINNER / STATUS                           */
    /* ============================================= */
    .stSpinner > div {{
        border-top-color: #81C784 !important;
    }}

    /* ============================================= */
    /* 18. GLOBAL TRANSITIONS                         */
    /* ============================================= */
    a, button, .stButton > button {{
        transition: all 0.25s ease !important;
    }}
</style>
""", unsafe_allow_html=True)

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
    crop_html = ""
    for icon, name in crops:
        crop_html += f"""
        <div class="crop-card">
            <span class="crop-icon">{icon}</span>
            <span class="crop-name">{name}</span>
        </div>"""
    st.markdown(crop_html, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### ⚙️ Tech Stack")
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
    <div style="text-align:center; margin-top:2rem; font-size:0.72rem; opacity:0.4;">
        v1.0 • AI-Powered Advisory
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
# SESSION STATE (CHAT HISTORY)
# =============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

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

# =============================================
# FOOTER
# =============================================
st.markdown("""
<div class="footer">
    Made with ❤️ using <strong>Streamlit</strong> · <strong>ChromaDB</strong> · <strong>Sentence Transformers</strong> · <strong>Groq</strong>
</div>
""", unsafe_allow_html=True)
