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
    /* 13. CLEAR CHAT BUTTON                          */
    /* ============================================= */
    section[data-testid="stSidebar"] .stButton > button {{
        background: rgba(198, 40, 40, 0.12) !important;
        color: #EF9A9A !important;
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
