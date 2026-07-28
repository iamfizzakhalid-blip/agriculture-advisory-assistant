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
st.set_page_config(page_title="Agri Assistant 🌾", layout="wide")

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("🌾 Agri Assistant")

    st.markdown("---")

    st.subheader("Supported Crops")

    st.write("🌾 Wheat")
    st.write("🌽 Maize")
    st.write("🌾 Rice")
    st.write("🧵 Cotton")
    st.write("🎋 Sugarcane")

    st.markdown("---")

    st.subheader("About")

    st.write("""
This application uses:

- ChromaDB
- Sentence Transformers
- Retrieval-Augmented Generation (RAG)
- Groq LLM

Developed as an AI-powered Agriculture Advisory System.
""")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🌾 Agriculture Advisory Assistant")

st.markdown("""
Ask questions about **Wheat, Rice, Cotton, Maize, and Sugarcane** using an AI assistant powered by Retrieval-Augmented Generation (RAG).

---
""")

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
    st.info(
        "👋 Welcome! Ask me anything about Wheat, Rice, Cotton, Sugarcane or Maize."
    )

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
        st.caption(f"⏱ Response time: {response_time:.2f} seconds")

        # 🔍 Retrieved Sources (clean + no duplicates)
        if results:
            seen = set()

            with st.expander("🔍 Retrieved Sources"):
                for item in results:
                    filename = item['metadata'].get('filename', '').lower()

                    if "wheat" in filename:
                        source_name = "Wheat Guide (PMD)"
                    elif "maize" in filename:
                        source_name = "Maize Post Harvest Guide"
                    elif "rice" in filename:
                        source_name = "Rice Cultivation Guide"
                    elif "cotton" in filename:
                        source_name = "Cotton Production Manual"
                    elif "sugarcane" in filename:
                        source_name = "Sugarcane Farming Guide"
                    else:
                        source_name = "Agriculture Source"

                    if source_name not in seen:
                        st.write(f"- {source_name}")
                        seen.add(source_name)

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
