# 🌾 Agriculture Advisory Assistant

A Retrieval-Augmented Generation (RAG) pipeline that provides intelligent agricultural advisory for Pakistani crops — wheat, rice, cotton, maize, and sugarcane.

## Project Structure

```
agriculture-advisory-assistant/
├── app/                        # Streamlit web application (future)
├── data/
│   ├── raw/                    # Original PDF documents per crop
│   │   ├── wheat/
│   │   ├── rice/
│   │   ├── cotton/
│   │   ├── maize/
│   │   └── sugarcane/
│   ├── clean/                  # Extracted & cleaned .txt files
│   │   ├── wheat/
│   │   ├── rice/
│   │   ├── cotton/
│   │   ├── maize/
│   │   └── sugarcane/
│   ├── chroma_db/              # ChromaDB vector store
├── scripts/                    # Pipeline scripts
│   ├── extract_text.py         # Step 1: PDF → cleaned text
│   ├── chunk_and_store.py      # Step 2: Chunk & embed into ChromaDB
│   └── query.py                # Step 3: Query the vector store
├── tests/                      # Verification & test scripts
│   ├── test_chromadb.py
│   ├── test_groq.py
│   ├── test_sentence_transformer.py
│   ├── test_setup.py
│   └── test_streamlit.py
├── checkpoints/                # Day-wise progress checkpoints
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
└── .env.example                # an example of actual .env

```

## Setup

```bash
# 1. Create (if not already) & activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
#    Create a .env file in the project root:
echo GROQ_API_KEY=your_api_key_here > .env
```

## RAG Pipeline

### Step 1 — Extract Text from PDFs

Place your crop PDFs in the corresponding `data/raw/<crop>/` folders, then run:

```bash
python scripts/extract_text.py
```

This will:
- Traverse every crop folder in `data/raw/`
- Extract text from every PDF using PyMuPDF
- Clean the text (remove page numbers, repeated headers/footers, excess whitespace)
- Save cleaned `.txt` files to `data/clean/<crop>/`

### Step 2 — Chunk & Store Embeddings

```bash
python scripts/chunk_and_store.py
```

Splits cleaned text into chunks, generates embeddings with `all-MiniLM-L6-v2`, and stores them in ChromaDB.

### Step 3 — Query

```bash
python scripts/query.py
```

Interactive query loop that retrieves relevant context from the vector store.

## Tech Stack

| Component        | Technology              |
| ---------------- | ----------------------- |
| Embeddings       | Sentence Transformers   |
| Vector Store     | ChromaDB                |
| LLM              | Groq (Llama 3.3 70B)   |
| Text Extraction  | PyMuPDF                 |
| Frontend         | Streamlit               |

## Verification

Run the test scripts to verify your environment:

```bash
python tests/test_setup.py                # All libraries import OK
python tests/test_chromadb.py             # ChromaDB read/write
python tests/test_sentence_transformer.py # Embedding generation
python tests/test_groq.py                 # Groq API connectivity
streamlit run tests/test_streamlit.py     # Streamlit UI
```

## Future Improvements

- **Unified Pipeline App** — Build a Streamlit interface that orchestrates the full RAG flow (extract → chunk → embed → query) in one click, eliminating the need to run each script manually.
- **Multi-Crop Query Support** — Extend the query system to search across all crops simultaneously and filter results by crop type.
- **Groq-Powered Answers** — Integrate Groq LLM to generate natural-language answers from retrieved context instead of returning raw chunks.
- **Auto-Ingest on Upload** — Allow users to upload new PDFs through the UI and automatically process them through the entire pipeline.
- **Evaluation & Metrics** — Add retrieval quality metrics (precision, recall, MRR) to benchmark and improve the RAG pipeline over time.