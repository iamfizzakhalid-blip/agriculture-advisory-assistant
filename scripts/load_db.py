"""
embeddings.json
      │
      ▼
Load embeddings
      │
      ▼
Prepare metadata
      │
      ▼
Insert into ChromaDB
      │
      ▼
User asks question
      │
      ▼
Convert question → embedding
      │
      ▼
Compare against all stored embeddings
      │
      ▼
Return Top-K similar chunks
"""

from __future__ import annotations

import json
import re # Used to remove '_chunk_001, _chunk_023,etc.' from filenames.
from pathlib import Path # Creates platform-independent paths.

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

# --- Configuration ---

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "agricultural_knowledge"
BATCH_SIZE = 100
TOP_K = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

CHUNK_SUFFIX_PATTERN = re.compile(r"_chunk_\d+$", re.IGNORECASE)


def load_embeddings_file(embeddings_path: Path) -> list[dict]:
    """Load embedding records from the JSON file produced by embed.py."""
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    try:
        with embeddings_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in embeddings file: {embeddings_path}") from exc

    if not isinstance(records, list):
        raise ValueError("Embeddings file must contain a JSON list of records.")

    return records


def parse_chunk_metadata(chunk_id: str) -> dict[str, str]:
    """
    Extract crop, filename, and source document from a chunk path/id string.

    Example:
        ../data/chunks/cotton/FAO_cotton_chunk_001.txt
        -> crop=cotton, filename=FAO_cotton_chunk_001.txt, source=FAO_cotton
    """
    normalized = chunk_id.replace("\\", "/")
    path = Path(normalized)
    filename = path.name

    crop = "unknown"
    parts = [part for part in path.parts if part not in (".", "..")]
    if "chunks" in parts:
        chunk_index = parts.index("chunks")
        if chunk_index + 1 < len(parts):
            crop = parts[chunk_index + 1]

    stem = path.stem
    source = CHUNK_SUFFIX_PATTERN.sub("", stem) or stem

    return {
        "crop": crop,
        "filename": filename,
        "source": source,
    }


def build_document_id(metadata: dict[str, str]) -> str:
    """Create a stable, unique document ID for ChromaDB."""
    return f"{metadata['crop']}/{metadata['filename']}"


def prepare_records(raw_records: list[dict]) -> list[dict]:
    """
    Validate and normalize raw embedding records for ChromaDB insertion.

    Skips entries that are missing required fields or have invalid embeddings.
    """
    prepared: list[dict] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(raw_records, start=1):
        chunk_id = record.get("chunk_id")
        text = record.get("text")
        embedding = record.get("embedding")

        if not chunk_id or not text or not embedding:
            print(f"  [SKIP] Record {index}: missing chunk_id, text, or embedding.")
            continue

        if not isinstance(embedding, list) or not embedding:
            print(f"  [SKIP] Record {index}: invalid embedding vector.")
            continue

        metadata = parse_chunk_metadata(str(chunk_id))
        doc_id = build_document_id(metadata)

        if doc_id in seen_ids:
            print(f"  [SKIP] Duplicate ID in source file: {doc_id}")
            continue

        seen_ids.add(doc_id)
        prepared.append(
            {
                "id": doc_id,
                "document": text.strip(),
                "embedding": embedding,
                "metadata": metadata,
            }
        )

    return prepared


def get_chroma_client(persist_directory: Path) -> chromadb.ClientAPI:
    """Initialize a persistent local ChromaDB client."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def get_or_create_collection(client: chromadb.ClientAPI) -> Collection:
    """Return the agricultural knowledge collection, creating it if needed."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Agricultural advisory knowledge base"},
    )


def get_existing_ids(collection: Collection) -> set[str]:
    """Fetch IDs already stored in the collection to avoid duplicate inserts."""
    existing = collection.get(include=[])
    ids = existing.get("ids", [])
    return set(ids)


def batch_insert_records(collection: Collection, records: list[dict]) -> int:
    """
    Insert records into ChromaDB in batches.

    Returns the number of documents inserted.
    """
    if not records:
        return 0

    existing_ids = get_existing_ids(collection)
    new_records = [record for record in records if record["id"] not in existing_ids]

    if not new_records:
        print("All records already exist in ChromaDB. No new inserts required.")
        return 0

    skipped = len(records) - len(new_records)
    if skipped:
        print(f"Skipping {skipped} records already present in the collection.")

    inserted = 0
    for start in range(0, len(new_records), BATCH_SIZE):
        batch = new_records[start : start + BATCH_SIZE]
        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )
        inserted += len(batch)
        print(f"  Inserted batch: {inserted}/{len(new_records)}")

    return inserted


def load_embedding_model() -> SentenceTransformer:
    """Load the same sentence-transformers model used during embedding generation."""
    return SentenceTransformer(MODEL_NAME)


def retrieve_similar_chunks(
    collection: Collection,
    model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K,
    crop: str | None = None,
) -> list[dict]:
    """
    Embed a user query and return the top-k most similar chunks with metadata.

    When ``crop`` is provided (e.g. "wheat"), results are restricted to that crop
    first. If that yields no matches, the search falls back to the full collection.

    Returns a list of dicts containing id, document text, metadata, and distance.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    query_embedding = model.encode(query.strip()).tolist()
    where_filter = None
    if crop and crop.strip().lower() not in {"", "unknown"}:
        where_filter = {"crop": crop.strip().lower()}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    if where_filter and not results.get("ids", [[]])[0]:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    matches: list[dict] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        matches.append(
            {
                "id": doc_id,
                "text": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )

    return matches


def print_retrieval_results(query: str, results: list[dict]) -> None:
    """Log semantic search results in a readable format."""
    print(f"\nQuery: {query}")
    print(f"Top-{len(results)} results:")
    print("-" * 60)

    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        metadata = item["metadata"]
        preview = item["text"][:180].replace("\n", " ")
        if len(item["text"]) > 180:
            preview += "..."

        print(f"[{rank}] ID       : {item['id']}")
        print(f"    Crop     : {metadata.get('crop', 'unknown')}")
        print(f"    Source   : {metadata.get('source', 'unknown')}")
        print(f"    Filename : {metadata.get('filename', 'unknown')}")
        print(f"    Distance : {item['distance']:.4f}")
        print(f"    Text     : {preview}")
        print("-" * 60)


def run_retrieval_tests(collection: Collection, model: SentenceTransformer) -> None:
    """Run sample queries to verify semantic retrieval."""
    test_queries = [
    "When should wheat be irrigated?",
    "Best practices for wheat harvesting?",
    "How often should rice fields be watered?",
    "Common diseases in rice crops?",
    "How to manage cotton pests?",
    "Fertilizers used for cotton?",
    "When should maize be planted?",
    "How to improve maize yield?",
    "Irrigation schedule for sugarcane?",
    "How to control weeds in sugarcane?"
]

    print("\n====================================")
    print("Semantic Retrieval Test")
    print("====================================")

    for query in test_queries:
        try:
            results = retrieve_similar_chunks(collection, model, query, top_k=TOP_K)
            print_retrieval_results(query, results)
        except Exception as exc:
            print(f"Retrieval failed for query '{query}': {exc}")


def main() -> None:
    print("====================================")
    print("ChromaDB Loader")
    print("====================================")
    print(f"Embeddings file : {EMBEDDINGS_FILE.relative_to(PROJECT_ROOT)}")
    print(f"ChromaDB path   : {CHROMA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Collection      : {COLLECTION_NAME}")
    print("====================================\n")

    raw_records = load_embeddings_file(EMBEDDINGS_FILE)
    print(f"Loaded {len(raw_records)} records from embeddings file.")

    records = prepare_records(raw_records)
    print(f"Prepared {len(records)} valid records for insertion.\n")

    if not records:
        print("No valid records to insert. Exiting.")
        return

    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)

    print("Inserting records into ChromaDB...")
    inserted = batch_insert_records(collection, records)

    collection_size = collection.count()
    print("\n====================================")
    print("Load Complete")
    print("====================================")
    print(f"Documents inserted : {inserted}")
    print(f"Collection size    : {collection_size}")
    print("====================================")

    model = load_embedding_model()
    run_retrieval_tests(collection, model)


if __name__ == "__main__":
    main()