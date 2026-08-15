import os
import json
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
CHUNKS_DIR = "../data/chunks"
OUTPUT_FILE = "../data/embeddings.json"

# =========================
# LOAD EXISTING EMBEDDINGS  (so reruns only embed NEW chunks)
# =========================
existing_data = []
existing_ids = set()

if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_ids = {item["chunk_id"] for item in existing_data}
        print(f"Loaded {len(existing_data)} existing embeddings from {OUTPUT_FILE}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ Could not read existing embeddings file ({e}); starting fresh.")
        existing_data = []
        existing_ids = set()
else:
    print("No existing embeddings file found; starting fresh.")

# =========================
# FIND NEW CHUNKS TO PROCESS
# =========================
new_chunk_paths = []

for root, dirs, files in os.walk(CHUNKS_DIR):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            if file_path in existing_ids:
                continue
            new_chunk_paths.append(file_path)

skipped_count = 0
new_data = []

if not new_chunk_paths:
    print("\nNo new chunks found — everything is already embedded.")
else:
    # =========================
    # LOAD MODEL  (only if there's actually new work to do)
    # =========================
    print(f"\nFound {len(new_chunk_paths)} new chunk(s) to embed.")
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Starting embedding generation...\n")

    for file_path in new_chunk_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                print(f"⚠️ Skipping empty file: {file_path}")
                skipped_count += 1
                continue

            # Generate embedding
            embedding = model.encode(text)

            # Store result
            new_data.append({
                "chunk_id": file_path,   # unique ID
                "text": text,
                "embedding": embedding.tolist()
            })

            print(f"✅ Processed: {file_path}")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

# =========================
# MERGE + SAVE OUTPUT
# =========================
all_data = existing_data + new_data

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2)

# =========================
# VERIFICATION OUTPUT
# =========================

assert len(all_data) == len(existing_data) + len(new_data), "Mismatch in chunk and embedding count!"

print("\n=========================")
print("Embedding Generation Complete!")
print("=========================")
print(f"Previously embedded : {len(existing_data)}")
print(f"Newly processed      : {len(new_data)}")
print(f"Skipped (empty)      : {skipped_count}")
print(f"Total embeddings stored: {len(all_data)}")

if len(all_data) > 0:
    print(f"Embedding dimension: {len(all_data[0]['embedding'])}")  # should be 384
else:
    print("⚠️ No embeddings generated!")

print(f"\nSaved to: {OUTPUT_FILE}")