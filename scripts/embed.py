import os
import json
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
CHUNKS_DIR = "../data/chunks"
OUTPUT_FILE = "../data/embeddings.json"

# =========================
# LOAD MODEL
# =========================
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# PROCESS CHUNKS
# =========================
all_data = []
file_count = 0

print("Starting embedding generation...\n")

for root, dirs, files in os.walk(CHUNKS_DIR):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                if not text:
                    print(f"⚠️ Skipping empty file: {file_path}")
                    continue

                # Generate embedding
                embedding = model.encode(text)

                # Store result
                all_data.append({
                    "chunk_id": file_path,   # unique ID
                    "text": text,
                    "embedding": embedding.tolist()
                })

                file_count += 1
                print(f"✅ Processed: {file_path}")

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

# =========================
# SAVE OUTPUT
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2)

# =========================
# VERIFICATION OUTPUT
# =========================

assert file_count == len(all_data), "Mismatch in chunk and embedding count!"

print("\n=========================")
print("Embedding Generation Complete!")
print("=========================")
print(f"Total files processed: {file_count}")
print(f"Total embeddings stored: {len(all_data)}")

if len(all_data) > 0:
    print(f"Embedding dimension: {len(all_data[0]['embedding'])}")  # should be 384
else:
    print("⚠️ No embeddings generated!")

print(f"\nSaved to: {OUTPUT_FILE}")