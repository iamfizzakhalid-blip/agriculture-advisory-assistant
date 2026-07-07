import os
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="chroma_db"
    )
)
collection = client.create_collection(name="wheat_data")

input_folder = "data/clean/wheat"

chunk_size = 300

def chunk_text(text):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


doc_id = 0

for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        file_path = os.path.join(input_folder, file)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(doc_id)]
            )

            doc_id += 1

print("All data stored in ChromaDB!")
