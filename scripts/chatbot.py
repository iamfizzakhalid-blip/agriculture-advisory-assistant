import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load ChromaDB
client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="chroma_db"
    )
)

collection = client.get_or_create_collection(name="wheat_data")

# Groq client
groq_client = Groq(api_key=api_key)

print("🌾 Wheat AI Assistant Ready! Type 'exit' to quit.\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    # Step 1: Embed query
    query_embedding = model.encode(query).tolist()

    # Step 2: Retrieve context
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(results['documents'][0])

    # Step 3: Send to Groq LLM
    prompt = f"""
You are an agriculture expert assistant for Pakistani farmers.

Answer the question  using ONLY the context below.
If answer is not in context, say "I don't have enough information." 

Context:
{context}

Question:
{query}

Answer:
"""

    response = groq_client.chat.completions.create(
       model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("\n🌾 AI Answer:\n")
    print(response.choices[0].message.content)
    print("\n----------------------\n")