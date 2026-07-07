import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="chroma_db"
    )
)
collection = client.get_collection(name="wheat_data")

while True:
    query = input("\nAsk something about wheat: ")

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\n--- Answer Context ---\n")

    for doc in results['documents'][0]:
        print(doc)
        print("\n-----------------\n")