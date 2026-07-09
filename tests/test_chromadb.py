import chromadb

client = chromadb.Client()

collection = client.create_collection("test")

collection.add(
    documents=["Wheat grows well in cool weather."],
    ids=["1"]
)

result = collection.query(
    query_texts=["wheat"],
    n_results=1
)

print(result["documents"])