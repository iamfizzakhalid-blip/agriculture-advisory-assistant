from scripts.load_db import (
    get_chroma_client,
    get_or_create_collection,
    load_embedding_model,
    retrieve_similar_chunks,
    CHROMA_DIR,
)

from scripts.llm_call import ask_llm

TOP_K = 10

def detect_crop(query: str) -> str:
    query = query.lower()

    crops = [
        "wheat",
        "rice",
        "cotton",
        "sugarcane",
        "maize",
    ]

    for crop in crops:
        if crop in query:
            return crop

    return "Unknown"

def build_context(results):
    return "\n\n".join(result["text"] for result in results)


def rag_pipeline(question, collection, model):

    results = retrieve_similar_chunks(
        collection=collection,
        model=model,
        query=question,
        top_k=TOP_K,
    )
    if not results:
        return None, "No relevant documents found."

    metadata = results[0]["metadata"]
    metadata["crop"] = detect_crop(question)
    
    context = build_context(results)
    answer = ask_llm(question, context)

    return results, answer


def print_results(results, query):

    print("\n" + "=" * 70)
    print("Retrieved Chunks")
    print("=" * 70)

    query_crop = detect_crop(query)

    for i, item in enumerate(results, start=1):

        metadata = item["metadata"]

        print(f"\n[{i}]")
        print(f"Crop     : {query_crop}")
        print(f"Source   : {metadata.get('source')}")
        print(f"Filename : {metadata.get('filename')}")
        print(f"Distance : {item['distance']:.4f}")

        preview = item["text"][:250].replace("\n", " ")
        if len(item["text"]) > 250:
            preview += "..."

        print(f"Text     : {preview}")


def main():

    print("=" * 70)
    print("Agriculture Advisory Assistant - RAG Pipeline")
    print("=" * 70)

    print("\nLoading embedding model...")
    model = load_embedding_model()

    print("Connecting to ChromaDB...")
    client = get_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)

    print("RAG pipeline is ready.\n")

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if not question:
            print("Please enter a valid question.\n")
            continue

        try:

            results, answer = rag_pipeline(
                question=question,
                collection=collection,
                model=model,
            )

            if results is None:
                print(answer)
                continue

            print_results(results,question)

            print("\n" + "=" * 70)
            print("Final Answer")
            print("=" * 70)
            print(answer)

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()