from scripts.load_db import (
    get_chroma_client,
    get_or_create_collection,
    load_embedding_model,
    retrieve_similar_chunks,
    CHROMA_DIR,
)

from scripts.llm_call import ask_llm
from scripts.lang_utils import normalize_query, translate_to_original_language

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


# 🔥 NEW: Roman Urdu detection helper
def detect_script_type(text: str) -> str:
    """
    Detect if input is:
    - Urdu script (Arabic)
    - Roman Urdu (Latin)
    - English
    """

    has_urdu = any('\u0600' <= c <= '\u06FF' for c in text)
    has_latin = any(c.isalpha() and c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text)

    if has_urdu:
        return "ur"
    elif has_latin:
        return "roman_ur"
    else:
        return "en"


def rag_pipeline(question, collection, model):

    # 🔹 Step 1: Normalize query (LLM-based)
    normalized_query = normalize_query(question)

    print(f"Detected language (LLM): {normalized_query.get('detected_language')}")
    print(f"English query: {normalized_query.get('english_query', question)}")

    retrieval_query = normalized_query.get("english_query", question)

    # 🔥 Step 2: OVERRIDE language detection (fix Roman Urdu issue)
    script_detected_lang = detect_script_type(question)

    print(f"Detected language (script-based): {script_detected_lang}")

    # 🔹 Step 3: Retrieve chunks
    results = retrieve_similar_chunks(
        collection=collection,
        model=model,
        query=retrieval_query,
        top_k=TOP_K,
    )

    if not results:
        return None, "No relevant documents found."

    metadata = results[0]["metadata"]
    metadata["crop"] = detect_crop(question)

    # 🔹 Step 4: Build context
    context = build_context(results)

    # 🔹 Step 5: Get answer from LLM (IN ENGLISH)
    answer = ask_llm(retrieval_query, context)

    # 🔥 Step 6: Decide final language (priority: script detection)
    if script_detected_lang in ["ur", "roman_ur"]:
        final_lang = script_detected_lang
    else:
        final_lang = normalized_query.get("detected_language", "en")

    print(f"Final language used: {final_lang}")

    # 🔹 Step 7: Translate back
    if final_lang == "en":
        final_answer = answer
    else:
        final_answer = translate_to_original_language(answer, final_lang)

    return results, final_answer


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

            print_results(results, question)

            print("\n" + "=" * 70)
            print("Final Answer")
            print("=" * 70)
            print(answer)

        except Exception as e:

            error_message = str(e)

            if "429" in error_message:
                print("\n⚠️ API limit reached. Please wait a few minutes and try again.")
            else:
                print(f"\nError: {e}")


if __name__ == "__main__":
    main()
