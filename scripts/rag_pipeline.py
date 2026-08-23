import re

from scripts.load_db import (
    get_chroma_client,
    get_or_create_collection,
    load_embedding_model,
    retrieve_similar_chunks,
    CHROMA_DIR,
)

from scripts.llm_call import ask_llm
from scripts.lang_utils import (
    normalize_query,
    resolve_output_language,
    translate_to_original_language,
)
from scripts.validation import validate_query_pre_rag, validate_answer_post_rag

INSUFFICIENT_ANSWER = "I do not have enough information to answer this question."

CROP_TERMS = {
    "wheat": ["wheat", "gandum", "gehun", "گندم"],
    "rice": ["rice", "chawal", "dhaan", "dhan", "چاول"],
    "cotton": ["cotton", "kapas", "kapaas", "کپاس"],
    "sugarcane": ["sugarcane", "ganna", "ganne", "گنا"],
    "maize": ["maize", "corn", "makai", "makki", "مکئی", "مکئي", "مکی"],
}

TOP_K = 8
MAX_CONTEXT_CHARS = 6000
MAX_LLM_CHUNKS = 5


class RagResult(dict):
    def __iter__(self):
        yield self.get("results", [])
        yield self.get("answer", "")


def normalize_answer_text(text: str) -> str:
    """Replace unicode spaces/dashes that break Windows console encoding."""
    if not text:
        return text
    replacements = {
        "\u202f": " ",
        "\u00a0": " ",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def detect_crop(query: str) -> str:
    query_lower = query.lower()

    for crop, terms in CROP_TERMS.items():
        for term in terms:
            if term in query_lower or term in query:
                return crop

    # Wheat-specific terms when the crop name is not mentioned explicitly.
    wheat_terms = [
        "black rust",
        "brown rust",
        "stripe rust",
        "yellow rust",
        "stem rust",
        "leaf rust",
        "puccinia graminis",
        "puccinia recondita",
        "puccinia striiformis",
        "karnal bunt",
        "loose smut",
    ]
    if any(term in query_lower for term in wheat_terms):
        return "wheat"

    return "Unknown"


def refine_results_by_dominant_crop(results: list[dict]) -> list[dict]:
    """Keep chunks from the dominant crop in the top results when no crop was named."""
    if len(results) < 2:
        return results

    top_crops = [
        item["metadata"].get("crop", "unknown")
        for item in results[:5]
        if item.get("metadata")
    ]
    crop_counts: dict[str, int] = {}
    for crop in top_crops:
        if crop not in {"", "unknown"}:
            crop_counts[crop] = crop_counts.get(crop, 0) + 1

    if not crop_counts:
        return results

    dominant_crop = max(crop_counts, key=crop_counts.get)
    if crop_counts[dominant_crop] < 2:
        return results

    filtered = [
        item for item in results
        if item.get("metadata", {}).get("crop") == dominant_crop
    ]
    return filtered or results


def build_context(results, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join top retrieved chunks, capped to avoid exceeding the LLM context window."""
    parts: list[str] = []
    total_len = 0

    for result in results[:MAX_LLM_CHUNKS]:
        text = result.get("text", "").strip()
        if not text:
            continue

        separator_len = 2 if parts else 0
        if total_len + separator_len + len(text) > max_chars:
            remaining = max_chars - total_len - separator_len
            if remaining > 200:
                parts.append(text[:remaining] + "...")
            break

        parts.append(text)
        total_len += separator_len + len(text)

    return "\n\n".join(parts)


def _match_phrase(text: str, phrase: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    return re.search(rf"\b{re.escape(phrase.lower())}\b", normalized) is not None


def classify_user_intent(question: str) -> dict:
    q = (question or "").strip()
    if not q:
        return {"category": "conversation", "response": "Hello! I’m your Agriculture Assistant. How can I help with your crops today?"}

    text = q.lower()

    greetings = [
        "hello", "hi", "hey", "assalam", "assalam o alaikum", "assalamualaikum",
        "salam", "good morning", "good evening", "good night"
    ]
    if any(_match_phrase(text, g) for g in greetings):
        return {"category": "conversation", "response": "Hello! I’m your Agriculture Assistant. How can I help with your crops today?"}

    thanks = ["thank you", "thanks", "many thanks", "shukriya", "shukria"]
    if any(_match_phrase(text, t) for t in thanks):
        return {"category": "conversation", "response": "You’re welcome! I’m here to help with crop, soil, irrigation, pest, and farm-management questions."}

    if any(_match_phrase(text, p) for p in ["goodbye", "bye", "see you", "take care"]):
        return {"category": "conversation", "response": "Goodbye! Feel free to ask me about your crops or farm management anytime."}

    if _match_phrase(text, "how are you"):
        return {"category": "conversation", "response": "I’m doing well — I’m here to help with agriculture and farm questions."}

    if any(_match_phrase(text, p) for p in ["who are you", "what are you", "what do you do", "what are you doing"]):
        return {"category": "conversation", "response": "I’m your Agriculture Assistant. I help with crop advice, irrigation, fertilizer, pest control, sowing decisions, and farm management."}

    if any(_match_phrase(text, p) for p in ["who am i", "what is my name", "who is this", "who am i in this chat"]):
        return {"category": "conversation", "response": "I don’t know your personal identity unless you share it in this chat. I only know what is explicitly written here."}

    if _match_phrase(text, "what can you help me with") or _match_phrase(text, "what can you help with"):
        return {"category": "conversation", "response": "I can help with crop planning, sowing time, fertilizer, irrigation, pest and disease management, soil health, and general farm advice."}

    out_of_scope_terms = [
        "weather", "temperature", "today's temperature", "today temperature", "bitcoin",
        "cricket", "football", "news", "politics", "election", "movie", "entertainment",
        "stock market", "currency", "gold price", "latest news", "who won"
    ]

    # Check agriculture keywords FIRST — if the question is clearly about
    # agriculture, allow it even if it incidentally contains an out-of-scope
    # word (e.g. "weather conditions that favor wheat rust").
    if _looks_agriculture_related(text):
        return {"category": "agriculture", "response": None}

    if any(term in text for term in out_of_scope_terms):
        return {"category": "out_of_scope", "response": "I'm designed for agriculture-related assistance. Please ask about crops, irrigation, pests, fertilizer, soil, or farm management."}

    # Default to agriculture for ambiguous or partially agricultural questions.
    # This prevents valid farm questions from being rejected as out-of-scope
    # when the retrieval later determines there is not enough information.
    return {"category": "agriculture", "response": None}


def _looks_agriculture_related(text: str) -> bool:
    if not text:
        return False

    q = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    q = re.sub(r"\s+", " ", q).strip()
    agriculture_keywords = [
        "wheat", "rice", "maize", "cotton", "sugarcane", "crop", "farming",
        "fertilizer", "irrigation", "pest", "disease", "seed", "harvest",
        "sowing", "planting", "soil", "field", "farmer", "agriculture",
        "pani", "beej", "fasal", "kheti", "zaroori", "khaad", "kheti", "naukar", "kisan",
        "khaad", "watering", "drip", "mulch", "yield", "weeds", "fungicide",
        "insecticide", "pesticide", "land", "farm", "tractor", "cropping", "crops",
        "kya", "ke", "liye", "fertilizer", "cotton", "cash crop", "sowing time",
        "rust", "pathogen", "wilt", "blight", "smut", "bunt", "lodging",
        "nutrient", "nitrogen", "phosphorus", "potassium", "urea", "dap",
        "variety", "varieties", "germination", "tillering", "herbicide",
        "makai", "makki", "gandum", "chawal", "dhaan", "kapaas", "ganna",
    ]
    if any(keyword in q for keyword in agriculture_keywords):
        return True
    urdu_keywords = ["فصل", "بیج", "مکئی", "گندم", "چاول", "کپاس", "گنا", "کھاد", "کاشت", "کسان"]
    return any(keyword in text for keyword in urdu_keywords)


def _is_insufficient_response(text: str) -> bool:
    if not text:
        return False
    lower = text.lower().strip()
    insufficient_markers = [
        "i do not have enough information to answer this question",
        "i don't have enough information to answer this question",
        "not enough information",
        "no relevant documents found",
        "unrelated question",
        "not related to the question",
        "unable to answer this question",
        "کافی معلومات نہیں",
        "kafi maloomat nahi",
        "kaafi maloomat nahi",
        "maloomat nahi hai",
        "maloomat nahin",
    ]
    return any(marker in lower or marker in text for marker in insufficient_markers)


def _clean_source_name(item: dict) -> str:
    metadata = item.get("metadata") or {}
    filename = metadata.get("filename") or metadata.get("source") or "Agriculture Source"
    name = filename.split("/")[-1]
    if name.lower().endswith(".txt"):
        name = name[:-4]
    name = name.replace("_", " ").strip()
    # Remove chunk suffix like " chunk 001"
    name = re.sub(r"\s+chunk\s+\d+$", "", name, flags=re.IGNORECASE).strip()
    return name or "Agriculture Source"


def _build_source_list(results):
    seen = set()
    sources = []
    for item in results:
        source_name = _clean_source_name(item)
        if source_name not in seen:
            seen.add(source_name)
            sources.append(source_name)
    return sources


def _classify_response(
    question: str,
    answer: str,
    validation: dict | None,
    english_query: str | None = None,
) -> str:
    if not answer:
        return "insufficient_info"

    al = answer.lower().strip()

    # If the LLM itself produced a refusal / insufficient answer, honour that.
    if _is_insufficient_response(answer):
        return "insufficient_info"
    if "unrelated" in al or "not related" in al:
        return "unrelated"

    agriculture_text = " ".join(
        part for part in (question, english_query) if part
    )
    if validation:
        if validation.get("is_relevant") is False and not _looks_agriculture_related(agriculture_text):
            return "unrelated"

    if not _looks_agriculture_related(agriculture_text):
        return "unrelated"

    return "answered"


def _localize_text(text: str, lang: str) -> str:
    if not text or lang == "en":
        return text
    translated = translate_to_original_language(text, lang)
    if not translated or not str(translated).strip():
        return text
    return translated


def rag_pipeline(question, collection, model):

    normalized_query = normalize_query(question)
    retrieval_query = normalized_query.get("english_query", question)
    final_lang = resolve_output_language(
        question,
        normalized_query.get("detected_language"),
    )

    intent = classify_user_intent(question)
    if intent["category"] == "conversation":
        return RagResult({
            "answer": _localize_text(intent["response"], final_lang),
            "status": "answered",
            "sources": [],
            "results": [],
        })
    if intent["category"] == "out_of_scope":
        return RagResult({
            "answer": _localize_text(intent["response"], final_lang),
            "status": "unrelated",
            "sources": [],
            "results": [],
        })

    pre_validation = validate_query_pre_rag(question, retrieval_query)
    if not pre_validation.get("is_translation_valid", True) or not pre_validation.get("intent_aligned", True):
        corrected_query = pre_validation.get("query_for_rag", retrieval_query)
        if corrected_query and corrected_query.strip():
            retrieval_query = corrected_query

    query_crop = detect_crop(retrieval_query)
    if query_crop == "Unknown":
        query_crop = detect_crop(question)

    results = retrieve_similar_chunks(
        collection=collection,
        model=model,
        query=retrieval_query,
        top_k=TOP_K,
        crop=query_crop,
    )

    if not results:
        return RagResult({
            "answer": _localize_text(INSUFFICIENT_ANSWER, final_lang),
            "status": "insufficient_info",
            "sources": [],
            "results": [],
        })

    results = refine_results_by_dominant_crop(results)

    metadata = results[0]["metadata"]
    metadata["crop"] = query_crop if query_crop != "Unknown" else detect_crop(question)

    context = build_context(results)
    english_answer = normalize_answer_text(ask_llm(retrieval_query, context))

    # Validate the English answer against the English context so the same
    # facts are used for every language. Translation happens after this.
    post_validation = validate_answer_post_rag(
        user_query=retrieval_query,
        detected_language="en",
        context=context,
        generated_answer=english_answer,
    )

    validated_answer = post_validation.get("final_answer", english_answer)
    if validated_answer and validated_answer.strip():
        if (
            _is_insufficient_response(validated_answer)
            and english_answer.strip()
            and not _is_insufficient_response(english_answer)
        ):
            pass
        else:
            english_answer = normalize_answer_text(validated_answer)

    english_answer = normalize_answer_text(english_answer)
    if not english_answer or not english_answer.strip():
        english_answer = INSUFFICIENT_ANSWER
    final_answer = _localize_text(english_answer, final_lang)
    if not final_answer or not str(final_answer).strip():
        final_answer = english_answer

    status = _classify_response(
        question,
        final_answer,
        post_validation,
        english_query=retrieval_query,
    )
    source_list = _build_source_list(results)

    return RagResult({
        "answer": final_answer,
        "status": status,
        "sources": source_list,
        "results": results,
    })


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
