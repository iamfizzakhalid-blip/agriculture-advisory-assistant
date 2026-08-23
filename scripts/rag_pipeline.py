import re

from scripts.load_db import (
    get_chroma_client,
    get_or_create_collection,
    load_embedding_model,
    retrieve_similar_chunks,
    CHROMA_DIR,
)

from scripts.llm_call import ask_llm
from scripts.lang_utils import normalize_query, translate_to_original_language
from scripts.validation import validate_query_pre_rag, validate_answer_post_rag

TOP_K = 10


class RagResult(dict):
    def __iter__(self):
        yield self.get("results", [])
        yield self.get("answer", "")


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
    ]
    return any(keyword in q for keyword in agriculture_keywords)


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
    ]
    return any(marker in lower for marker in insufficient_markers)


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


def _classify_response(question: str, answer: str, validation: dict | None) -> str:
    if not answer:
        return "insufficient_info"

    ql = (question or "").lower().strip()
    al = answer.lower().strip()

    # If the LLM itself produced a refusal / insufficient answer, honour that.
    if _is_insufficient_response(al):
        return "insufficient_info"
    if "unrelated" in al or "not related" in al:
        return "unrelated"

    # If the answer is substantive, only let validation downgrade it when the
    # question is clearly not agriculture-related.
    if validation:
        if validation.get("is_relevant") is False and not _looks_agriculture_related(ql):
            return "unrelated"

    if not _looks_agriculture_related(ql):
        return "unrelated"

    # Substantive, agriculture-related answer → show it with sources.
    return "answered"


# Roman Urdu detection helper
def detect_script_type(text: str) -> str:
    """Simple script detector.

    Returns:
    - 'ur' for Arabic-script (Urdu) characters
    - 'latin' for Latin alphabet characters
    - 'other' for none of the above
    """

    has_urdu = any('\u0600' <= c <= '\u06FF' for c in text)
    has_latin = any(c.isalpha() and c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text)

    if has_urdu:
        return "ur"
    elif has_latin:
        return "latin"
    else:
        return "other"


def rag_pipeline(question, collection, model):

    intent = classify_user_intent(question)
    if intent["category"] in {"conversation", "out_of_scope"}:
        return RagResult({
            "answer": intent["response"],
            "status": "answered" if intent["category"] == "conversation" else "unrelated",
            "sources": [],
            "results": [],
        })

    def _is_conversational_or_identity(q: str) -> bool:
        """Return True for short conversational or identity questions we should not answer from LLM knowledge."""
        if not q:
            return False
        ql = q.lower().strip()
        # common conversational starters and identity questions
        conversational_tokens = [
            "hello",
            "hi",
            "hey",
            "who am i",
            "who are you",
            "who are",
            "what is my name",
            "what's my name",
            "where am i",
            "how are you",
            "introduce yourself",
        ]
        # exact-match short phrases or presence
        for t in conversational_tokens:
            if t in ql:
                return True

        # also treat very short non-specific queries as conversational
        tokens = ql.split()
        if len(tokens) <= 3 and any(w.endswith('?') or w in {"hello","hi","hey"} for w in tokens):
            return True

        return False


    # Normalize query (LLM-based)
    normalized_query = normalize_query(question)

    print(f"Detected language (LLM): {normalized_query.get('detected_language')}")
    print(f"English query: {normalized_query.get('english_query', question)}")

    retrieval_query = normalized_query.get("english_query", question)

    # ----------------------------------------------------------
    # PRE-RAG VALIDATION (Second Groq Model)
    # ----------------------------------------------------------
    pre_validation = validate_query_pre_rag(question, retrieval_query)
    print(f"Pre-RAG validation: {pre_validation}")

    # If the translation is invalid or intent is misaligned, use the
    # corrected query from the validation model instead.
    if not pre_validation.get("is_translation_valid", True) or not pre_validation.get("intent_aligned", True):
        corrected_query = pre_validation.get("query_for_rag", retrieval_query)
        if corrected_query and corrected_query.strip():
            print(f"Validation override: using corrected query -> {corrected_query}")
            retrieval_query = corrected_query

    # OVERRIDE language detection
    script_detected_lang = detect_script_type(question)

    print(f"Detected language (script-based): {script_detected_lang}")

    # Retrieve chunks
    results = retrieve_similar_chunks(
        collection=collection,
        model=model,
        query=retrieval_query,
        top_k=TOP_K,
    )

    if not results:
        return RagResult({
            "answer": "I do not have enough information to answer this question.",
            "status": "insufficient_info",
            "sources": [],
            "results": [],
        })

    metadata = results[0]["metadata"]
    metadata["crop"] = detect_crop(question)

    # Build context
    context = build_context(results)

    # Get answer from LLM (IN ENGLISH)
    # If the user asked a conversational/identity question, do not let the LLM
    # hallucinate an identity or greeting — instead return the standard
    # fallback message per the prompt template.
    if _is_conversational_or_identity(question):
        answer = "I do not have enough information to answer this question."
    else:
        answer = ask_llm(retrieval_query, context)

    # Decide final language
    # Prefer the LLM-based detection (`normalized_query`) for Roman Urdu vs English
    # because simple script checks (presence of Latin letters) can misclassify
    # English as Roman Urdu. However, if Arabic-script Urdu characters are
    # present, force 'ur' since that's unambiguous.
    final_lang = normalized_query.get("detected_language", "en")
    if script_detected_lang == "ur":
        # Arabic-script Urdu detected — force Urdu output
        final_lang = "ur"

    # Conservative heuristic fallback: if LLM returned 'en' but the script
    # detector saw Latin letters and the original query contains common
    # Roman-Urdu tokens, treat it as Roman Urdu. This helps when the LLM
    # detection or Groq client is unavailable.
    # Only consider Roman-Urdu if the script is Latin and roman-indicator tokens appear
    if final_lang == "en" and script_detected_lang == "latin":
        tokens = set(t.strip("?!.;,()").lower() for t in question.split())
        roman_indicators = {"ka", "ke", "ki", "hai", "hain", "nahi", "kab", "kya", "kyun", "kaise", "jais", "mera", "meri", "tum", "ap"}
        if tokens & roman_indicators:
            final_lang = "roman_ur"

    print(f"Final language used: {final_lang} (LLM-detected: {normalized_query.get('detected_language')}, script-detected: {script_detected_lang})")

    # Translate back
    if final_lang == "en":
        final_answer = answer
    else:
        final_answer = translate_to_original_language(answer, final_lang)

    # ----------------------------------------------------------
    # POST-RAG VALIDATION (Second Groq Model)
    # ----------------------------------------------------------
    post_validation = validate_answer_post_rag(
        user_query=question,
        detected_language=final_lang,
        context=context,
        generated_answer=final_answer,
    )
    print(f"Post-RAG validation: {post_validation}")

    # Use the validated/corrected answer
    validated_answer = post_validation.get("final_answer", final_answer)
    if validated_answer and validated_answer.strip():
        final_answer = validated_answer

    status = _classify_response(question, final_answer, post_validation)
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
