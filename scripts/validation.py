"""
Validation layer using a second Groq LLM.

This module provides pre-RAG query validation and post-RAG answer validation
using a separate Groq API key (SECOND_GROQ_API_KEY). The existing Groq client
and API key are NOT modified or used here.

If the second API key is missing or the validation model fails, all functions
fall back gracefully so the existing RAG pipeline continues to work uninterrupted.
"""

import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

from scripts.groq_utils import complete_chat

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import streamlit as st
except Exception:
    st = None

load_dotenv()


# ------------------------------------------------------------------
# Client setup (completely separate from the primary Groq client)
# ------------------------------------------------------------------

def _get_second_api_key():
    """Retrieve the second Groq API key from env or Streamlit secrets."""
    api_key = os.getenv("SECOND_GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as _st
        return _st.secrets.get("SECOND_GROQ_API_KEY")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_validation_client():
    """Return a dedicated Groq client for the validation layer, or None."""
    if Groq is None:
        return None

    api_key = _get_second_api_key()

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ------------------------------------------------------------------
# JSON helpers (reuse patterns from lang_utils)
# ------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fenced_match = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        return fenced_match.group(1).strip()
    return stripped


def _extract_json_payload(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return cleaned[start : end + 1]
    return cleaned


def _safe_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON from LLM output, returning None on failure."""
    try:
        payload = _extract_json_payload(text)
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


# ------------------------------------------------------------------
# Pre-RAG query validation
# ------------------------------------------------------------------

def _default_pre_validation(user_query: str, english_query: str) -> dict:
    """Fallback result when validation is unavailable."""
    return {
        "language": "unknown",
        "is_translation_valid": True,
        "intent_aligned": True,
        "is_agriculture_related": True,
        "query_for_rag": english_query,
        "reason": "Validation model unavailable; passing through.",
    }


def validate_query_pre_rag(user_query: str, english_query: str) -> dict:
    """
    Validate the user query and its English translation BEFORE sending to RAG.

    Parameters
    ----------
    user_query : str
        The original query as typed by the user.
    english_query : str
        The English translation produced by normalize_query (lang_utils).

    Returns
    -------
    dict with keys:
        language, is_translation_valid, intent_aligned,
        is_agriculture_related, query_for_rag, reason
    """
    client = _get_validation_client()
    if client is None:
        return _default_pre_validation(user_query, english_query)

    system_prompt = (
        "You are a strict validation layer for a Pakistani agriculture advisory chatbot. "
        "You will receive the user's original query and an English translation of that query. "
        "Your job is to validate the translation and classify the query.\n\n"
        "Return a JSON object with EXACTLY these keys:\n"
        "- language (string): the language/script of the ORIGINAL query. "
        "Use one of: en, ur, roman_ur, or the ISO code if another language.\n"
        "- is_translation_valid (boolean): true if the English translation "
        "accurately represents the original query.\n"
        "- intent_aligned (boolean): true if the original query and the English "
        "translation share the same intent/meaning.\n"
        "- is_agriculture_related (boolean): true only for agriculture/farm questions or conversational "
        "chatbot interactions that belong in this assistant. false for unrelated real-world topics such as "
        "weather, temperatures, sports scores, news, politics, entertainment, cryptocurrency prices, and general non-agriculture questions.\n"
        "- query_for_rag (string): the best English query to send to the retrieval system. "
        "If the query is conversational or clearly out of scope, set this to an empty string or a safe short greeting. "
        "For agriculture questions, if the translation is correct, use it; if it is wrong or misleading, provide a corrected English version based on the original query's actual meaning.\n"
        "- reason (string): a brief explanation of your assessment.\n\n"
        "IMPORTANT:\n"
        "- Greeting and conversational questions (hello, hi, how are you, who are you, what do you do, thank you, goodbye, etc.), identify them on your own, are valid and should NOT be sent to RAG.\n"
        "- Weather, news, politics, sports, entertainment, Bitcoin, and other out-of-scope questions are not agriculture-related and should not be sent to RAG.\n"
        "- Do NOT blindly trust the translation. If the original and English versions do not match, prefer the original query's true meaning.\n"
        "- If there is suspicious or mistranslated wording that could cause incorrect retrieval, flag it and provide a safe corrected query.\n"
        "- Return ONLY valid JSON, no markdown, no commentary."
    )

    user_prompt = (
        f"Original query: {user_query}\n"
        f"English translation: {english_query}"
    )

    try:
        content, _ = complete_chat(
            client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0,
        )
        if not content:
            return _default_pre_validation(user_query, english_query)
        parsed = _safe_parse_json(content)

        if parsed is None:
            return _default_pre_validation(user_query, english_query)

        # Ensure all required keys exist with safe defaults
        result = {
            "language": parsed.get("language", "unknown"),
            "is_translation_valid": parsed.get("is_translation_valid", True),
            "intent_aligned": parsed.get("intent_aligned", True),
            "is_agriculture_related": parsed.get("is_agriculture_related", True),
            "query_for_rag": parsed.get("query_for_rag", english_query),
            "reason": parsed.get("reason", ""),
        }

        # Safety: if the validated query_for_rag is empty, fall back
        if not result["query_for_rag"] or not result["query_for_rag"].strip():
            result["query_for_rag"] = english_query

        return result

    except Exception:
        return _default_pre_validation(user_query, english_query)


# ------------------------------------------------------------------
# Post-RAG answer validation
# ------------------------------------------------------------------

def _default_post_validation(answer: str) -> dict:
    """Fallback result when validation is unavailable."""
    return {
        "is_relevant": True,
        "is_supported": True,
        "language_matches": True,
        "contains_hindi_wording": False,
        "final_answer": answer,
        "reason": "Validation model unavailable; passing through.",
    }


def validate_answer_post_rag(
    user_query: str,
    detected_language: str,
    context: str,
    generated_answer: str,
) -> dict:
    """
    Validate the RAG-generated answer BEFORE displaying it to the user.

    Parameters
    ----------
    user_query : str
        The original query as typed by the user.
    detected_language : str
        The detected language of the original query (e.g. en, ur, roman_ur).
    context : str
        The retrieved context/sources used by the RAG pipeline.
    generated_answer : str
        The answer produced by the RAG pipeline (possibly translated).

    Returns
    -------
    dict with keys:
        is_relevant, is_supported, language_matches,
        contains_hindi_wording, final_answer, reason
    """
    client = _get_validation_client()
    if client is None:
        return _default_post_validation(generated_answer)

    system_prompt = (
        "You are a strict answer-validation layer for a Pakistani agriculture advisory chatbot.\n\n"
        "You will receive:\n"
        "1. The user's original query\n"
        "2. The detected language of that query\n"
        "3. The retrieved context/sources from the knowledge base\n"
        "4. The generated answer\n\n"
        "Your job is to validate the generated answer and return a JSON object "
        "with EXACTLY these keys:\n"
        "- is_relevant (boolean): true if the answer is relevant to the user's actual question.\n"
        "- is_supported (boolean): true if the answer is supported by the retrieved context.\n"
        "- language_matches (boolean): true if the answer is in the same language as the user's query.\n"
        "- contains_hindi_wording (boolean): true if the answer accidentally uses Hindi/Sanskrit-derived "
        "vocabulary when the user is using Urdu or Roman Urdu. Check for Hindi/Sanskrit words like: "
        "matra, spasht, adhik, krishi, jal, avashyak, ropan, beejan, fasal utpadan, mitti prabandhan, "
        "khet, kheti, kisaan, paudha, ped, ugana, gehun, dhan, kapas, keetnashak, kharpatwar, "
        "sinchai, upaj, vishesh, samay, prakar, pranali, sambhav, samasya, suraksha, vikas, kshetra, "
        "prayas, parinaam. If ANY of these appear, replace them with Pakistani Urdu equivalents.\n"
        "- final_answer (string): if the answer is good, return it unchanged. "
        "If the answer has problems (wrong language, Hindi words in Urdu answer, irrelevant content, "
        "hallucinated information), provide a corrected/rewritten version. "
        "The corrected answer MUST be in the SAME language as the user's original query.\n"
        "- reason (string): brief explanation of your assessment.\n\n"
        "IMPORTANT:\n"
        "- If the answer contains hallucinated or unsupported agricultural information, "
        "remove it and keep only supported facts.\n"
        "- If the generated answer is already in English, keep final_answer in English. "
        "Do not translate it in this step.\n"
        "- NEVER replace a relevant, context-supported answer with "
        "'I do not have enough information to answer this question.' "
        "Use that message only when the answer is unsupported or irrelevant.\n"
        "- If the answer is completely irrelevant, rewrite it as: "
        "'I do not have enough information to answer this question.'\n"
        "- Return ONLY valid JSON, no markdown, no commentary."
    )

    # Truncate context to avoid hitting token limits
    context_truncated = context[:3000] if len(context) > 3000 else context

    user_prompt = (
        f"User query: {user_query}\n"
        f"Detected language: {detected_language}\n"
        f"Retrieved context:\n{context_truncated}\n\n"
        f"Generated answer:\n{generated_answer}"
    )

    try:
        content, _ = complete_chat(
            client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0,
            continue_on_length=True,
        )
        if not content:
            return _default_post_validation(generated_answer)
        parsed = _safe_parse_json(content)

        if parsed is None:
            return _default_post_validation(generated_answer)

        result = {
            "is_relevant": parsed.get("is_relevant", True),
            "is_supported": parsed.get("is_supported", True),
            "language_matches": parsed.get("language_matches", True),
            "contains_hindi_wording": parsed.get("contains_hindi_wording", False),
            "final_answer": parsed.get("final_answer", generated_answer),
            "reason": parsed.get("reason", ""),
        }

        # Safety: if final_answer is empty, fall back to the original
        if not result["final_answer"] or not result["final_answer"].strip():
            result["final_answer"] = generated_answer

        return result

    except Exception:
        return _default_post_validation(generated_answer)
