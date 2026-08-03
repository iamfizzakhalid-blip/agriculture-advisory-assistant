import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

try:
    from groq import Groq
except Exception:
    Groq = None


load_dotenv()


def _get_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_groq_client():
    if Groq is None:
        return None

    api_key = _get_api_key()

    if not api_key:
        return None

    return Groq(api_key=api_key)


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


def _fallback_result(user_query: str) -> dict:
    return {
        "detected_language": "en",
        "english_query": user_query,
    }


def _clean_english_query(user_query: str) -> str:
    return " ".join(user_query.split()).strip()


def normalize_query(user_query: str) -> dict:
    # Downstream callers rely on this exact JSON shape:
    # {"detected_language": "en|ur|roman_ur", "english_query": "..."}
    expected_keys = {"detected_language", "english_query"}

    if not user_query:
        return _fallback_result(user_query)

    client = _get_groq_client()

    if client is None:
        return _fallback_result(user_query)

    system_prompt = (
        "You detect whether a user query is English, Urdu in Arabic script, "
        "or Roman Urdu in Latin letters, and you translate non-English queries "
        "into plain English for retrieval. Return strict JSON only with exactly "
        "these keys: detected_language and english_query. Use detected_language "
        "values only from en, ur, roman_ur. If the input is already English, "
        "keep english_query as a lightly cleaned version of the original query "
        "and do not rewrite its meaning. If the input is Urdu or Roman Urdu, "
        "translate it to clear natural English. Do not add markdown, code fences, "
        "or commentary."
    )

    user_prompt = (
        "Analyze this query and respond with JSON only:\n"
        f"{user_query}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        payload = _extract_json_payload(content)
        parsed = json.loads(payload)

        if not isinstance(parsed, dict):
            return _fallback_result(user_query)

        detected_language = parsed.get("detected_language")
        english_query = parsed.get("english_query")

        if detected_language not in {"en", "ur", "roman_ur"}:
            return _fallback_result(user_query)

        if not isinstance(english_query, str):
            return _fallback_result(user_query)

        if detected_language == "en":
            english_query = _clean_english_query(user_query)

        result = {
            "detected_language": detected_language,
            "english_query": english_query,
        }

        if set(result.keys()) != expected_keys:
            return _fallback_result(user_query)

        return result

    except Exception:
        return _fallback_result(user_query)