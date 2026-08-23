import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import streamlit as st
except Exception:
    st = None

# Load environment variables
load_dotenv()

from scripts.groq_utils import complete_chat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "scripts" / "prompt_template.txt"
MAX_CONTEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 2500

api_key = os.getenv("GROQ_API_KEY")

# If running on Streamlit Cloud, read from Secrets
if api_key is None and st is not None:
    api_key = st.secrets.get("GROQ_API_KEY")

client = Groq(api_key=api_key) if (Groq is not None and api_key) else None # Creates a connection object


def load_prompt_template():
    """
    Load the prompt template from file.
    """
    try:
        with PROMPT_TEMPLATE_PATH.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt template not found: {PROMPT_TEMPLATE_PATH}"
        )


def _truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    context = context.strip()
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n...[context truncated]..."


def _call_groq(
    messages: list[dict],
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> tuple[str | None, str | None]:
    """Send one chat completion request and return (content, finish_reason)."""
    content, finish_reason = complete_chat(
        client,
        messages,
        max_tokens=max_tokens,
        temperature=0.1,
        continue_on_length=True,
    )
    if content:
        return _normalize_llm_text(content), finish_reason
    return None, finish_reason


def _normalize_llm_text(text: str) -> str:
    """Replace unicode spaces/dashes that break Windows console encoding."""
    for src, dst in {
        "\u202f": " ",
        "\u00a0": " ",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
    }.items():
        text = text.replace(src, dst)
    return text


def ask_llm(question, context):
    """
    Sends the user question and retrieved context to the Groq LLM.
    Uses system/user message separation for better instruction adherence.
    """

    if client is None:
        return "Error communicating with Groq API: GROQ_API_KEY not found."

    # Validate inputs
    if not question.strip():
        return "Please enter a valid question."

    if not context.strip():
        return "No context was provided."

    prompt_template = load_prompt_template()
    context = _truncate_context(context)

    # Split the template into system instructions and user content.
    # The template contains instructions followed by context/question placeholders.
    # We separate them so the LLM treats instructions as authoritative system rules.
    final_prompt = prompt_template.format(
        context=context,
        question=question
    )

    # Find where the Retrieved Context section starts — everything before it
    # is system-level instructions, everything from it onward is user content.
    context_marker = "Retrieved Context:"
    marker_pos = final_prompt.find(context_marker)

    if marker_pos > 0:
        system_instructions = final_prompt[:marker_pos].strip()
        user_content = final_prompt[marker_pos:].strip()
    else:
        # Fallback: treat entire prompt as user message
        system_instructions = (
            "You are an Agriculture Advisory Assistant for Pakistani farmers. "
            "Answer ONLY from the provided context. Never use your own knowledge. "
            "If the context does not contain the answer, say: "
            "'I do not have enough information to answer this question.'"
        )
        user_content = final_prompt

    try:
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ]
        content, finish_reason = _call_groq(messages)

        # Continue if the model hit the output token limit mid-answer.
        if content and finish_reason == "length":
            continuation_messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Continue the answer from where you stopped. "
                        "Cover any remaining parts of the question that were not answered yet. "
                        "Do not repeat what you already wrote."
                    ),
                },
            ]
            extra_content, extra_reason = _call_groq(continuation_messages)
            if extra_content:
                content = f"{content}\n\n{extra_content}"
                finish_reason = extra_reason

        # Retry once with a smaller context if the model returns nothing.
        if content is None and len(context) > 2500:
            shorter_context = _truncate_context(context, max_chars=2500)
            retry_prompt = prompt_template.format(context=shorter_context, question=question)
            retry_marker_pos = retry_prompt.find(context_marker)
            if retry_marker_pos > 0:
                retry_system = retry_prompt[:retry_marker_pos].strip()
                retry_user = retry_prompt[retry_marker_pos:].strip()
            else:
                retry_system = system_instructions
                retry_user = retry_prompt
            retry_messages = [
                {"role": "system", "content": retry_system},
                {"role": "user", "content": retry_user},
            ]
            content, finish_reason = _call_groq(retry_messages)

        if content:
            return content

        return "I do not have enough information to answer this question."

    except Exception as e:
        return f"Error communicating with Groq API: {e}"


def main():

    print("=" * 60)
    print("Day 14: Prompt Template Test")
    print("=" * 60)

    # Temporary context until ChromaDB retrieval is integrated
    context = """
    Wheat should be sown from mid-October to mid-November.
    Rice requires standing water during most of its growth period.
    """

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        answer = ask_llm(question, context)

        print("\n" + "=" * 60)
        print("Question")
        print("=" * 60)
        print(question)

        print("\n" + "=" * 60)
        print("Answer")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()