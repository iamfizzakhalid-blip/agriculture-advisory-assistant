import os
import time
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
        with open("scripts/prompt_template.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            "Prompt template not found: scripts/prompt_template.txt"
        )


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
        start_time = time.time()

        response = client.chat.completions.create(  # actual API call.
            model="llama-3.3-70b-versatile",
            temperature=0.1,  # Low temperature for grounded, deterministic answers
            max_tokens=500,   # Allow sufficient space for detailed answers
            messages=[
                {
                    "role": "system",
                    "content": system_instructions
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )

        elapsed_time = time.time() - start_time
        print(f"\nResponse Time: {elapsed_time:.2f} seconds")

        return response.choices[0].message.content

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