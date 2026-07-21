import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


# 🔹 Load prompt template from file
def load_prompt_template():
    with open("scripts/prompt_template.txt", "r", encoding="utf-8") as f:
        return f.read()


# 🔹 Ask LLM using context + prompt template
def ask_llm(question, context):

    prompt_template = load_prompt_template()

    # Fill placeholders
    final_prompt = prompt_template.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ]
    )

    return response.choices[0].message.content


def main():

    print("=" * 60)
    print("Day 14: Prompt Template Test")
    print("=" * 60)

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        # 🔴 TEMP CONTEXT (simulate retrieval)
        context = """
        Wheat should be sown from mid-October to mid-November.
        Rice requires standing water during most of its growth period.
        """

        answer = ask_llm(question, context)

        print("\n" + "=" * 60)
        print("Answer")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()