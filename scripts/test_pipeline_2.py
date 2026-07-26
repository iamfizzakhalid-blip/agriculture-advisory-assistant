# For Day 17, with 30 agriculture-related queries (6 for each crop) but before pipeline refinement

import csv
import time

from rag_pipeline import rag_pipeline
from load_db import (
    load_embedding_model,
    get_chroma_client,
    get_or_create_collection,
    CHROMA_DIR,
)

model = load_embedding_model()

client = get_chroma_client(CHROMA_DIR)

collection = get_or_create_collection(client)

questions = [
    # Wheat
    "What is the best time to sow wheat?",
    "How much irrigation does wheat require?",
    "Which fertilizer should be used for wheat?",
    "How can weeds in wheat be controlled?",
    "What are the common diseases affecting wheat?",
    "What is the recommended seed rate for wheat?",

    # Rice
    "When should rice be transplanted?",
    "Which fertilizer is recommended for rice?",
    "How often should rice be irrigated?",
    "What are the common pests of rice?",
    "How can rice diseases be prevented?",
    "What is the recommended seed rate for rice?",

    # Cotton
    "What is the recommended sowing time for cotton?",
    "How can cotton pests be controlled?",
    "Which fertilizer should be applied to cotton?",
    "How often should cotton be irrigated?",
    "What are the common diseases affecting cotton?",
    "How can weeds in cotton fields be managed?",

    # Sugarcane
    "When should sugarcane be planted?",
    "What fertilizer should be used for sugarcane?",
    "How often should sugarcane be irrigated?",
    "What are the common pests of sugarcane?",
    "How can sugarcane diseases be controlled?",
    "What is the recommended spacing for sugarcane?",

    # Maize
    "When should maize be planted?",
    "Which fertilizer is best for maize?",
    "How can weeds in maize be controlled?",
    "How much irrigation does maize require?",
    "What are the common diseases affecting maize?",
    "What is the recommended seed rate for maize?",
]

with open(
    "logs/test_log_2.csv",
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Question",
        "Crop",
        "Source",
        "Filename",
        "Distance",
        "Answer",
        "Response Time"
    ])

    for question in questions:

        start = time.time()

        results, answer = rag_pipeline(
            question,
            collection,
            model
        )

        response_time = time.time() - start

        if results is None:
            writer.writerow([
                question,
                "",
                "",
                "",
                "",
                answer,
                round(response_time, 2)
            ])
            continue

        top = results[0]

        metadata = top["metadata"]

        crop = metadata.get("crop")

        source = metadata.get("source")

        filename = metadata.get("filename")

        distance = round(top["distance"], 4)

        writer.writerow([
            question,
            crop,
            source,
            filename,
            distance,
            answer,
            round(response_time, 2)
        ])
        print(f"Completed: {question}")

