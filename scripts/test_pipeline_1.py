# For Day 16, with only 10 agriculture-related queries (2 for each crop)

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

    # Rice
    "When should rice be transplanted?",
    "Which fertilizer is best for rice?",

    # Cotton
    "How can cotton pests be controlled?",
    "What is the recommended sowing time for cotton?",

    # Sugarcane
    "How often should sugarcane be irrigated?",
    "What fertilizer should be used for sugarcane?",

    # Maize
    "When should maize be planted?",
    "How can weeds in maize be controlled?"
]

with open(
    "logs/test_log_1.csv",
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

