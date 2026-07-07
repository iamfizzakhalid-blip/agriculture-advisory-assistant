import os
from PyPDF2 import PdfReader

input_folder = "data/raw/wheat"
output_folder = "data/clean/wheat"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(input_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, file)
        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        output_file = os.path.join(output_folder, file.replace(".pdf", ".txt"))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Processed: {file}")

print("All wheat documents cleaned!")