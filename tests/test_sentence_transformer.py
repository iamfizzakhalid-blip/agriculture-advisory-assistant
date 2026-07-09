from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Rice needs standing water during early growth."

embedding = model.encode(text)

print("Sentence:", text)
print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])