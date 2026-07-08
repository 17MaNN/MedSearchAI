import os
from endee import Endee
from sentence_transformers import SentenceTransformer
import fitz
from dotenv import load_dotenv

load_dotenv()

ENDEE_TOKEN = os.environ.get("ENDEE_TOKEN")
if not ENDEE_TOKEN:
    raise RuntimeError(
        "ENDEE_TOKEN is not set. Create a .env file (see .env.example) "
        "with ENDEE_TOKEN=your_token_here"
    )

client = Endee(token=ENDEE_TOKEN)

COLLECTION_NAME = "medical"
DIMENSION = 384
VECTOR_FIELD = "embedding"  # name of the dense-vector field in this collection

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Create (or reuse) the collection ---
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Using existing collection '{COLLECTION_NAME}'")
except Exception:
    client.create_collection(
        name=COLLECTION_NAME,
        fields=[
            {
                "name": VECTOR_FIELD,
                "type": "vector",
                "params": {
                    "dimension": DIMENSION,
                    "space_type": "cosine",
                    "precision": "int8",
                },
            }
        ],
    )
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Created new collection '{COLLECTION_NAME}'")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


pdf_path = "medical.pdf"
raw_text = extract_text_from_pdf(pdf_path)
docs = chunk_text(raw_text)
print(f"Chunks created: {len(docs)}")

# --- Upsert in batches ---
objects = []
for i, text in enumerate(docs):
    vector = model.encode(text).tolist()
    objects.append({
        "id": str(i),
        "meta": {"text": text},
        "fields": {VECTOR_FIELD: vector},
    })

BATCH_SIZE = 50
for i in range(0, len(objects), BATCH_SIZE):
    collection.upsert(objects[i:i + BATCH_SIZE])

print("PDF data inserted")

# --- Query ---
query = "What is diabetes?"
query_vector = model.encode(query).tolist()

results = collection.search(
    fields={VECTOR_FIELD: {"query": query_vector, "limit": 3}}
)

hits = results["results"][VECTOR_FIELD]

print("\nRetrieved Results:\n")
for hit in hits:
    text = hit.get("meta", {}).get("text", "")
    score = hit.get("similarity")
    print(f"{text}\nScore: {score}\n")

context = " ".join(hit.get("meta", {}).get("text", "") for hit in hits[:2])
print("Context:\n", context)

prompt = f"""
You are a professional medical assistant.

Rules:
- Use ONLY the given context
- Do NOT add external knowledge
- If answer is not present, say "I don't know"
- Answer clearly in 2-4 lines

Context:
{context}

Question:
{query}

Answer:
"""

import ollama

response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

print("\nFinal Answer:\n")
print(response["message"]["content"])