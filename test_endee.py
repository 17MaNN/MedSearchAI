from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np
import fitz 

client = Endee(token="fnqrjpe7:LbHSmuETDzepyMfkNaFCZb3yM7YWJBfp:as1")

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "medical"
DIMENSION = 384

try:
    client.delete_index(INDEX_NAME)
    print(" Old index deleted")
except:
    print(" No previous index")

client.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    space_type="cosine",
    precision=Precision.INT8
)
print(" Fresh index created")

index = client.get_index(INDEX_NAME)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

pdf_path = "medical.pdf"
raw_text = extract_text_from_pdf(pdf_path)
docs = chunk_text(raw_text)

print(f" Chunks created: {len(docs)}")

vectors = []
stored_vectors = []

for i, text in enumerate(docs):
    vector = model.encode(text).tolist()

    vectors.append({
        "id": str(i),
        "vector": vector,
        "meta": {"text": text}
    })

    stored_vectors.append((text, vector))

for i in range(0, len(vectors), 50):
    index.upsert(vectors[i:i+50])

print(" PDF data inserted")

query = "What is diabetes?"
query_vector = model.encode(query).tolist()

results = index.query(
    vector=query_vector,
    top_k=3
)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n Retrieved Results:\n")

retrieved_texts = []

for text, vec in stored_vectors:
    score = cosine_similarity(query_vector, vec)
    retrieved_texts.append((text, score))

retrieved_texts = sorted(retrieved_texts, key=lambda x: x[1], reverse=True)

top_texts = [t[0] for t in retrieved_texts[:3]]

for t, s in retrieved_texts[:3]:
    print(f" {t}\n Score: {round(s,4)}\n")

context = " ".join([t[0] for t in retrieved_texts[:2]])

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

response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

print("\n Final Answer:\n")
print(response["message"]["content"])