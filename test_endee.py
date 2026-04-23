from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np

# 🔑 Your API token
client = Endee(token="fnqrjpe7:LbHSmuETDzepyMfkNaFCZb3yM7YWJBfp:as1")

# 🧠 Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "medical"
DIMENSION = 384

# 🧹 DELETE old index (fix duplicate data issue)
try:
    client.delete_index(INDEX_NAME)
    print("🗑️ Old index deleted")
except:
    print("ℹ️ No previous index found")

# ✅ Create fresh index
client.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    space_type="cosine",
    precision=Precision.INT8
)
print("✅ Fresh index created")

# 📂 Get index
index = client.get_index(INDEX_NAME)

# 📄 Sample data
docs = [
    "Diabetes is a chronic disease that affects blood sugar levels.",
    "Hypertension is high blood pressure and can lead to heart disease.",
    "Asthma affects the airways and makes breathing difficult."
]

# 🔁 Insert data + store vectors locally for scoring
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

index.upsert(vectors)
print("✅ Data inserted")

# 🔍 Query
query = "What is diabetes?"
query_vector = model.encode(query).tolist()

results = index.query(
    vector=query_vector,
    top_k=2
)

print("\n🔎 Retrieved Results with Scores:\n")

# 🧠 Cosine similarity function
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Print results + compute scores manually
retrieved_texts = []

for text, vec in stored_vectors:
    score = cosine_similarity(query_vector, vec)
    retrieved_texts.append(text)
    print(f"📄 {text}\n⭐ Score: {round(score, 4)}\n")

# 🧠 Build context (top relevant texts only)
context = " ".join(retrieved_texts[:2])

print("🧠 Context for LLM:\n")
print(context)

# 🤖 Strict prompt (no hallucination)
prompt = f"""
You are a medical assistant.

STRICT RULES:
- Answer ONLY using the provided context
- DO NOT add any extra knowledge
- DO NOT explain beyond context
- If answer is not fully in context, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

# 🤖 Generate answer using Ollama
response = ollama.chat(
    model="phi",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\n🤖 Final Answer:\n")
print(response["message"]["content"])