from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
import ollama

# 🔑 Your API token
client = Endee(token="fnqrjpe7:LbHSmuETDzepyMfkNaFCZb3yM7YWJBfp:as1")

# 🧠 Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "medical"
DIMENSION = 384

# ✅ Create index safely
try:
    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
        precision=Precision.INT8
    )
    print("✅ Index created")
except Exception as e:
    print("⚠️ Index may already exist:", e)

# 📂 Get index
index = client.get_index(INDEX_NAME)

# 📄 Sample data (replace with real dataset later)
docs = [
    "Diabetes is a chronic disease that affects blood sugar levels.",
    "Hypertension is high blood pressure and can lead to heart disease.",
    "Asthma affects the airways and makes breathing difficult."
]

# 🔁 Insert data
vectors = []
for i, text in enumerate(docs):
    vector = model.encode(text).tolist()
    vectors.append({
        "id": str(i),
        "vector": vector,
        "meta": {"text": text}
    })

index.upsert(vectors)
print("✅ Data inserted")

# 🔍 Query
query = "What is diabetes?"
query_vector = model.encode(query).tolist()

results = index.query(
    vector=query_vector,
    top_k=2
)

print("\n🔎 Retrieved Results:\n")

# ✅ Handle BOTH response types
retrieved_texts = []

if isinstance(results, list):
    for match in results:
        text = match.get("meta", {}).get("text", "No text")
        score = match.get("score", "N/A")
        retrieved_texts.append(text)
        print(f"📄 {text}\n⭐ Score: {score}\n")
else:
    for match in results.matches:
        text = getattr(match, "meta", {}).get("text", "No text")
        score = getattr(match, "score", "N/A")
        retrieved_texts.append(text)
        print(f"📄 {text}\n⭐ Score: {score}\n")

# 🧠 Build context
context = " ".join(retrieved_texts)

print("🧠 Context for LLM:\n")
print(context)

# 🤖 Generate answer using Ollama (phi model)
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

response = ollama.chat(
    model="phi",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\n🤖 Final Answer:\n")
print(response["message"]["content"])