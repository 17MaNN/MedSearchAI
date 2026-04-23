from flask import Flask, request, jsonify, send_file
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np

app = Flask(__name__)

# 🔑 Init once (important)
client = Endee(token="")
model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "medical"

# 📂 Load index
index = client.get_index(INDEX_NAME)


# 🧠 Cosine similarity (optional scoring)
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 🌐 Serve frontend
@app.route("/")
def home():
    return send_file("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # 🔍 Embed query
    query_vector = model.encode(query).tolist()

    # 🔎 Search in Endee
    results = index.query(
        vector=query_vector,
        top_k=5
    )

    # 🧠 Extract context
    retrieved_texts = []

    if isinstance(results, list):
        for match in results:
            text = match.get("meta", {}).get("text", "")
            if text:
                retrieved_texts.append(text)
    else:
        for match in results.matches:
            text = getattr(match, "meta", {}).get("text", "")
            if text:
                retrieved_texts.append(text)

    context = " ".join(retrieved_texts[:3])

    # 🤖 Prompt
    prompt = f"""
You are a medical assistant.

STRICT RULES:
- Use ONLY the provided context
- DO NOT add extra knowledge
- If not found → say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    # 🤖 Call Ollama
    response = ollama.chat(
        model="phi",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    return jsonify({
        "query": query,
        "answer": answer,
        "context_used": retrieved_texts[:3]
    })


# 🚀 Run server
if __name__ == "__main__":
    app.run(debug=True)