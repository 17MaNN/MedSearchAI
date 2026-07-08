import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from endee import Endee
from huggingface_hub import InferenceClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file (never committed)

app = Flask(__name__)

ENDEE_TOKEN = os.environ.get("ENDEE_TOKEN")
if not ENDEE_TOKEN:
    raise RuntimeError(
        "ENDEE_TOKEN is not set. Set it as an environment variable "
        "(locally: in .env, on Render: in the dashboard's Environment tab)."
    )

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
        "and set it as an environment variable."
    )

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. Get a free token (read access is enough) at "
        "https://huggingface.co/settings/tokens and set it as an environment variable."
    )

client = Endee(token=ENDEE_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

COLLECTION_NAME = "medical"
VECTOR_FIELD = "embedding"  # must match the field name used in test_endee.py
GROQ_MODEL = "llama-3.1-8b-instant"  # fast + generous free tier
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same model used to ingest

collection = client.get_collection(COLLECTION_NAME)


def embed(text: str) -> list:
    """Get a single 384-dim sentence embedding via HF's hosted Inference API
    instead of loading sentence-transformers/torch locally (too heavy for
    Render's 512MB free tier)."""
    result = hf_client.feature_extraction(text, model=EMBEDDING_MODEL)
    arr = np.array(result)
    if arr.ndim == 2:
        # Some models return per-token embeddings -> mean-pool to one vector
        arr = arr.mean(axis=0)
    return arr.tolist()


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_vector = embed(query)

    results = collection.search(
        fields={VECTOR_FIELD: {"query": query_vector, "limit": 3}}
    )
    hits = results["results"][VECTOR_FIELD]

    retrieved_texts = []
    for hit in hits:
        text = hit.get("meta", {}).get("text", "")
        if text:
            retrieved_texts.append(text)

    context = " ".join(retrieved_texts[:2])

    prompt = f"""You are a medical assistant. Answer in 2-3 sentences maximum.

STRICT RULES:
- Use ONLY the provided context
- DO NOT add extra knowledge
- If not found in context, say "I don't know"
- Be concise. No preamble, no repeating the question, no filler.

Context:
{context}

Question:
{query}

Answer (2-3 sentences max):"""

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=150,
    )

    answer = completion.choices[0].message.content

    return jsonify({
        "query": query,
        "answer": answer,
        "context_used": retrieved_texts[:2]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)