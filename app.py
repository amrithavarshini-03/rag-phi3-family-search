from dataclasses import dataclass
from typing import List
import numpy as np
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ================== OLLAMA CONFIG ==================
OLLAMA_MODEL = "phi3:latest"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


# ================== DATA STRUCTURE ==================
@dataclass
class Document:
    id: str
    text: str
    source: str = ""


# ================== LOAD & CHUNK TEXT ==================
def load_text_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunk_size = 300
    overlap = 80   # IMPORTANT

    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append(content[start:end])
        start = end - overlap

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                id=f"text_chunk_{i}",
                text=chunk.strip(),
                source="family.txt"
            )
        )
    return docs



# ================== TF-IDF RAG ==================
class SimpleTfidfRAG:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.docs[idx], float(similarities[idx])))

        return results


# ================== PROMPT BUILDER ==================
def build_prompt(context: str, question: str) -> str:
    return f"""
You are a strict question-answering assistant.

Rules:
- Answer ONLY using the context below.
- Do NOT use outside knowledge.
- If the answer is not in the context, say:
  "I don't know based on the provided data."

Context:
{context}

Question:
{question}

Answer:
""".strip()


# ================== OLLAMA CALL ==================
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR calling Ollama] {e}"


# ================== MAIN ==================
def main():
    print("=== RAG using Text File with Phi-3 ===")

    docs = load_text_file("family.txt")
    rag = SimpleTfidfRAG(docs)

    while True:
        query = input("\nAsk about the family (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        # 1️⃣ RETRIEVE using plain query
        results = rag.retrieve(query)

        if not results:
            print("\nAnswer:")
            print("I don't know based on the provided data.")
            continue

        print("\nTop Retrieved Chunks (with similarity scores):")
        for i, (doc, score) in enumerate(results, start=1):
            print(f"\n[{i}] Score: {score:.3f}")
            print(doc.text[:200] + "...")

        # 2️⃣ BUILD CONTEXT
        context = "\n".join([doc.text for doc, _ in results])
        # 3️⃣ BUILD PROMPT
        prompt = build_prompt(context, query)
        
        # 4️⃣ GENERATE ANSWER
        answer = call_ollama(prompt)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
