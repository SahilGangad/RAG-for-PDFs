# generate.py
import os
from dotenv import load_dotenv
from retrieve import retrieve, DEFAULT_COLLECTION

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "groq"
GEN_MODEL_OLLAMA = "llama3"
GEN_MODEL_GROQ = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # Groq's hosted Llama 3 8B — matches your local model size

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- Answer strictly based on the context below. Do not use outside knowledge.
- Only use a piece of context if it DIRECTLY addresses the question being asked. A chunk that merely shares a keyword or topic with the question, without actually answering it, must be ignored for that part of the question.
- Never combine fragments from different parts of the context into a sentence that doesn't appear as such in the source. Do not present anything as a quote unless it is copied exactly.
- If the context does not contain enough information to answer the question, respond exactly: "I don't know based on the provided document."
- If the context only partially answers the question, answer only the part that is directly supported, and explicitly state which part is not covered by the document.
- Be concise and direct."""

def build_prompt(query, chunks):
    context = "\n\n---\n\n".join(chunks)
    return f"""Context:
{context}

Question: {query}

Answer:"""

def call_ollama(messages):
    import ollama
    response = ollama.chat(
        model=GEN_MODEL_OLLAMA,
        messages=messages,
        options={"temperature": 0, "seed": 42}
    )
    return response["message"]["content"]

def call_groq(messages):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GEN_MODEL_GROQ,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def generate_answer(query, top_k=3, collection_name=DEFAULT_COLLECTION):
    chunks, distances = retrieve(query, top_k=top_k, collection_name=collection_name)
    prompt = build_prompt(query, chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    if LLM_PROVIDER == "groq":
        answer = call_groq(messages)
    else:
        answer = call_ollama(messages)

    return answer, chunks, distances

def main():
    query = input("Ask a question about your document: ")
    answer, chunks, distances = generate_answer(query)
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Retrieved chunks ---")
    for i, (chunk, dist) in enumerate(zip(chunks, distances)):
        print(f"\nChunk {i+1} (distance {dist:.2f}):")
        print(chunk[:200])

if __name__ == "__main__":
    main()