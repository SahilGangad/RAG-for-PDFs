# ingest.py
import ollama
from pypdf import PdfReader
import chromadb

PDF_PATH = "data/slregression.pdf"   # change to your actual filename
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text"

def load_pdf_text(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
        embeddings.append(response["embedding"])
        if (i + 1) % 10 == 0:
            print(f"  embedded {i+1}/{len(chunks)}")
    return embeddings

def main():
    print("Loading PDF...")
    text = load_pdf_text(PDF_PATH)
    print(f"Extracted {len(text)} characters")

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print("Embedding chunks via Ollama (nomic-embed-text)...")
    embeddings = embed_chunks(chunks)

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="doc_chunks")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,   # already plain lists from ollama, no .tolist() needed
        documents=chunks
    )

    print(f"Done. {collection.count()} chunks stored in chroma_db/")

if __name__ == "__main__":
    main()