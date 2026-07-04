# ingest.py
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

PDF_PATH = "data/slregression.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
def get_embed_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

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
    model = get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings.tolist()   # sentence-transformers returns numpy arrays — ChromaDB wants plain lists

def main():
    print("Loading PDF...")
    text = load_pdf_text(PDF_PATH)
    print(f"Extracted {len(text)} characters")

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print("Embedding chunks (sentence-transformers)...")
    embeddings = embed_chunks(chunks)

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="doc_chunks")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)

    print(f"Done. {collection.count()} chunks stored in chroma_db/")

if __name__ == "__main__":
    main()