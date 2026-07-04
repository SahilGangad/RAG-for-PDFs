# retrieve.py
import chromadb
from ingest import get_embed_model

TOP_K = 3
DEFAULT_COLLECTION = "doc_chunks"

def get_query_embedding(query):
    model = get_embed_model()
    return model.encode([query])[0].tolist()

def retrieve(query, top_k=TOP_K, collection_name=DEFAULT_COLLECTION):
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name=collection_name)

    query_embedding = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0], results["distances"][0]

def main():
    query = input("Ask a question about your document: ")
    chunks, distances = retrieve(query)
    print(f"\nTop {len(chunks)} matches:\n")
    for i, (chunk, dist) in enumerate(zip(chunks, distances)):
        print(f"--- Match {i+1} (distance: {dist:.4f}) ---")
        print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
        print()

if __name__ == "__main__":
    main()