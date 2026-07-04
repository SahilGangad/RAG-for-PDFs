# app.py
import streamlit as st
import os
import hashlib
from ingest import load_pdf_text, chunk_text, embed_chunks
from generate import generate_answer
import chromadb

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Document Q&A")
st.caption("Upload a PDF, then ask questions about it — powered by local RAG (ChromaDB + Ollama + Llama 3)")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()[:16]  # short hash, plenty unique for this use

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getbuffer()
    content_hash = file_hash(file_bytes)
    collection_name = f"doc_{content_hash}"

    if st.session_state.get("current_hash") != content_hash:
        client = chromadb.PersistentClient(path="chroma_db")

        existing_collections = [c.name for c in client.list_collections()]

        if collection_name in existing_collections:
            # Already embedded before (this session or a past one) — skip straight to using it
            st.session_state.current_hash = content_hash
            st.session_state.current_file = uploaded_file.name
            st.session_state.collection_name = collection_name
            st.session_state.history = []
            st.success(f"Using cached index for {uploaded_file.name} (already embedded previously)")
        else:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            with st.spinner("Reading and indexing document..."):
                text = load_pdf_text(file_path)
                chunks = chunk_text(text)
                embeddings = embed_chunks(chunks)

                collection = client.get_or_create_collection(name=collection_name)
                ids = [f"chunk_{i}" for i in range(len(chunks))]
                collection.add(ids=ids, embeddings=embeddings, documents=chunks)

            st.session_state.current_hash = content_hash
            st.session_state.current_file = uploaded_file.name
            st.session_state.collection_name = collection_name
            st.session_state.history = []
            st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

# --- Chat ---
if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.get("current_hash"):
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            with st.expander("Sources"):
                for i, dist in enumerate(entry["distances"]):
                    st.caption(f"Chunk {i+1} — distance: {dist:.2f}")

    query = st.chat_input("Ask a question about your document...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, chunks, distances = generate_answer(
                    query, collection_name=st.session_state.collection_name
                )
            st.write(answer)
            with st.expander("Sources"):
                for i, dist in enumerate(distances):
                    st.caption(f"Chunk {i+1} — distance: {dist:.2f}")

        st.session_state.history.append({
            "question": query, "answer": answer, "distances": distances
        })
else:
    st.info("Upload a PDF above to get started.")