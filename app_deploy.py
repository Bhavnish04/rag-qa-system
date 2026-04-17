import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.llms import HuggingFaceHub
import chromadb

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📄 RAG Document QA System")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    use_rerank = st.checkbox("Use Re-ranking", value=True)

    st.markdown("### Model Info")
    st.write("LLM: FLAN-T5 (HuggingFace)")
    st.write("Embeddings: all-MiniLM-L6-v2")

# -------------------------------
# Upload + Query
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF(s)", accept_multiple_files=True
)

query = st.text_input("Ask a question about your documents")

# -------------------------------
# Run pipeline
# -------------------------------
if st.button("Get Answer"):

    # -------------------------------
    # Edge cases
    # -------------------------------
    if not uploaded_files:
        st.warning("Upload at least one PDF")
        st.stop()

    if not query.strip():
        st.warning("Enter a valid question")
        st.stop()

    docs = []

    # -------------------------------
    # Load PDFs
    # -------------------------------
    for file in uploaded_files:
        filepath = "temp.pdf"
        with open(filepath, "wb") as f:
            f.write(file.read())

        loader = PyPDFLoader(filepath)
        docs.extend(loader.load())

    # -------------------------------
    # Chunking
    # -------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    st.sidebar.write(f"Chunks created: {len(chunks)}")

    # -------------------------------
    # Embeddings + Vector DB
    # -------------------------------
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    client = chromadb.Client()
    collection = client.create_collection("rag_docs")

    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk.page_content).tolist()

        collection.add(
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[str(i)],
            embeddings=[embedding]
        )

    # -------------------------------
    # Query embedding
    # -------------------------------
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )

    # -------------------------------
    # Filtering
    # -------------------------------
    retrieved_chunks = [
        doc for doc in results['documents'][0]
        if (
            len(doc.strip()) > 120
            and "." in doc
            and "literature" not in doc.lower()
        )
    ]

    if not retrieved_chunks:
        st.error("No relevant content found")
        st.stop()

    # -------------------------------
    # Re-ranking
    # -------------------------------
    confidence = None

    if use_rerank:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = reranker.predict(pairs)

        scored_chunks = list(zip(retrieved_chunks, scores))
        scored_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)

        top_chunks = [chunk for chunk, score in scored_chunks[:3]]
        confidence = max(scores)

    else:
        top_chunks = retrieved_chunks[:3]

    # -------------------------------
    # LLM (FIXED)
    # -------------------------------
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if hf_token is None:
        st.error("HuggingFace API token not found in secrets")
        st.stop()

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    # -------------------------------
    # Generate answer
    # -------------------------------
    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a precise AI assistant.

Summarize clearly using ONLY the context.
If unsure, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("📌 Answer")
    st.write(answer)

    if confidence is not None:
        if confidence > -5:
            st.success(f"High confidence ({confidence:.2f})")
        elif confidence > -8:
            st.warning(f"Medium confidence ({confidence:.2f})")
        else:
            st.error(f"Low confidence ({confidence:.2f})")

    st.subheader("🔍 Top Retrieved Chunks")

    for i, chunk in enumerate(top_chunks):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(chunk[:300])
        st.write("---")