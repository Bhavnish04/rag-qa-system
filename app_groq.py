import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📄 RAG Document QA System")

# -------------------------------
# Load embedding model (cached)
# -------------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embed_model()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_rerank = st.checkbox("Use Re-ranking", value=True)

    st.markdown("### Model Info")
    st.write("LLM: LLaMA3 (Groq)")
    st.write("Embeddings: all-MiniLM-L6-v2")

# -------------------------------
# Inputs
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF(s)", accept_multiple_files=True
)

query = st.text_input("Ask a question about your documents")

# -------------------------------
# Run pipeline
# -------------------------------
if st.button("Get Answer"):

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
        filepath = f"temp_{file.name}"

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

    if not chunks:
        st.error("No content extracted from PDFs")
        st.stop()

    # -------------------------------
    # ChromaDB (reset each run)
    # -------------------------------
    client_db = chromadb.Client()

    try:
        client_db.delete_collection("rag_docs")
    except:
        pass

    collection = client_db.create_collection("rag_docs")

    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk.page_content).tolist()

        collection.add(
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[f"{i}_{hash(chunk.page_content)}"],
            embeddings=[embedding]
        )

    # -------------------------------
    # Query
    # -------------------------------
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )

    retrieved_chunks = [
        doc for doc in results['documents'][0]
        if len(doc.strip()) > 50
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

    if not top_chunks:
        st.error("No chunks after processing")
        st.stop()

    # -------------------------------
    # Groq LLM (FINAL)
    # -------------------------------
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("Groq API key not found. Add it in Streamlit Secrets.")
        st.stop()

    client = Groq(api_key=groq_api_key)

    context = "\n\n".join(top_chunks)

    prompt = f"""
Answer ONLY using the context below.
If unsure, say "I don't know".

Context:
{context}

Question:
{query}

Answer in 2-3 sentences:
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.2
        )

        answer = response.choices[0].message.content

    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        st.stop()

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("📌 Answer")
    st.success(answer)

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