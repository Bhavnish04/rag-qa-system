import streamlit as st
import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

PRELOADED_DIR = "data/manuals"  # bundled product manuals (.pdf and/or .txt)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Product Manual Assistant", layout="wide", page_icon="🗂️")

st.markdown("""
<style>
    .app-header {
        background: linear-gradient(135deg, #0058A3 0%, #00A3E0 50%, #FF6B35 100%);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
    }
    .app-header h1 {
        color: white !important;
        margin: 0 0 6px 0;
        font-size: 2.1rem;
    }
    .app-header p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0;
        font-size: 1.05rem;
    }
    .section-box {
        background: #1C2333;
        border: 1px solid #2E3648;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .section-box h4 {
        margin-top: 0;
        color: #FF9F6B;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #0058A3 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35, #FF9F6B);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 28px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #FF9F6B, #FF6B35);
        color: white;
    }
</style>
<div class="app-header">
    <h1>🗂️ Product Manual Q&A Assistant</h1>
    <p>Multilingual RAG demo — ask questions about product manuals in German, English or French</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load embedding model (cached)
# -------------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embed_model = load_embed_model()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_rerank = st.checkbox("Use Re-ranking", value=True)

    st.markdown("### Groq API Key")
    groq_api_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Paste your Groq API key here (free at console.groq.com/keys), or set it as the GROQ_API_KEY environment variable instead."
    )

    st.markdown("### Model Info")
    st.write("LLM: GPT-OSS 120B (Groq)")
    st.write("Embeddings: paraphrase-multilingual-MiniLM-L12-v2")

# -------------------------------
# Inputs
# -------------------------------
preloaded_paths = sorted(
    glob.glob(os.path.join(PRELOADED_DIR, "*.pdf")) + glob.glob(os.path.join(PRELOADED_DIR, "*.txt"))
)
preloaded_names = [os.path.basename(p) for p in preloaded_paths]

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("#### 📚 Pre-loaded manuals")
if preloaded_names:
    selected_preloaded = st.multiselect(
        "Select from the bundled manuals (no upload needed — instant, reliable for live demo)",
        options=preloaded_names,
        default=preloaded_names,
    )
else:
    selected_preloaded = []
    st.caption(f"No manuals found in `{PRELOADED_DIR}`. Add PDFs/txt there, or upload below instead.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("#### ➕ Or upload additional manual(s)")
uploaded_files = st.file_uploader(
    "Upload manual(s) (PDF)", accept_multiple_files=True
)
st.markdown('</div>', unsafe_allow_html=True)

query = st.text_input("💬 Ask a question about the manuals (German, English or French)")

# -------------------------------
# Run pipeline
# -------------------------------
if st.button("Get Answer"):

    if not uploaded_files and not selected_preloaded:
        st.warning("Select a pre-loaded manual or upload a PDF")
        st.stop()

    if not query.strip():
        st.warning("Enter a valid question")
        st.stop()

    docs = []

    # -------------------------------
    # Load pre-loaded manuals (selected)
    # -------------------------------
    for name in selected_preloaded:
        path = os.path.join(PRELOADED_DIR, name)
        if name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif name.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": name}))

    # -------------------------------
    # Load uploaded PDFs
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
        reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

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
    # Groq LLM
    # -------------------------------
    context = "\n\n".join(top_chunks)

    prompt = f"""
Answer ONLY using the context below.
If unsure, say "I don't know".
Always answer in the SAME language as the question (German question -> German answer,
English question -> English answer, French question -> French answer).

Context:
{context}

Question:
{query}

Answer in 2-3 sentences:
"""

    try:
        api_key = groq_api_key_input.strip() or os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar (or set GROQ_API_KEY).")
            st.stop()

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content

    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
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