from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
# Step 1: Load PDF
loader = PyPDFLoader("data/DL-Slides.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")



from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding model loaded")

# Test embedding on one chunk
sample_text = chunks[0].page_content
embedding = embed_model.encode(sample_text)

print(f"Embedding size: {len(embedding)}")


import chromadb
from sentence_transformers import CrossEncoder

# -------------------------------
# Create DB + store embeddings
# -------------------------------
client = chromadb.Client()
collection = client.create_collection(name="rag_docs")

print("ChromaDB collection created")

for i, chunk in enumerate(chunks):
    embedding = embed_model.encode(chunk.page_content).tolist()
    
    collection.add(
        documents=[chunk.page_content],
        metadatas=[chunk.metadata],
        ids=[str(i)],
        embeddings=[embedding]
    )

print("All chunks stored in ChromaDB")

# -------------------------------
# Query
# -------------------------------
query = "Summarize the main topic of this document in 2 sentences"

query_embedding = embed_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=8
)

# -------------------------------
# Step 1: Filter chunks
# -------------------------------
retrieved_chunks = [
    doc for doc in results['documents'][0]
    if (
        len(doc.strip()) > 120
        and "." in doc
        and not doc.strip().lower().startswith(("0-", "1.", "2.", "3."))
        and "literature" not in doc.lower()
        and "chapter" not in doc.lower()
    )
]

# -------------------------------
# Step 2: Re-ranking
# -------------------------------
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [(query, chunk) for chunk in retrieved_chunks]
scores = reranker.predict(pairs)

scored_chunks = list(zip(retrieved_chunks, scores))
scored_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)

# -------------------------------
# Step 3: Top chunks
# -------------------------------
top_chunks = [chunk for chunk, score in scored_chunks[:3]]

# -------------------------------
# Debug: Show reranked results
# -------------------------------
print("\nTop chunks AFTER reranking:\n")

for i, (chunk, score) in enumerate(scored_chunks[:3]):
    print(f"Rank {i+1} | Score: {score:.4f}")
    print(chunk[:200])
    print("\n---\n")

# -------------------------------
# (Optional) Context for LLM
# -------------------------------
context = "\n\n".join(top_chunks)
    
#retrieval
from langchain_ollama import OllamaLLM

# Load LLM
llm = OllamaLLM(model="mistral")

# Combine retrieved chunks into context
filtered_chunks = [
    doc for doc in results['documents'][0]
    if len(doc.strip()) > 50
]

context = "\n\n".join(filtered_chunks)

# Prompt
prompt = f"""
You are a precise AI assistant.

Summarize the document clearly in 2–3 sentences.
Focus only on the main topic and key areas.

Use ONLY the context.
Do not include minor sections or lists.

Context:
{context}

Question:
{query}
"""

# Generate response
response = llm.invoke(prompt)

print("\nFinal Answer:\n")
print(response)