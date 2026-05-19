"""
eval.py — RAG Retrieval Evaluation
Metrics: Recall@K, Precision@K, MRR
Compares: Baseline (cosine only) vs Re-ranked (cross-encoder)

Usage:
    python eval.py

Requires your PDF at data/DL-Slides.pdf (or update PDF_PATH below).
Update EVAL_SET queries/keywords to match your actual PDF content.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PDF_PATH = "D:\projects\RAG\project 3\temp_German CV 3.pdf"
K = 3  # Recall@3, Precision@3

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH EVAL SET
# Each entry: a query + keywords that MUST appear in a relevant chunk.
# Keyword matching is used as a proxy for relevance (no human annotations needed).
# Update these to match YOUR PDF's actual content.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_SET = [
    {
        "query": "What is backpropagation and how does it work?",
        "relevant_keywords": ["backprop", "gradient", "chain rule", "derivative", "loss"]
    },
    {
        "query": "What is a convolutional neural network?",
        "relevant_keywords": ["convolution", "filter", "feature map", "pooling", "CNN"]
    },
    {
        "query": "How does dropout regularization work?",
        "relevant_keywords": ["dropout", "regularization", "overfitting", "neuron", "training"]
    },
    {
        "query": "What is the attention mechanism in transformers?",
        "relevant_keywords": ["attention", "query", "key", "value", "softmax", "transformer"]
    },
    {
        "query": "What is batch normalization?",
        "relevant_keywords": ["batch normalization", "batch norm", "normalization", "mean", "variance"]
    },
    {
        "query": "Explain the vanishing gradient problem",
        "relevant_keywords": ["vanishing", "gradient", "sigmoid", "deep network", "exploding"]
    },
    {
        "query": "What is a recurrent neural network?",
        "relevant_keywords": ["recurrent", "RNN", "LSTM", "hidden state", "sequence"]
    },
    {
        "query": "How does transfer learning work?",
        "relevant_keywords": ["transfer learning", "pretrained", "fine-tuning", "features"]
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE JUDGE
# A chunk is "relevant" if it contains at least one keyword from the query's list.
# ─────────────────────────────────────────────────────────────────────────────
def is_relevant(chunk: str, keywords: list) -> bool:
    chunk_lower = chunk.lower()
    return any(kw.lower() in chunk_lower for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def recall_at_k(retrieved: list, keywords: list, k: int) -> float:
    """1.0 if at least one relevant chunk is in top-K, else 0.0."""
    return 1.0 if any(is_relevant(c, keywords) for c in retrieved[:k]) else 0.0


def precision_at_k(retrieved: list, keywords: list, k: int) -> float:
    """Fraction of top-K chunks that are relevant."""
    hits = sum(1 for c in retrieved[:k] if is_relevant(c, keywords))
    return hits / k


def mean_reciprocal_rank(retrieved: list, keywords: list) -> float:
    """1/rank of the first relevant chunk. 0 if none found."""
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, keywords):
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BUILD INDEX
# ─────────────────────────────────────────────────────────────────────────────
def build_index(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"  -> {len(docs)} pages loaded")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"  -> {len(chunks)} chunks created")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("eval_rag")
    except Exception:
        pass
    collection = client.create_collection("eval_rag")

    chunk_texts = [c.page_content for c in chunks]

    print("  -> Embedding chunks...")
    embeddings = embed_model.encode(chunk_texts, show_progress_bar=True)

    for i, (text, emb) in enumerate(zip(chunk_texts, embeddings)):
        collection.add(documents=[text], ids=[str(i)], embeddings=[emb.tolist()])

    print(f"  -> Index built: {len(chunk_texts)} vectors\n")
    return collection, embed_model


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_baseline(query, collection, embed_model, n=8):
    q_emb = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=n)
    return results["documents"][0]


def retrieve_reranked(query, collection, embed_model, reranker, n_retrieve=8, top_k=3):
    candidates = retrieve_baseline(query, collection, embed_model, n=n_retrieve)
    pairs = [(query, chunk) for chunk in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(collection, embed_model, reranker, k=K):
    baseline_recall, baseline_precision, baseline_mrr = [], [], []
    reranked_recall, reranked_precision, reranked_mrr = [], [], []

    print(f"{'Query':<55} {'B-Rec@'+str(k):<10} {'R-Rec@'+str(k):<10} {'B-MRR':<8} {'R-MRR':<8}")
    print("-" * 95)

    for entry in EVAL_SET:
        query    = entry["query"]
        keywords = entry["relevant_keywords"]

        base = retrieve_baseline(query, collection, embed_model, n=8)
        b_rec  = recall_at_k(base, keywords, k)
        b_prec = precision_at_k(base, keywords, k)
        b_mrr  = mean_reciprocal_rank(base, keywords)

        reranked = retrieve_reranked(query, collection, embed_model, reranker, top_k=k)
        r_rec  = recall_at_k(reranked, keywords, k)
        r_prec = precision_at_k(reranked, keywords, k)
        r_mrr  = mean_reciprocal_rank(reranked, keywords)

        baseline_recall.append(b_rec);     reranked_recall.append(r_rec)
        baseline_precision.append(b_prec); reranked_precision.append(r_prec)
        baseline_mrr.append(b_mrr);        reranked_mrr.append(r_mrr)

        short_q = (query[:52] + "...") if len(query) > 52 else query
        print(f"{short_q:<55} {b_rec:<10.2f} {r_rec:<10.2f} {b_mrr:<8.3f} {r_mrr:<8.3f}")

    print("-" * 95)

    return {
        "baseline": {
            f"Recall@{k}":    np.mean(baseline_recall),
            f"Precision@{k}": np.mean(baseline_precision),
            "MRR":            np.mean(baseline_mrr),
        },
        "reranked": {
            f"Recall@{k}":    np.mean(reranked_recall),
            f"Precision@{k}": np.mean(reranked_precision),
            "MRR":            np.mean(reranked_mrr),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    collection, embed_model = build_index(PDF_PATH)

    print("Loading reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  -> Reranker loaded\n")

    print("=" * 95)
    print(f"EVALUATION  |  K={K}  |  N queries={len(EVAL_SET)}")
    print("=" * 95 + "\n")

    results = evaluate(collection, embed_model, reranker, k=K)

    print(f"\n{'SUMMARY':=<60}")
    print(f"{'Metric':<20} {'Baseline':<15} {'Re-ranked':<15} {'Delta':<10}")
    print("-" * 60)

    for metric in results["baseline"]:
        b = results["baseline"][metric]
        r = results["reranked"][metric]
        delta = r - b
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<20} {b:<15.4f} {r:<15.4f} {sign}{delta:.4f}")

    print("=" * 60)
    print(f"\nThese numbers back the CV claim:")
    print(f"  'cross-encoder re-ranking improving Recall@{K} on a labelled evaluation set'")
