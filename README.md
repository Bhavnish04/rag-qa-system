RAG Document QA System
An end-to-end Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents. Built with LangChain, ChromaDB, SBERT, and Mistral (via Ollama).
Features

Upload single or multiple PDFs
Semantic search using SBERT embeddings (all-MiniLM-L6-v2)
Vector storage and retrieval with ChromaDB
Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) for improved Recall@3
Local LLM inference via Mistral (Ollama) — no API key needed
Retrieval confidence score (sigmoid-normalised)
Toggle re-ranking on/off from the sidebar

