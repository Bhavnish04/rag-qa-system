# Multilingual RAG for Product Manuals

Ask questions about PDF product manuals in **German, English or French** and get answers taken only from the documents, with the source passages shown and a confidence level for each answer.

Built as a case study for technical product documentation (e.g. warehouse equipment manuals), where the person asking often doesn't speak the language the manual is written in.

![App screenshot](docs/Screenshot%202026-08-18%20203601.png)

## Example

**Question (English):** *What safety distance should be kept from people when transporting or lifting goods with the electric stacker?*

**Answer:**
> Beim Transportieren oder Anheben von Waren muss ein Sicherheitsabstand von 600 mm zu Personen eingehalten werden.
> When transporting or lifting goods, a safety distance of 600 mm from people must be maintained.

## How it works

1. **Load:** PDFs are loaded (bundled manuals or your own uploads) and split into chunks.
2. **Embed:** each chunk is embedded with `paraphrase-multilingual-MiniLM-L12-v2`, so a question in one language can match text in another.
3. **Retrieve:** the most similar chunks are found by vector search.
4. **Re-rank:** a cross-encoder reads the question and each candidate chunk together and re-orders them. This is slower but more precise. It can be switched on/off in the sidebar to compare.
5. **Answer:** an LLM (GPT-OSS 120B via Groq) writes the answer using only the retrieved chunks.
6. **Confidence:** each answer shows a confidence level based on the retrieval score, so you can tell when the system found something solid and when it may be guessing.

```
PDF manuals ─► chunks ─► multilingual embeddings ─► vector search (top-k)
                                                        │
                                                        ▼
                                       cross-encoder re-ranking (optional)
                                                        │
                                                        ▼
                                LLM answer from retrieved chunks + confidence
```

## Tech stack

| Part | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Re-ranking | Multilingual cross-encoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Vector store | ChromaDB (in-memory) |
| LLM | GPT-OSS 120B via Groq API |
| PDF parsing / chunking | LangChain |

An earlier version of this project ran fully locally with Mistral via Ollama and an English-only embedding model (`all-MiniLM-L6-v2`).

## Run it

```bash
git clone https://github.com/Bhavnish04/rag-qa-system.git
cd rag-qa-system
pip install -r requirements.txt
streamlit run app_groq.py
```

Enter your Groq API key in the sidebar (get one at console.groq.com), or set it as the `GROQ_API_KEY` environment variable. Then upload your own PDFs and ask a question.

To get the pre-loaded selector, put your own manuals (`.pdf` or `.txt`) into `data/manuals/`. The manuals used in the demo are not included in this repo.

## Evaluation

`eval.py` measures retrieval quality with and without re-ranking:

- **Recall@5:** how often the correct passage appears in the top 5 results
- **Precision@5:** how many of the top 5 results are relevant
- **MRR:** how high the first correct passage is ranked

| Setup | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Without re-ranking | – | – | – |
| With re-ranking | – | – | – |

*Results to be added after running `eval.py` on the bundled manuals.*

## Limitations

- Tested on a small set of 3 manuals; results on large document collections may differ.
- Answers depend on PDF text extraction; scanned PDFs without a text layer won't work.
- Requires a Groq API key for answer generation.

## Author

Bhavnish Singhall · M.Sc. Computational Linguistics, University of Stuttgart
