# Document Q&A with RAG

A Retrieval-Augmented Generation (RAG) app that answers natural-language questions from uploaded PDF documents, grounded in the document's actual content to reduce hallucination.

## How it works

1. **Extract** — pulls text out of an uploaded PDF
2. **Chunk** — splits the text into smaller overlapping pieces
3. **Embed** — converts each chunk into a vector using a sentence-transformer model
4. **Store & Retrieve** — stores vectors in ChromaDB and retrieves the most relevant chunks for a given question using semantic similarity search
5. **Generate** — sends the retrieved chunks + question to an LLM (Google Gemini API) with an engineered prompt, which generates a grounded answer
6. **Display** — shows the answer along with the source passages used, for transparency

## Tech stack

- Python
- Streamlit (UI)
- pypdf (PDF text extraction)
- LangChain text splitters (chunking)
- sentence-transformers (embeddings)
- ChromaDB (vector storage & retrieval)
- Google Gemini API (answer generation)

## Run locally