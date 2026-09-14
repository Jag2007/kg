# Knowledge Graph Studio

A FastAPI app that converts PDFs into a MongoDB-backed knowledge graph and answers questions using graph retrieval plus chunk-based semantic retrieval.

## Tech Stack

- Python
- FastAPI
- MongoDB Atlas with `pymongo`
- Hebbrix API (OpenAI-compatible)
- PyMuPDF
- sentence-transformers
- unittest

## Project Structure

```text
Knowledge Graphs/
├── app.py
├── kg_app/
│   ├── api/
│   │   └── server.py
│   ├── core/
│   │   ├── embeddings.py
│   │   ├── extractor.py
│   │   ├── query_engine.py
│   │   └── utils.py
│   ├── db/
│   │   ├── graph.py
│   │   └── mongo.py
│   └── state/
│       ├── chunk_store.py
│       └── document_store.py
├── static/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

`app.py` stays at the root as the `uvicorn` entry point.

## Main Features

- Upload a PDF and build a document-scoped knowledge graph
- Extract triples using Hebbrix or another OpenAI-compatible API
- Store graph data in MongoDB
- Store chunks locally for semantic retrieval
- Ask natural-language questions about the active uploaded PDF
- Combine graph lookup, semantic chunk lookup, and multi-hop fallback

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

```env
HEBBRIX_API_KEY=your_hebbrix_api_key
HEBBRIX_BASE_URL=https://api.hebbrix.com/v1
HEBBRIX_TRIPLE_MODEL=gpt-5-nano
HEBBRIX_METADATA_MODEL=gpt-5-nano
MONGO_URI=your_mongodb_connection_string
DB_NAME=knowledge_graph
```

## Run

```bash
uvicorn app:app --reload
```

Open:

- Frontend: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## How It Works

### Upload flow

- extract text from the PDF
- split it into hybrid chunks
- generate triples from chunks
- clean and deduplicate triples
- store graph data in MongoDB
- store chunk metadata locally
- mark the uploaded PDF as the active document

### Question flow

- search only within the active uploaded PDF
- retrieve relevant graph facts
- retrieve relevant semantic chunks
- combine the best evidence
- return a natural-language answer

## Tests

```bash
python -m unittest tests/test_smoke.py
```

## Notes

- Re-upload a PDF after major extraction or retrieval changes so the stored graph is rebuilt with the latest logic.
- Large PDFs will still take longer because triple extraction depends on LLM calls.
