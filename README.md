# Knowledge Graph Studio

Upload a PDF, turn it into a knowledge graph, and ask questions about it, including multi-hop
questions like *"Where is the company that Arjun works for headquartered?"*. Everything runs on
your own machine. No API key is needed.

```
PDF ──► page text ──► hybrid chunks ──► spaCy extraction ──► graph (Neo4j or embedded JSON)
                                         │  entities              Entity -[RELATION]-> Entity
                                         │  relations             Chunk  -[MENTIONS]-> Entity
                                         ▼
question ──► anchor entities ──► 1–3 hop expansion ──► graph facts + ranked passages ──► answer
                                 (hop-decayed scores)                                   (Ollama or rule-based)
```

## How it works

**Upload**
1. PyMuPDF extracts page-aware text.
2. Text is split into hybrid chunks: headings, paragraphs and sentence-safe overlap.
3. spaCy (`en_core_web_sm`) extracts entities from NER and noun chunks, and relations from
   dependency patterns:
   - subject–verb–object
   - passive voice (`ACQUIRED_BY`)
   - verb + preposition (`WORKS_AT`, `RUNS_ON`)
   - copula (`IS_A`, `CAPITAL_OF`)
   - appositives (`FOUNDER_OF`)
   - relative clauses
   - pronoun carry-over

   Negated statements are skipped.
4. Every chunk records which entities it mentions.
5. Triples, chunk→entity mentions and chunk embeddings are stored.

**Ask**
1. The rule-based graph engine answers direct (single-hop) questions from entity neighbourhoods.
2. In parallel, the question's anchor entities are expanded through the graph, hop by hop:
   - Typed relations carry their confidence.
   - Entities that share a chunk get a weaker co-mention link.
   - A node's score decays with distance: `1 / (1 + 0.6·(hop−1))`.
3. Chunks that mention the reached entities are boosted. This surfaces *bridge* passages that
   share no words with the question.
4. Passages are re-ranked at the sentence level with embeddings, question terms and graph entities.
5. A model writes the answer only from those facts and passages, and says *"It is not in the
   uploaded document"* otherwise: local Ollama (default `qwen2.5:3b`), or Groq
   (`openai/gpt-oss-120b`) when Ollama isn't available.
6. Without a model, multi-hop questions get a direction-aware chain built from typed relations,
   for example: *"Arjun Mehta works at Helix Labs. Maya Rao founded Helix Labs."* Other
   questions are answered with the PDF's own best-matching sentence.

Each answer shows how it was produced: the anchor entities, the graph facts with their hop
distance, and the source passages. The graph view highlights the path that was used.

## Tech stack

Python · FastAPI · spaCy · Neo4j (Cypher) · sentence-transformers (bge-small) · Ollama ·
PyMuPDF · vanilla JS frontend with an SVG force-directed graph view

## Quick start (local)

```bash
# 1. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt

# 2. Neo4j Community in Docker (free, no trial needed)
docker compose up -d neo4j          # browser at http://localhost:7474

# 3. Optional: local answer model
ollama pull qwen2.5:3b

# 4. Config
cp .env.example .env                # set NEO4J_PASSWORD to match docker-compose (default kgstudio123)

# 5. Run
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. API docs are at `/docs`.

Neo4j and Ollama are both optional:
- If Neo4j isn't reachable, the app uses the embedded JSON graph store.
- If Ollama isn't running, Groq writes the answers when `GROQ_API_KEY` is set; otherwise they
  come straight from the graph and the PDF's sentences.

`GET /health` shows which of them are live.

To run the whole stack in containers instead, use `docker compose up --build`. Ollama still runs
on the host.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `KG_GRAPH_BACKEND` | `auto` | `auto` (Neo4j if reachable, else embedded) · `neo4j` · `embedded` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` | Neo4j connection |
| `KG_EXTRACTOR` | `spacy` | `spacy` (local) · `llm` (OpenAI-compatible endpoint) · `hybrid` |
| `KG_ANSWER_LLM` | `auto` | `auto` (Ollama if running, else Groq if `GROQ_API_KEY` is set) · `ollama` · `groq` · `openai` · `none` |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Any Ollama model. `mistral` is more accurate if you have the RAM |
| `GROQ_API_KEY` / `GROQ_ANSWER_MODEL` | `openai/gpt-oss-120b` | Hosted answer writer for `groq` (and `auto` without Ollama) |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_ANSWER_MODEL` | | For `KG_ANSWER_LLM=openai` or `KG_EXTRACTOR=llm` |
| `KG_EMBEDDING_BACKEND` | `auto` | `auto` (bge-small) · `local` (hash embeddings, no download) |
| `KG_STATE_DIR` | `kg_app/state` | Where chunk/graph JSON state is written |
| `HEBBRIX_NATIVE_MODE` | `0` | `1` + `HEBBRIX_API_KEY` switches upload and ask to the Hebbrix API |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/upload_pdf` | Build the graph for a PDF (multipart `file`) |
| `POST` | `/ask` | `{"question": "..."}` → answer, mode, graph evidence, passages |
| `GET` | `/graph` | Nodes and edges of the active document for visualisation |
| `GET` | `/health` | Graph backend, extractor, answer model, active document |

## Deploy (Render)

`render.yaml` deploys the app as a free web service:
- graph: embedded store
- embeddings: hash
- answers: written by Groq (`openai/gpt-oss-120b`) from the graph evidence. Paste `GROQ_API_KEY`
  in the Render dashboard; without it the answer is the PDF's best-matching sentence.

Any other OpenAI-compatible provider works with `KG_ANSWER_LLM=openai` plus `OPENAI_API_KEY`,
`OPENAI_BASE_URL` and `OPENAI_ANSWER_MODEL`. To keep the graph across deploys, point `NEO4J_URI`
at a hosted Neo4j and set `KG_GRAPH_BACKEND=auto`.

## Tests

```bash
pytest -q
```

The suite runs without Neo4j, Ollama or model downloads. It covers the extractor patterns, both
graph stores, multi-hop expansion and the full upload → ask flow.

## Project structure

```text
kg_app/
├── api/server.py           # FastAPI app: upload, ask, graph, health
├── core/
│   ├── local_extractor.py  # spaCy entity + relation extraction
│   ├── extractor.py        # triple validation, optional LLM extraction
│   ├── query_engine.py     # rule-based answers + multi-hop graph evidence
│   ├── answer_llm.py       # Ollama / OpenAI-compatible answer writer
│   ├── embeddings.py       # bge-small embeddings with hash fallback
│   └── utils.py            # PDF text, hybrid chunking, helpers
├── db/graph.py             # Neo4jGraphStore + EmbeddedGraphStore
└── state/                  # chunk store and active document
static/                     # frontend
tests/
```
