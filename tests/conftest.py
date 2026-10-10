import os
import tempfile

# Deterministic test environment: no Neo4j, no Ollama, no model downloads.
# The legacy suites mock `extract_triples_llm`, so they run the LLM extractor path.
os.environ.setdefault("KG_EXTRACTOR", "llm")
os.environ.setdefault("KG_ANSWER_LLM", "none")
os.environ.setdefault("KG_GRAPH_BACKEND", "embedded")
os.environ.setdefault("KG_EMBEDDING_BACKEND", "local")
os.environ.setdefault("KG_STATE_DIR", tempfile.mkdtemp(prefix="kg-test-state-"))
