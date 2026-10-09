"""
Optional answer writer backed by a local Ollama model.

The graph and the chunk search decide *what* the answer is built from; the
model only turns that evidence into a readable answer. When Ollama is not
running (or KG_ANSWER_LLM=none) callers fall back to the rule-based answer.
"""

import os
import time

import requests

NOT_FOUND_ANSWER = "It is not in the uploaded document. Please check the text."

_STATUS_CACHE = {"checked_at": 0.0, "ok": False}
_STATUS_TTL_SECONDS = 30


def _base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def model_name() -> str:
    return os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def is_available() -> bool:
    mode = os.environ.get("KG_ANSWER_LLM", "auto").strip().lower()
    if mode in {"none", "off", "0", "false"}:
        return False
    now = time.time()
    if now - _STATUS_CACHE["checked_at"] < _STATUS_TTL_SECONDS:
        return _STATUS_CACHE["ok"]
    ok = False
    try:
        response = requests.get(f"{_base_url()}/api/tags", timeout=1.5)
        if response.ok:
            wanted = model_name() if ":" in model_name() else f"{model_name()}:latest"
            ok = any(str(item.get("name", "")) == wanted for item in response.json().get("models", []))
    except Exception:
        ok = False
    _STATUS_CACHE.update(checked_at=now, ok=ok)
    return ok


def _build_prompt(question: str, facts: list[str], passages: list[str], draft: str) -> str:
    fact_block = "\n".join(f"- {fact}" for fact in facts[:20]) or "- (none)"
    passage_block = "\n\n".join(
        f"[{index}] {passage[:700]}" for index, passage in enumerate(passages[:4], start=1)
    ) or "(none)"
    draft_block = f"\nDraft answer from the graph engine (may be incomplete): {draft}\n" if draft else ""
    return f"""You answer questions about ONE uploaded PDF using only the evidence below.

Knowledge graph facts (subject -> relation -> object; hop = distance from the question's entity):
{fact_block}

Passages from the PDF:
{passage_block}
{draft_block}
Question: {question}

Rules:
- Use only the facts and passages above. Never use outside knowledge.
- For questions that need several steps, chain the facts (e.g. A works at B, B is in C -> A is in C) and say the chain briefly.
- Answer in 1-4 plain sentences. No bullet points, no mention of "facts", "passages" or "graph".
- The answer is often a short detail inside a passage (a number, a name, a port, a date). Look carefully before giving up.
- Only if nothing above is related to the question, reply exactly: {NOT_FOUND_ANSWER}

Answer:"""


def generate_answer(question: str, facts: list[str], passages: list[str], draft: str = "") -> str:
    if not facts and not passages:
        return ""
    return _complete(_build_prompt(question, facts, passages, draft))


def _complete(prompt: str) -> str:
    payload = {
        "model": model_name(),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": int(os.environ.get("OLLAMA_MAX_TOKENS", "220")),
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "2048")),
        },
    }
    try:
        response = requests.post(
            f"{_base_url()}/api/generate",
            json=payload,
            timeout=int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "90")),
        )
        response.raise_for_status()
        text = str(response.json().get("response", "")).strip()
    except Exception as error:
        print(f"[answer_llm] Ollama call failed: {error}", flush=True)
        _STATUS_CACHE.update(checked_at=time.time(), ok=False)
        return ""
    if NOT_FOUND_ANSWER.lower().rstrip(".") in text.lower():
        return NOT_FOUND_ANSWER
    return text.strip().strip('"').strip()
