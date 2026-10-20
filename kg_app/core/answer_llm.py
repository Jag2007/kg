"""
Optional answer writer.

The graph and the chunk search decide *what* the answer is built from; the
model only turns that evidence into a readable answer.

KG_ANSWER_LLM:
    auto    - local Ollama when it is running, otherwise Groq when GROQ_API_KEY
              is set (default)
    ollama  - local Ollama
    groq    - Groq's hosted models (GROQ_API_KEY), for hosts like Render that
              cannot run a local model
    openai  - any OpenAI-compatible endpoint (OPENAI_BASE_URL / OPENAI_API_KEY)
    none    - no model; the answer is built from the evidence sentences
"""

import os
import time

import requests

NOT_FOUND_ANSWER = "It is not in the uploaded document. Please check the text."
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_STATUS_CACHE = {"checked_at": 0.0, "ok": False}
_STATUS_TTL_SECONDS = 30


def _base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _mode() -> str:
    return os.environ.get("KG_ANSWER_LLM", "auto").strip().lower()


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def _ollama_running() -> bool:
    now = time.time()
    if now - _STATUS_CACHE["checked_at"] < _STATUS_TTL_SECONDS:
        return _STATUS_CACHE["ok"]
    ok = False
    try:
        response = requests.get(f"{_base_url()}/api/tags", timeout=1.5)
        if response.ok:
            wanted = _ollama_model() if ":" in _ollama_model() else f"{_ollama_model()}:latest"
            ok = any(str(item.get("name", "")) == wanted for item in response.json().get("models", []))
    except Exception:
        ok = False
    _STATUS_CACHE.update(checked_at=now, ok=ok)
    return ok


def _provider() -> str:
    """The writer to use right now: "ollama", "groq", "openai", or "" for none."""
    mode = _mode()
    if mode in {"none", "off", "0", "false"}:
        return ""
    if mode == "openai":
        return "openai" if os.environ.get("OPENAI_API_KEY") else ""
    if mode == "groq":
        return "groq" if os.environ.get("GROQ_API_KEY") else ""
    if _ollama_running():
        return "ollama"
    if mode == "auto" and os.environ.get("GROQ_API_KEY"):
        return "groq"
    return ""


def model_name() -> str:
    provider = _provider() or _mode()
    if provider == "openai":
        return os.environ.get("OPENAI_ANSWER_MODEL", "gpt-4o-mini")
    if provider == "groq":
        return os.environ.get("GROQ_ANSWER_MODEL", "openai/gpt-oss-120b")
    return _ollama_model()


def is_available() -> bool:
    return bool(_provider())


def _build_prompt(question: str, facts: list[str], passages: list[str], draft: str) -> str:
    fact_block = "\n".join(f"- {fact}" for fact in facts[:20]) or "- (none)"
    passage_block = "\n\n".join(
        f"[{index}] {passage[:700]}" for index, passage in enumerate(passages[:4], start=1)
    ) or "(none)"
    draft_block = f"\nDraft answer from the graph engine (may be incomplete): {draft}\n" if draft else ""
    return f"""You answer questions about ONE uploaded PDF using only the evidence below.

Passages from the PDF (most relevant first):
{passage_block}

Short facts extracted from the PDF (automatic, may be noisy; closest to the question first):
{fact_block}
{draft_block}
Question: {question}

Rules:
- Use only the facts and passages above. Never use outside knowledge.
- Prefer the passages' own wording for definitions and explanations; use the facts only to connect steps.
- Ignore a fact that does not make sense as an English sentence.
- For questions that need several steps, chain the facts (e.g. A works at B, B is in C -> A is in C) and say the chain briefly.
- Answer in 1-4 plain sentences. No bullet points, no mention of "facts", "passages" or "graph".
- The answer is often a short detail inside a passage (a number, a name, a port, a date). Look carefully before giving up.
- Only if nothing above is related to the question, reply exactly: {NOT_FOUND_ANSWER}

Answer:"""


def generate_answer(question: str, facts: list[str], passages: list[str], draft: str = "") -> str:
    if not facts and not passages:
        return ""
    return _complete(_build_prompt(question, facts, passages, draft))


def _complete_chat(prompt: str, provider: str) -> str:
    if provider == "groq":
        base_url, api_key = GROQ_BASE_URL, os.environ.get("GROQ_API_KEY", "")
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "")
    model = model_name()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": int(os.environ.get("OLLAMA_MAX_TOKENS", "220")),
    }
    if model.startswith("openai/gpt-oss"):
        # gpt-oss reasons before it answers, and those tokens count towards max_tokens.
        payload["reasoning_effort"] = "low"
        payload["max_tokens"] += 600

    # Groq's free tier allows a few thousand tokens per minute: on 429, wait as told and retry once.
    for attempt in range(2):
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "90")),
        )
        if response.status_code != 429 or attempt == 1:
            break
        time.sleep(min(10.0, float(response.headers.get("retry-after") or 2)))
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"] or "")


def _complete(prompt: str) -> str:
    provider = _provider()
    if provider in {"groq", "openai"}:
        try:
            text = _complete_chat(prompt, provider).strip()
        except Exception as error:
            print(f"[answer_llm] {provider} call failed: {error}", flush=True)
            return ""
        if NOT_FOUND_ANSWER.lower().rstrip(".") in text.lower():
            return NOT_FOUND_ANSWER
        return text.strip('"').strip()
    payload = {
        "model": _ollama_model(),
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
