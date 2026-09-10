from pathlib import Path
import asyncio
import json
import os
import re
import requests
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kg_app.core.utils import (
    extract_text_from_pdf,
    extract_pdf_pages,
    build_hybrid_chunks,
    extract_keywords,
    split_into_sentences,
    summarize_text,
)
from kg_app.core.extractor import extract_triples_llm, extract_triples_fallback, precompute_chunk_metadata
from kg_app.db.graph import GraphStore
from kg_app.core.query_engine import ask_question
from kg_app.state.document_store import set_active_document, get_active_document
from kg_app.state.chunk_store import save_document_chunks, get_document_chunks

app = FastAPI(title="Knowledge Graph Builder")
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_TARGET_WORDS = int(os.environ.get("KG_TARGET_WORDS", "550"))
UPLOAD_MIN_WORDS = int(os.environ.get("KG_MIN_WORDS", "380"))
UPLOAD_MAX_WORDS = int(os.environ.get("KG_MAX_WORDS", "700"))
UPLOAD_OVERLAP_WORDS = int(os.environ.get("KG_OVERLAP_WORDS", "80"))
UPLOAD_MAX_CHUNKS = max(1, int(os.environ.get("KG_MAX_CHUNKS", "10")))
UPLOAD_WORKERS = max(1, min(2, int(os.environ.get("KG_UPLOAD_WORKERS", "1"))))
GROUNDING_STOP_WORDS = {
    "a", "an", "and", "are", "about", "after", "all", "also", "any", "before", "by",
    "can", "do", "does", "for", "from", "how", "in", "into", "is", "it", "its",
    "me", "of", "on", "or", "pdf", "question", "tell", "that", "the", "their",
    "them", "this", "those", "to", "uploaded", "what", "when", "where", "which",
    "who", "why", "with", "document",
}

OVERVIEW_NOISE_MARKERS = {
    "prerequisites",
    "objectives",
    "errata",
    "example code",
    "editor",
    "reviewer",
    "reviewers",
    "copyright",
    "isbn",
    "web page",
    "oreilly",
    "o'reilly",
    "acknowledg",
    "this book is aimed",
    "this book is here to help",
}


def _is_definition_question(question: str) -> bool:
    return str(question or "").strip().lower().startswith(("what is", "what are"))


def _is_overview_question_text(question: str) -> bool:
    text = str(question or "").strip().lower()
    patterns = (
        "what is this pdf about",
        "what does this pdf talk about",
        "what is the pdf about",
        "what is the pdf talking about",
        "what is the pdf talking",
        "what is the document about",
        "what does the pdf talk about",
        "what does the document talk about",
        "what is this document about",
        "what is this document talking about",
        "what is this pdf talking about",
        "what is the pdf talking about",
        "what is the pdf talking",
        "main topics",
        "give me a summary",
        "summarize",
        "summary",
        "overview",
    )
    return any(pattern in text for pattern in patterns)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


VERBOSE_BACKEND_LOGS = _env_flag("KG_VERBOSE_BACKEND_LOGS", "1")


def _backend_log(message: str) -> None:
    if VERBOSE_BACKEND_LOGS:
        print(f"[server] {message}", flush=True)


def _use_hebbrix_native() -> bool:
    return bool(os.environ.get("HEBBRIX_API_KEY")) and _env_flag("HEBBRIX_NATIVE_MODE", "1")


def _hebbrix_base_url() -> str:
    return os.environ.get("HEBBRIX_BASE_URL", "https://api.hebbrix.com/v1").rstrip("/")


def _hebbrix_timeout() -> int:
    return max(10, int(os.environ.get("HEBBRIX_HTTP_TIMEOUT_SECONDS", "120")))


def _encode_hebbrix_active_id(collection_id: str, document_id: str) -> str:
    return f"hebbrix:{collection_id}:{document_id}"


def _decode_hebbrix_active_id(value: str) -> dict:
    text = str(value or "").strip()
    if not text.startswith("hebbrix:"):
        return {}
    _, collection_id, document_id = text.split(":", 2)
    return {"collection_id": collection_id, "document_id": document_id}


def _hebbrix_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ.get('HEBBRIX_API_KEY', '').strip()}",
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _extract_hebbrix_id(payload: dict, *keys: str) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in ("id", "document_id", "collection_id"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    for container_key in ("document", "collection", "job", "result"):
        container = payload.get(container_key)
        if isinstance(container, dict):
            for key in keys or ("id",):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for nested_key in ("id", "document_id", "collection_id", "job_id"):
                nested_value = container.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in keys or ("id",):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_hebbrix_document(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    document = payload.get("document")
    if isinstance(document, dict):
        return document
    data = payload.get("data")
    if isinstance(data, dict):
        nested_document = data.get("document")
        if isinstance(nested_document, dict):
            return nested_document
        if any(key in data for key in ("id", "status", "collection_id", "original_filename")):
            return data
    if any(key in payload for key in ("id", "status", "collection_id", "original_filename")):
        return payload
    return {}


def _normalise_hebbrix_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "data", "chunks", "memories"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _hebbrix_request(method: str, path: str, *, payload=None, multipart: tuple[dict, tuple[str, bytes, str]] | None = None):
    url = f"{_hebbrix_base_url()}{path}"
    try:
        if multipart is not None:
            fields, file_tuple = multipart
            file_name, file_bytes, content_type = file_tuple
            response = requests.request(
                method.upper(),
                url,
                headers=_hebbrix_headers(),
                data={key: str(value) for key, value in fields.items()},
                files={"file": (file_name, file_bytes, content_type)},
                timeout=_hebbrix_timeout(),
            )
        else:
            response = requests.request(
                method.upper(),
                url,
                headers=_hebbrix_headers("application/json") if payload is not None else _hebbrix_headers(),
                json=payload,
                timeout=_hebbrix_timeout(),
            )

        if response.status_code >= 400:
            try:
                parsed = response.json()
                details = parsed.get("error", {}) if isinstance(parsed, dict) else {}
                message = details.get("message") or response.text or response.reason
            except Exception:
                message = response.text or response.reason
            allow_header = response.headers.get("Allow")
            if allow_header:
                message = f"{message} (Allow: {allow_header})"
            raise RuntimeError(f"Hebbrix API error ({response.status_code}) on {path}: {message}")

        if not response.text.strip():
            return {}
        return response.json()
    except Exception as error:
        raise RuntimeError(f"Hebbrix request failed for {path}: {error}") from error


def _create_hebbrix_collection(file_name: str) -> dict:
    name = f"{Path(file_name).stem}-{uuid.uuid4().hex[:8]}"
    payload = _hebbrix_request("POST", "/collections", payload={"name": name})
    collection_id = _extract_hebbrix_id(payload, "id", "collection_id")
    if not collection_id:
        raise RuntimeError("Hebbrix collection creation did not return an id.")
    _backend_log(f"Created Hebbrix collection {collection_id} for upload.")
    return payload


def _upload_hebbrix_document(file_name: str, content: bytes, collection_id: str) -> dict:
    upload_paths = ["/documents", "/documents/upload"]
    last_error = None
    for path in upload_paths:
        try:
            payload = _hebbrix_request(
                "POST",
                path,
                multipart=(
                    {"collection_id": collection_id, "extract_entities": "true"},
                    (file_name, content, "application/pdf"),
                ),
            )
            document_id = _extract_hebbrix_id(payload, "id", "document_id")
            if not document_id:
                raise RuntimeError(f"Hebbrix upload response from {path} did not return a document id.")
            _backend_log(f"Uploaded PDF to Hebbrix document {document_id} via {path}.")
            return payload
        except Exception as error:
            last_error = error
            if path != upload_paths[-1]:
                _backend_log(
                    f"Hebbrix upload via {path} failed ({error}). Trying the next available upload path."
                )
                continue
            raise
    raise RuntimeError(f"Hebbrix document upload failed: {last_error}")


def _wait_for_hebbrix_document(document_id: str) -> dict:
    max_wait = max(20, int(os.environ.get("HEBBRIX_DOCUMENT_WAIT_SECONDS", "180")))
    poll_seconds = max(2, int(os.environ.get("HEBBRIX_DOCUMENT_POLL_SECONDS", "4")))
    started = time.time()
    last_payload = {}
    while True:
        payload = _hebbrix_request("GET", f"/documents/{document_id}")
        last_payload = payload if isinstance(payload, dict) else {}
        document = _extract_hebbrix_document(last_payload)
        status = str(document.get("status", "") or last_payload.get("status", "")).strip().lower()
        if status in {"completed", "processed", "ready", "indexed", "searchable", "enriched"} or not status:
            if status:
                _backend_log(f"Hebbrix document {document_id} status: {status}.")
            return last_payload
        if status == "failed":
            raise RuntimeError(
                document.get("error_message")
                or last_payload.get("error_message")
                or f"Hebbrix document processing failed for {document_id}."
            )
        if time.time() - started > max_wait:
            raise RuntimeError(f"Timed out waiting for Hebbrix document {document_id} to finish processing.")
        _backend_log(f"Hebbrix document {document_id} status: {status}. Waiting {poll_seconds}s.")
        time.sleep(poll_seconds)


def _fetch_hebbrix_chunks(document_id: str) -> list[dict]:
    payload = _hebbrix_request("GET", f"/documents/{document_id}/chunks")
    items = _normalise_hebbrix_items(payload)
    chunks: list[dict] = []
    for index, item in enumerate(items, start=1):
        text = str(item.get("text") or item.get("content") or item.get("chunk_text") or "").strip()
        if not text:
            continue
        chunks.append(
            {
                "id": str(item.get("id") or f"chunk_{index}"),
                "text": text,
                "summary": str(item.get("summary", "")).strip(),
                "keywords": item.get("keywords", []) if isinstance(item.get("keywords", []), list) else [],
                "section": str(item.get("section", "Document")).strip() or "Document",
                "page": item.get("page"),
            }
        )
    return chunks


def _build_hebbrix_summary(file_name: str, chunks: list[dict], document_payload: dict) -> str:
    document = _extract_hebbrix_document(document_payload)
    explicit = str(document.get("summary", "") or document_payload.get("summary", "")).strip()
    if explicit:
        return explicit
    if chunks:
        return summarize_text(" ".join(chunk.get("text", "") for chunk in chunks[:3]))
    return f"The uploaded document {file_name} was indexed by Hebbrix."


def _search_hebbrix(question: str, collection_id: str, document_id: str = "") -> dict:
    payload = {
        "query": question,
        "collection_id": collection_id,
        "limit": 8,
    }
    if document_id:
        payload["document_id"] = document_id
    return _hebbrix_request("POST", "/search", payload=payload)


def _search_hebbrix_reason(question: str, collection_id: str, *, include_steps: bool = True) -> dict:
    payload = {
        "query": question,
        "collection_id": collection_id,
    }
    if include_steps:
        payload["include_steps"] = True
    return _hebbrix_request("POST", "/search/reason", payload=payload)


def _chat_hebbrix(question: str, collection_id: str, document_id: str = "", context_snippets: list[str] | None = None) -> dict:
    model = os.environ.get("HEBBRIX_CHAT_MODEL", "gpt-5-nano")
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the currently uploaded PDF. "
                "Do not use outside knowledge. "
                "Answer in 1-3 concise sentences. "
                "Do not mention excerpts, sources, or the retrieval process. "
                "If the question asks for a definition, give the definition directly. "
                "If the answer is not clearly supported by the uploaded PDF, reply exactly: "
                "It is not in the uploaded document. Please check the text."
            ),
        }
    ]
    if context_snippets:
        context_block = "\n\n".join(
            f"Excerpt {index + 1}:\n{snippet}"
            for index, snippet in enumerate(context_snippets[:3])
            if str(snippet).strip()
        ).strip()
        if context_block:
            messages.append(
                {
                    "role": "user",
                    "content": f"Use only these excerpts from the uploaded PDF.\n\n{context_block}\n\nQuestion: {question}",
                }
            )
        else:
            messages.append({"role": "user", "content": question})
    else:
        messages.append({"role": "user", "content": question})

    payload = {
        "model": model,
        "collection_id": collection_id,
        "messages": messages,
        "stream": False,
    }
    if document_id:
        payload["document_id"] = document_id
    return _hebbrix_request(
        "POST",
        "/chat/completions",
        payload=payload,
    )


def _query_hebbrix_graph(question: str, collection_id: str) -> dict:
    return _hebbrix_request(
        "POST",
        "/knowledge-graph/query",
        payload={
            "query": question,
            "collection_id": collection_id,
            "limit": 8,
        },
    )


def _extract_answer_from_hebbrix(payload: dict) -> str:
    if isinstance(payload.get("choices"), list):
        for choice in payload.get("choices", []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    for key in ("answer", "response", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    items = _normalise_hebbrix_items(payload)
    snippets: list[str] = []
    for item in items[:3]:
        for key in ("snippet", "summary", "content", "text"):
            value = str(item.get(key, "")).strip()
            if value:
                snippets.append(value)
                break
    return " ".join(snippets).strip()


def _format_hebbrix_results(payload: dict) -> list[dict]:
    items = _normalise_hebbrix_items(payload)
    results: list[dict] = []
    for item in items[:8]:
        text = _normalise_passage_text(item.get("snippet") or item.get("content") or item.get("text") or "")
        score = item.get("score")
        results.append(
            {
                "text": text,
                "score": score,
                "source": item.get("source") or item.get("id") or "hebbrix",
            }
        )
    return results


def _normalise_hebbrix_sources(payload) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    for key in ("sources", "results", "contexts", "references", "citations", "chunks"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            items = [item for item in value if isinstance(item, dict)]
            if items:
                return items

    memory_context = payload.get("memory_context")
    if isinstance(memory_context, dict):
        memories = memory_context.get("memories")
        if isinstance(memories, list):
            return [item for item in memories if isinstance(item, dict)]

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("sources", "results", "contexts", "chunks"):
            value = data.get(key)
            if isinstance(value, list):
                items = [item for item in value if isinstance(item, dict)]
                if items:
                    return items
    return []


def _hebbrix_result_document_id(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("document_id", "source_document_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("document_id", "source_document_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _format_hebbrix_reason_results(payload: dict, hebbrix_document_id: str = "") -> list[dict]:
    items = _normalise_hebbrix_sources(payload)
    results: list[dict] = []
    for item in items:
        item_document_id = _hebbrix_result_document_id(item)
        if hebbrix_document_id and item_document_id and item_document_id != hebbrix_document_id:
            continue

        text = _normalise_passage_text(item.get("content") or item.get("snippet") or item.get("text") or "")
        if not text:
            continue

        results.append(
            {
                "text": text,
                "score": item.get("score"),
                "source": item.get("memory_id") or item.get("id") or "hebbrix_reason",
            }
        )
        if len(results) >= 8:
            break
    return results


def _is_toc_like_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if "table of contents" in lowered:
        return True
    chapter_hits = len(re.findall(r"\bchapter\s+\d+\b", value, flags=re.IGNORECASE))
    dotted_leaders = ".." in value or "..." in value
    numbered_titles = len(
        re.findall(r"(?:\b[A-Z][A-Za-z&/-]+(?:\s+[A-Z][A-Za-z&/-]+){0,5}\s+\d{2,4}\b)", value)
    )
    page_numbers = len(re.findall(r"\b\d{2,4}\b", value))
    roman_page_marker = bool(re.search(r"\|\s*[ivxlcdm]+\b", lowered))
    return (
        chapter_hits >= 2
        or (chapter_hits >= 1 and dotted_leaders)
        or numbered_titles >= 4
        or (page_numbers >= 8 and len(value.split()) < 220)
        or roman_page_marker
    )


def _filter_hebbrix_items_to_document(payload: dict, hebbrix_document_id: str) -> list[dict]:
    items = _normalise_hebbrix_items(payload)
    if not hebbrix_document_id:
        filtered = items
    else:
        filtered = [
            item for item in items
            if _hebbrix_result_document_id(item) == hebbrix_document_id
        ]
    return [
        item for item in items
        if item in filtered and not _is_toc_like_text(item.get("content") or item.get("text") or item.get("snippet") or "")
    ]


def _tokenize_grounding_text(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", str(text or "").lower()):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if token and token not in GROUNDING_STOP_WORDS:
            tokens.add(token)
    return tokens


def _hebbrix_query_variants(question: str) -> list[str]:
    variants: list[str] = []
    seen = set()

    def add_variant(value: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip())
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(cleaned)

    add_variant(question)
    tokens = list(_tokenize_grounding_text(question))
    if tokens:
        add_variant(" ".join(tokens))
    if len(tokens) == 1:
        token = tokens[0]
        add_variant(token)
        if token.endswith("s") and len(token) > 4:
            add_variant(token[:-1])
        elif len(token) > 2:
            add_variant(f"{token}s")
    return variants[:4]


def _normalise_passage_text(text: str) -> str:
    value = str(text or "").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _split_hebbrix_sentences(text: str) -> list[str]:
    value = _normalise_passage_text(text).replace("\n", " ").strip()
    if not value:
        return []

    placeholder = "__HBX_DOT__"
    protected = value
    protected = re.sub(
        r"\b([A-Z])\.(?=\s+[A-Z][a-z])",
        lambda match: f"{match.group(1)}{placeholder}",
        protected,
    )
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.(?=\s)",
        lambda match: f"{match.group(1)}{placeholder}",
        protected,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [
        part.replace(placeholder, ".").strip()
        for part in parts
        if len(part.replace(placeholder, ".").strip()) > 2
    ]


def _trim_to_question_focus(text: str, question: str) -> str:
    value = str(text or "").strip()
    question_tokens = sorted(_tokenize_grounding_text(question), key=len, reverse=True)
    if not value or not question_tokens:
        return value
    lowered = value.lower()
    positions = [lowered.find(token) for token in question_tokens if lowered.find(token) != -1]
    if not positions:
        return value
    start = min(positions)
    prefix = value[:start].strip(" -:;,.")
    if start <= 0 or not prefix:
        return value
    if re.search(r"\d", prefix):
        return value[start:].strip()
    prefix_tokens = prefix.split()
    if len(prefix_tokens) >= 4 and prefix == prefix.title():
        return value[start:].strip()
    return value


def _is_heading_like_sentence(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if re.match(r"^chapter\s+\d+[:\s]", lowered):
        return True
    if re.match(r"^(table|figure)\s+\d+[:.\s]", lowered):
        return True
    if value.endswith("?") and value == value.title():
        return True
    if len(value.split()) <= 10 and value == value.title() and not re.search(r"\b(is|are|was|were|means|refers)\b", lowered):
        return True
    return False


def _fix_hebbrix_acronym_artifacts(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"^([A-Z]{2,})\)\s+are\b", r"\1s are", value)
    value = re.sub(r"^([A-Z]{2,})\)\s+is\b", r"\1 is", value)
    value = re.sub(r"^([A-Za-z0-9_-]+)\)", r"\1", value)
    value = re.sub(r"^[)\]}]+", "", value)
    return value


def _clean_hebbrix_sentence(text: str, question: str = "") -> str:
    value = _normalise_passage_text(text)
    if not value:
        return ""
    value = re.sub(r"([A-Za-z])[‐‑-]\s+([A-Za-z])", r"\1\2", value)
    value = _fix_hebbrix_acronym_artifacts(value)
    value = re.sub(r"^(Figure|Table)\s+\d+(?:-\d+)?\.?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^[A-Z][A-Za-z ]{0,40}\s+setup\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*\d{1,4}\s+", "", value)
    value = re.sub(r"\s+\d{1,4}\s*$", "", value)
    value = re.sub(r"\bPage\s+\d+\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value).strip(" -:;,.")
    value = _fix_hebbrix_acronym_artifacts(value)
    if _is_heading_like_sentence(value):
        return ""
    return value.strip()


def _looks_like_noisy_answer(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if _is_toc_like_text(value):
        return True
    if re.search(r"^\d{1,4}\s+[A-Z]", value):
        return True
    if re.search(r"\bchapter\s+\d+\b", value, flags=re.IGNORECASE) and len(value.split()) > 12:
        return True
    return False


def _clean_hebbrix_answer_text(text: str, question: str = "") -> str:
    sentences: list[str] = []
    seen = set()
    for sentence in _split_hebbrix_sentences(text):
        cleaned = _clean_hebbrix_sentence(sentence, question)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(cleaned)
    if sentences:
        return " ".join(sentences[:2]).strip()
    return _clean_hebbrix_sentence(text, question)


def _score_hebbrix_passage(question: str, passage: str) -> int:
    if not passage:
        return -100
    if _is_toc_like_text(passage):
        return -100
    question_tokens = _tokenize_grounding_text(question)
    passage_tokens = _tokenize_grounding_text(passage)
    if not question_tokens or not passage_tokens:
        return -50

    overlap = question_tokens & passage_tokens
    score = len(overlap) * 10
    lowered = passage.lower()
    word_count = len(passage.split())
    is_definition_question = _is_definition_question(question)
    score += sum(lowered.count(token) for token in question_tokens) * 3
    for token in question_tokens:
        if token in lowered:
            score += 2
            if re.search(rf"\b(?:a|an|the)\s+{re.escape(token)}s?\s+(is|are)\b", lowered):
                score += 18
            if re.search(rf"\b{re.escape(token)}s?\s+(is|are|refers to|means)\b", lowered):
                score += 16
            if re.search(rf"\b{re.escape(token)}\b\s+(is|are|refers to|means)\b", lowered):
                score += 6
            if re.search(rf"\b(is|are)\s+\b{re.escape(token)}\b", lowered):
                score += 4
            if is_definition_question and re.search(
                rf"\b{re.escape(token)}s?\b.*\b(is|are|refers to|means|deals with|consists of)\b",
                lowered,
            ):
                score += 18
    if any(marker in lowered for marker in (" is ", " are ", " refers to ", " known as ", " called ")):
        score += 4
    if is_definition_question:
        if any(phrase in lowered for phrase in (
            "at its essentials",
            "is a branch of",
            "is a type of",
            "is learning by",
            "is learning through",
            "is learning via",
            "deals with learning",
        )):
            score += 16
        if any(phrase in lowered for phrase in (
            "is exciting",
            "is different from",
            "in this chapter",
            "we will discuss",
            "applications are",
            "for the rest of us",
        )):
            score -= 20
    if word_count <= 45:
        score += 8
    elif word_count <= 80:
        score += 3
    elif word_count > 120:
        score -= min(18, ((word_count - 120) // 12 + 1) * 3)
    if "chapter " in lowered and len(overlap) <= 1:
        score -= 6
    return score


def _extract_grounded_sentences(results: list[dict], question: str, limit: int = 3) -> list[str]:
    candidates: list[tuple[int, str]] = []
    seen = set()
    for result in results[:5]:
        text = _normalise_passage_text(result.get("text", ""))
        if not text:
            continue
        for sentence in _split_hebbrix_sentences(text):
            cleaned = _clean_hebbrix_sentence(sentence, question)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            score = _score_hebbrix_passage(question, cleaned)
            if len(cleaned.split()) <= 35:
                score += 4
            if re.search(r"\b(is|are|refers to|means|defined as)\b", cleaned.lower()):
                score += 6
            candidates.append((score, cleaned))
    ranked = [sentence for score, sentence in sorted(candidates, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:limit]


def _score_overview_snippet(text: str) -> int:
    value = _normalise_passage_text(text)
    if not value or _is_toc_like_text(value):
        return -100
    lowered = value.lower()
    score = min(20, len(value.split()))
    if any(marker in lowered for marker in OVERVIEW_NOISE_MARKERS):
        score -= 40
    if any(marker in lowered for marker in (
        "chapter",
        "covers",
        "discusses",
        "explains",
        "learning",
        "network",
        "model",
        "algorithm",
        "data",
        "sequence",
        "reinforcement",
        "convolution",
        "generative",
        "neural",
    )):
        score += 20
    return score


def _build_hebbrix_overview_context(chunks: list[dict], limit: int = 6) -> list[str]:
    candidates: list[tuple[int, str]] = []
    seen = set()
    for chunk in chunks:
        text = _normalise_passage_text(chunk.get("text", ""))
        if not text or _is_toc_like_text(text):
            continue
        cleaned_sentences = []
        for sentence in _split_hebbrix_sentences(text):
            cleaned = _clean_hebbrix_sentence(sentence)
            if not cleaned:
                continue
            if len(cleaned.split()) < 6:
                continue
            cleaned_sentences.append(cleaned)
            if len(cleaned_sentences) >= 2:
                break
        if not cleaned_sentences:
            continue
        snippet = " ".join(cleaned_sentences[:2]).strip()
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append((_score_overview_snippet(snippet), snippet))
    ranked = [snippet for score, snippet in sorted(candidates, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:limit]


def _select_hebbrix_chunk_passages(question: str, chunks: list[dict], limit: int = 3) -> list[str]:
    question_tokens = list(_tokenize_grounding_text(question))
    if not question_tokens:
        return []

    candidates: list[tuple[int, str]] = []
    seen = set()
    for chunk in chunks:
        raw_text = str(chunk.get("text", "") or "")
        text = _normalise_passage_text(raw_text)
        if not text:
            continue
        lowered = text.lower()

        # Prefer paragraph-level passages from Hebbrix chunks to avoid mid-word or table-fragment snippets.
        paragraphs = [
            _normalise_passage_text(block)
            for block in re.split(r"\n\s*\n+", raw_text)
            if _normalise_passage_text(block)
        ]
        for index, paragraph in enumerate(paragraphs):
            para_lower = paragraph.lower()
            if not any(token in para_lower for token in question_tokens):
                continue
            snippet = paragraph
            if len(snippet) < 180 and index + 1 < len(paragraphs):
                snippet = _normalise_passage_text(f"{snippet} {paragraphs[index + 1]}")
            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((_score_hebbrix_passage(question, snippet), snippet))

        # Then fall back to short sentence windows for definitional answers.
        sentences = _split_hebbrix_sentences(text)
        if sentences:
            for index in range(len(sentences)):
                window = " ".join(sentences[index:index + 3]).strip()
                if not window:
                    continue
                key = window.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((_score_hebbrix_passage(question, window), window))

    ranked = [snippet for score, snippet in sorted(candidates, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:limit]


def _results_support_question(question: str, results: list[dict]) -> bool:
    return _question_text_support(
        question,
        [result.get("text", "") for result in results[:3]],
    )


def _question_text_support(question: str, texts: list[str]) -> bool:
    if not texts:
        return False
    question_tokens = _tokenize_grounding_text(question)
    if not question_tokens:
        return True

    text_tokens: set[str] = set()
    for text in texts:
        text_tokens.update(_tokenize_grounding_text(text))

    overlap = question_tokens & text_tokens
    required = 1 if len(question_tokens) <= 1 else min(2, len(question_tokens))
    return len(overlap) >= required


def _build_grounded_hebbrix_answer(results: list[dict], question: str = "") -> str:
    ranked_sentences = _extract_grounded_sentences(results, question, limit=3)
    if ranked_sentences:
        best = ranked_sentences[0]
        if len(ranked_sentences) > 1 and len(best.split()) < 18:
            second = ranked_sentences[1]
            if second.lower() != best.lower() and len(second.split()) <= 28:
                return f"{best} {second}".strip()
        return best

    snippets: list[str] = []
    seen = set()
    for result in results[:4]:
        text = _normalise_passage_text(result.get("text", ""))
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        snippets.append(text)
    if not snippets:
        return ""
    best_snippet = snippets[0]
    summary_sentences = [
        cleaned
        for cleaned in (_clean_hebbrix_sentence(sentence, question) for sentence in _split_hebbrix_sentences(best_snippet))
        if cleaned
    ]
    if summary_sentences:
        return " ".join(summary_sentences[:2]).strip()
    return _clean_hebbrix_sentence(best_snippet, question)


def _answer_supported_by_results(answer: str, results: list[dict]) -> bool:
    answer_tokens = _tokenize_grounding_text(answer)
    if not answer_tokens:
        return False
    result_tokens: set[str] = set()
    for result in results[:3]:
        result_tokens.update(_tokenize_grounding_text(result.get("text", "")))
    if not result_tokens:
        return False
    overlap = answer_tokens & result_tokens
    required = 2 if len(answer_tokens) >= 6 else 1
    return len(overlap) >= required


def _merge_chunks_for_limit(chunks: list[dict], max_chunks: int) -> list[dict]:
    if len(chunks) <= max_chunks:
        return chunks

    merged_chunks: list[dict] = []
    total = len(chunks)
    groups = min(max_chunks, total)
    for index in range(groups):
        start = (index * total) // groups
        end = ((index + 1) * total) // groups
        group = chunks[start:end]
        if not group:
            continue

        text = " ".join(str(chunk.get("text", "")).strip() for chunk in group if str(chunk.get("text", "")).strip()).strip()
        if not text:
            continue

        section = next((str(chunk.get("section", "")).strip() for chunk in group if str(chunk.get("section", "")).strip()), "Document")
        pages = [chunk.get("page") for chunk in group if chunk.get("page") is not None]
        merged_chunks.append(
            {
                "id": f"chunk_{len(merged_chunks) + 1}",
                "text": text,
                "summary": summarize_text(text),
                "keywords": extract_keywords(text),
                "section": section or "Document",
                "page": min(pages) if pages else None,
            }
        )

    return merged_chunks

class QuestionRequest(BaseModel):
    question: str


def build_document_summary(file_name: str, triples: list[dict]) -> str:
    if not triples:
        return f"The uploaded document {file_name} was processed, but no structured summary could be created."

    topics: list[str] = []
    seen = set()
    for triple in triples:
        for value in (triple.get("subject", ""), triple.get("object", "")):
            text = str(value).strip().replace("_", " ")
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                topics.append(text)
            if len(topics) >= 6:
                break
        if len(topics) >= 6:
            break

    if not topics:
        return f"The uploaded document {file_name} contains structured facts."
    if len(topics) == 1:
        return f"The uploaded document mainly discusses {topics[0]}."
    return "The uploaded document mainly discusses " + ", ".join(topics[:-1]) + f", and {topics[-1]}."


@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract knowledge triples, and save to MongoDB."""
    graph = None
    try:
        _backend_log(f"Upload started for file '{file.filename or 'unknown'}'.")
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return {
                "error": "Please upload a PDF file.",
                "trace": "",
            }

        content = await file.read()
        if not content:
            return {
                "error": "The uploaded PDF is empty.",
                "trace": "",
            }
        _backend_log(f"Read {len(content)} bytes from upload.")

        if _use_hebbrix_native():
            _backend_log("Using Hebbrix-native document pipeline for this upload.")
            collection_payload = _create_hebbrix_collection(file.filename)
            collection_id = _extract_hebbrix_id(collection_payload, "id", "collection_id")
            uploaded_document = _upload_hebbrix_document(file.filename, content, collection_id)
            hebbrix_document_id = _extract_hebbrix_id(uploaded_document, "id", "document_id")
            document_payload = _wait_for_hebbrix_document(hebbrix_document_id)
            document_record = _extract_hebbrix_document(document_payload)
            active_document_id = _encode_hebbrix_active_id(collection_id, hebbrix_document_id)
            hebbrix_chunks = _fetch_hebbrix_chunks(hebbrix_document_id)
            set_active_document(active_document_id, file.filename)
            summary = _build_hebbrix_summary(file.filename, hebbrix_chunks, document_payload)
            graph_items = int(
                document_record.get("relationships_count")
                or document_record.get("triples_count")
                or document_record.get("entities_count")
                or document_record.get("memory_count")
                or len(hebbrix_chunks)
            )
            _backend_log(
                f"Hebbrix-native upload complete with {len(hebbrix_chunks)} chunks and {graph_items} graph items."
            )
            return {
                "message": "PDF uploaded to Hebbrix and indexed successfully.",
                "file_name": file.filename,
                "chunks_processed": len(hebbrix_chunks),
                "triples_added": graph_items,
                "document_id": active_document_id,
                "summary": summary,
                "sample_triples": [],
                "debug": {
                    "provider": "hebbrix",
                    "collection_id": collection_id,
                    "hebbrix_document_id": hebbrix_document_id,
                    "hebbrix_status": document_record.get("status"),
                    "chunks_processed": len(hebbrix_chunks),
                    "graph_items": graph_items,
                    "summary": summary,
                },
            }
        
        # 1. Process PDF into page-aware text
        try:
            pages = extract_pdf_pages(content)
        except Exception:
            pages = [{"page": 1, "text": extract_text_from_pdf(content)}]
        _backend_log(f"Extracted readable text from {len(pages)} pages.")
        text = "\n".join(page["text"] for page in pages)
        if not text.strip():
            return {
                "error": "No readable text was found in the uploaded PDF.",
                "trace": "",
            }
        
        # 2. Build rich hybrid chunks: headings + paragraphs + sentence-safe sliding overlap.
        all_chunks = build_hybrid_chunks(
            pages,
            target_words=UPLOAD_TARGET_WORDS,
            min_words=UPLOAD_MIN_WORDS,
            max_words=UPLOAD_MAX_WORDS,
            overlap_ratio=UPLOAD_OVERLAP_WORDS / max(1, UPLOAD_TARGET_WORDS),
        )
        _backend_log(f"Built {len(all_chunks)} hybrid chunks for upload processing.")
        if not all_chunks:
            return {
                "error": "The document could not be split into usable text chunks.",
                "trace": "",
            }

        # Limit chunks for LLM triple extraction only; keep all chunks for semantic retrieval.
        extraction_chunks = all_chunks
        if len(all_chunks) > UPLOAD_MAX_CHUNKS:
            extraction_chunks = _merge_chunks_for_limit(all_chunks, UPLOAD_MAX_CHUNKS)
            _backend_log(
                f"Using {len(extraction_chunks)} merged chunks for triple extraction "
                f"(all {len(all_chunks)} chunks kept for semantic retrieval)."
            )

        extraction_chunks = precompute_chunk_metadata(extraction_chunks)
        _backend_log("Chunk metadata enrichment step finished.")
        chunk_texts = [chunk["text"] for chunk in extraction_chunks]
        
        graph = GraphStore()
        document_id = str(uuid.uuid4())
        _backend_log(f"Created document id {document_id}.")
        extracted_triples_debug = []
        all_triples = []
        seen = set()
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
            _backend_log(
                f"Starting LLM triple extraction for {len(chunk_texts)} chunks with {UPLOAD_WORKERS} worker(s)."
            )
            tasks = [loop.run_in_executor(executor, extract_triples_llm, chunk_text) for chunk_text in chunk_texts]
            raw_batches = await asyncio.gather(*tasks, return_exceptions=True)

        triple_batches: list[list[dict]] = []
        for index, batch in enumerate(raw_batches, start=1):
            if isinstance(batch, Exception):
                print(f"[server] Chunk {index}/{len(raw_batches)} extraction failed: {batch}", flush=True)
                fallback_triples = extract_triples_fallback(chunk_texts[index - 1])
                _backend_log(
                    f"Chunk {index}/{len(raw_batches)} fallback extraction returned {len(fallback_triples)} triples."
                )
                triple_batches.append(fallback_triples)
            else:
                if not batch:
                    fallback_triples = extract_triples_fallback(chunk_texts[index - 1])
                    _backend_log(
                        f"Chunk {index}/{len(raw_batches)} fallback extraction returned {len(fallback_triples)} triples."
                    )
                    triple_batches.append(fallback_triples)
                    continue
                _backend_log(
                    f"Chunk {index}/{len(raw_batches)} extraction returned {len(batch)} triples."
                )
                triple_batches.append(batch)

        # 3. Accumulate validated triples (dedup across the whole document)
        for triples in triple_batches:
            for t in triples:
                key = (t["subject"].strip().lower(), t["relation"].strip().lower(), t["object"].strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                all_triples.append(t)
                if len(extracted_triples_debug) < 10:
                    extracted_triples_debug.append((t["subject"], t["relation"], t["object"]))
        _backend_log(f"Accumulated {len(all_triples)} unique triples across all chunks.")

        summary = build_document_summary(file.filename, all_triples)
        graph.store_document(document_id, file.filename, summary)
        triples_added = graph.insert_triples(all_triples, document_id)
        _backend_log(
            f"Stored document summary and inserted {triples_added} triples into MongoDB."
        )
        save_document_chunks(document_id, file.filename, all_chunks)
        set_active_document(document_id, file.filename)
        _backend_log("Saved chunk state and set the uploaded document as active.")
        
        # Failsafe exactly as requested
        if triples_added == 0:
            return {
                "error": "No structured knowledge was extracted from the document.",
                "note": "The PDF text chunks were still saved for semantic retrieval.",
                "trace": "",
                "debug": {
                    "chunks_processed": len(all_chunks),
                    "extraction_chunks": len(extraction_chunks),
                    "sample_triples": extracted_triples_debug,
                    "triples_added": triples_added,
                    "workers_used": UPLOAD_WORKERS,
                    "document_id": document_id,
                },
            }

        return {
            "message": "PDF processed successfully. The knowledge graph is ready.",
            "file_name": file.filename,
            "chunks_processed": len(all_chunks),
            "triples_added": triples_added,
            "document_id": document_id,
            "summary": summary,
            "sample_triples": extracted_triples_debug,
            "debug": {
                "file_name": file.filename,
                "chunks_processed": len(all_chunks),
                "extraction_chunks": len(extraction_chunks),
                "sample_triples": extracted_triples_debug,
                "triples_added": triples_added,
                "workers_used": UPLOAD_WORKERS,
                "document_id": document_id,
                "summary": summary,
            },
        }

    except Exception as e:
        print("ERROR IN UPLOAD_PDF:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise HTTPException(
            status_code=502,
            detail={
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        ) from e
    finally:
        if graph is not None:
            try:
                graph.close()
            except Exception:
                pass

@app.post("/ask")
async def ask(q: QuestionRequest):
    """Ask a natural language question against the Knowledge Graph."""
    question = q.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please provide a question.")

    try:
        _backend_log(f"Received question: {question}")
        active_document = get_active_document()
        document_id = str(active_document.get("document_id", "")).strip()
        if not document_id:
            return {
                "query": "",
                "results": [],
                "answer": "Please upload a PDF before asking a question.",
                "debug": {
                    "question": question,
                    "error": "No active document found.",
                },
            }

        hebbrix_ref = _decode_hebbrix_active_id(document_id)
        if hebbrix_ref and _use_hebbrix_native():
            collection_id = hebbrix_ref.get("collection_id", "")
            hebbrix_document_id = hebbrix_ref.get("document_id", "")
            _backend_log(
                f"Using Hebbrix-native retrieval for collection {collection_id} and document {hebbrix_document_id}."
            )
            if _is_overview_question_text(question):
                hebbrix_chunks = _fetch_hebbrix_chunks(hebbrix_document_id)
                overview_context = _build_hebbrix_overview_context(hebbrix_chunks)
                answer = ""
                if overview_context:
                    overview_prompt = (
                        "Summarize the uploaded PDF in 2 concise sentences. "
                        "Mention only the main topics covered in the PDF. "
                        "Do not mention excerpts, page numbers, or the retrieval process. "
                        "If the excerpts are insufficient, reply exactly: "
                        "It is not in the uploaded document. Please check the text."
                    )
                    chat_payload = _chat_hebbrix(
                        overview_prompt,
                        collection_id,
                        hebbrix_document_id,
                        overview_context[:4],
                    )
                    candidate_answer = _clean_hebbrix_answer_text(
                        _extract_answer_from_hebbrix(chat_payload),
                        question,
                    )
                    if candidate_answer and not _looks_like_noisy_answer(candidate_answer):
                        answer = candidate_answer
                if not answer:
                    answer = _build_hebbrix_summary(
                        active_document.get("file_name", "document.pdf"),
                        [{"text": snippet} for snippet in overview_context],
                        {},
                    )
                return {
                    "triples_added": 0,
                    "query": question,
                    "results": [{"text": snippet, "score": None, "source": "hebbrix_overview"} for snippet in overview_context[:4]],
                    "answer": answer or "It is not in the uploaded document. Please check the text.",
                    "steps": ["Hebbrix document overview"],
                    "debug": {
                        "provider": "hebbrix",
                        "mode": "native-document-overview",
                        "collection_id": collection_id,
                        "document_id": document_id,
                        "hebbrix_document_id": hebbrix_document_id,
                        "results_count": len(overview_context[:4]),
                    },
                }

            try:
                reason_payload = _search_hebbrix_reason(question, collection_id, include_steps=True)
            except Exception as error:
                _backend_log(f"Hebbrix search/reason failed, falling back to manual search path: {error}")
                reason_payload = {}

            reason_results = _format_hebbrix_reason_results(reason_payload, hebbrix_document_id)
            reason_answer = _clean_hebbrix_answer_text(
                _extract_answer_from_hebbrix(reason_payload),
                question,
            )
            if (
                reason_answer
                and reason_answer != "It is not in the uploaded document. Please check the text."
                and not _looks_like_noisy_answer(reason_answer)
                and _question_text_support(
                    question,
                    [reason_answer] + [result.get("text", "") for result in reason_results[:3]],
                )
            ):
                return {
                    "triples_added": 0,
                    "query": question,
                    "results": reason_results,
                    "answer": reason_answer,
                    "steps": ["Hebbrix search reason"],
                    "debug": {
                        "provider": "hebbrix",
                        "mode": "native-document-search-reason",
                        "collection_id": collection_id,
                        "document_id": document_id,
                        "hebbrix_document_id": hebbrix_document_id,
                        "results_count": len(reason_results),
                        "reasoning_context": reason_payload.get("reasoning_context"),
                    },
                }

            results: list[dict] = []
            raw_search_results: list[dict] = []
            for query_variant in _hebbrix_query_variants(question):
                search_payload = _search_hebbrix(query_variant, collection_id, hebbrix_document_id)
                search_items = _filter_hebbrix_items_to_document(search_payload, hebbrix_document_id)
                results = _format_hebbrix_results({"results": search_items})
                if results and not raw_search_results:
                    raw_search_results = results
                if _results_support_question(question, results):
                    break
            if not _results_support_question(question, results):
                results = []
            passage_results: list[dict] = []
            if not results:
                hebbrix_chunks = _fetch_hebbrix_chunks(hebbrix_document_id)
                passage_snippets = _select_hebbrix_chunk_passages(question, hebbrix_chunks, limit=3)
                passage_results = [{"text": snippet, "score": None, "source": "hebbrix_chunk"} for snippet in passage_snippets]
                if passage_results and not _results_support_question(question, passage_results):
                    passage_results = []
            if not results and passage_results:
                results = passage_results
            answer = ""
            grounded_answer = _build_grounded_hebbrix_answer(passage_results or results, question=question)
            if grounded_answer and _answer_supported_by_results(grounded_answer, passage_results or results):
                answer = grounded_answer

            # Use validated results for context, fall back to raw search results when validation
            # cleared them (e.g. keyword mismatch despite document having relevant content).
            validated_context = passage_results or results
            chat_context = [r.get("text", "") for r in (validated_context or raw_search_results)[:3]]
            if chat_context and (not answer or not _is_definition_question(question)):
                chat_payload = _chat_hebbrix(
                    question,
                    collection_id,
                    hebbrix_document_id,
                    chat_context,
                )
                candidate_answer = _clean_hebbrix_answer_text(
                    _extract_answer_from_hebbrix(chat_payload),
                    question,
                )
                if (
                    candidate_answer
                    and not _looks_like_noisy_answer(candidate_answer)
                    and candidate_answer != "It is not in the uploaded document. Please check the text."
                    and _answer_supported_by_results(candidate_answer, validated_context or raw_search_results)
                    and _score_hebbrix_passage(question, candidate_answer) >= _score_hebbrix_passage(question, answer)
                ):
                    answer = candidate_answer.strip()

            if not answer:
                answer = "It is not in the uploaded document. Please check the text."
            return {
                "triples_added": 0,
                "query": question,
                "results": results,
                "answer": answer,
                "steps": ["Hebbrix document search"],
                "debug": {
                    "provider": "hebbrix",
                    "mode": "native-document-only-search-chat",
                    "collection_id": collection_id,
                    "document_id": document_id,
                    "hebbrix_document_id": hebbrix_document_id,
                    "results_count": len(results),
                    "passage_results_count": len(passage_results),
                },
            }
        elif _use_hebbrix_native():
            return {
                "query": "",
                "results": [],
                "answer": "Please upload a PDF in Hebbrix mode before asking a question.",
                "debug": {
                    "question": question,
                    "error": "No Hebbrix-backed active document found.",
                    "document_id": document_id,
                },
            }

        saved_chunks = get_document_chunks(document_id)
        graph = GraphStore()
        triple_count = graph.count_triples(document_id)
        graph.close()
        if not saved_chunks and triple_count <= 0:
            return {
                "query": "",
                "results": [],
                "answer": "Please upload and finish processing a PDF before asking a question.",
                "debug": {
                    "question": question,
                    "error": "Active document has no saved chunks or graph triples.",
                    "document_id": document_id,
                },
            }
        _backend_log(
            f"Answering against document {document_id} with {len(saved_chunks)} saved chunks and {triple_count} triples."
        )

        response = ask_question(question, document_id=document_id)
        _backend_log(
            f"Question answered against document {document_id} with {len(response.get('results', []))} result rows."
        )
        results = response.get("results", [])
        answer = response.get("answer", "").strip()

        if not results:
            answer = "It is not in the uploaded document. Please check the text."
        elif not answer:
            answer = "A result was found in the uploaded document."

        return {
            "triples_added": response.get("triples_added", 0),
            "query": response.get("query", ""),
            "results": results,
            "answer": answer,
            "steps": response.get("steps_taken", response.get("steps", [])),
            "debug": {
                "question": question,
                "query": response.get("query", ""),
                "results": results,
                "steps": response.get("steps_taken", response.get("steps", [])),
                "triples_added": response.get("triples_added", 0),
                "document_id": document_id,
                "active_file_name": active_document.get("file_name", ""),
            },
        }
    except Exception as e:
        print("ERROR IN ASK:", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "query": "",
            "results": [],
            "answer": "It is not in the uploaded document. Please check the text.",
            "debug": {
                "question": question,
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        }
