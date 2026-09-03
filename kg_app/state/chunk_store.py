import json
from pathlib import Path

from kg_app.core.embeddings import embed_text

STATE_PATH = Path(__file__).resolve().parent / ".document_chunks.json"


def _load_payload() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _save_payload(payload: dict) -> None:
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _normalise_chunk(chunk, index: int) -> dict:
    if isinstance(chunk, dict):
        text = str(chunk.get("text", "")).strip()
        if not text:
            return {}
        return {
            "id": str(chunk.get("id") or f"chunk_{index}"),
            "text": text,
            "summary": str(chunk.get("summary", "")).strip(),
            "keywords": [str(keyword).strip() for keyword in chunk.get("keywords", []) if str(keyword).strip()]
            if isinstance(chunk.get("keywords", []), list)
            else [],
            "section": str(chunk.get("section", "Document")).strip() or "Document",
            "page": chunk.get("page"),
            "embedding": chunk.get("embedding") if isinstance(chunk.get("embedding"), list) else embed_text(text),
        }

    text = str(chunk).strip()
    if not text:
        return {}
    return {
        "id": f"chunk_{index}",
        "text": text,
        "summary": "",
        "keywords": [],
        "section": "Document",
        "page": None,
        "embedding": embed_text(text),
    }


def save_document_chunks(document_id: str, file_name: str, chunks: list) -> None:
    payload = _load_payload()
    normalised_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        normalised = _normalise_chunk(chunk, index)
        if normalised:
            normalised_chunks.append(normalised)

    payload[document_id] = {
        "file_name": file_name,
        "chunks": normalised_chunks,
    }
    _save_payload(payload)


def get_document_chunks(document_id: str) -> list[dict]:
    payload = _load_payload()
    document = payload.get(document_id, {})
    chunks = document.get("chunks", [])
    if isinstance(chunks, list):
        normalised_chunks = []
        for index, chunk in enumerate(chunks, start=1):
            normalised = _normalise_chunk(chunk, index)
            if normalised:
                normalised_chunks.append(normalised)
        return normalised_chunks
    return []


def clear_document_chunks(document_id: str) -> None:
    payload = _load_payload()
    if document_id in payload:
        payload.pop(document_id, None)
        _save_payload(payload)


def clear_all_document_chunks() -> None:
    if STATE_PATH.exists():
        try:
            STATE_PATH.unlink()
        except Exception:
            pass
