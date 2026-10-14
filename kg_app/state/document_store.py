import json
import os
from pathlib import Path

STATE_DIR = Path(os.environ.get("KG_STATE_DIR", Path(__file__).resolve().parent))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / ".active_document.json"


def set_active_document(document_id: str, file_name: str) -> None:
    payload = {
        "document_id": document_id,
        "file_name": file_name,
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def get_active_document() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}
