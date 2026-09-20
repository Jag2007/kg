"""
Graph storage for the knowledge graph.

EmbeddedGraphStore keeps the graph in a JSON file on disk, so no external
database is needed. The interface is small so other backends can be added.

Data model (both backends):
    Document  -- one per uploaded PDF
    Entity    -- canonical name, scoped to a document
    Chunk     -- a passage of the PDF; Chunk -MENTIONS-> Entity
    Entity -[RELATION {confidence, chunk_id, sentence}]-> Entity
"""

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from kg_app.core.utils import normalise_relation_for_storage

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

STATE_DIR = Path(os.environ.get("KG_STATE_DIR", Path(__file__).resolve().parents[1] / "state"))
EMBEDDED_PATH = STATE_DIR / ".graph_store.json"

DEFAULT_CONFIDENCE = 0.7


def canonical_key(name: str) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"('s|’s)$", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


class BaseGraphStore:
    backend = "base"

    # ------------------------------------------------------------------ writes
    def store_document(self, document_id: str, file_name: str, summary: str) -> None:
        raise NotImplementedError

    def insert_triples(self, triples: list[dict], document_id: str) -> int:
        raise NotImplementedError

    def store_mentions(self, document_id: str, mentions: dict[str, list[str]]) -> int:
        raise NotImplementedError

    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ------------------------------------------------------------- raw reads
    def count_triples(self, document_id: str | None = None) -> int:
        raise NotImplementedError

    def get_document_summary(self, document_id: str) -> str:
        raise NotImplementedError

    def _load_document_triples(self, document_id: str) -> list[dict]:
        raise NotImplementedError

    def _load_mentions(self, document_id: str) -> dict[str, list[str]]:
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _normalise_rows(triples: Iterable[dict], document_id: str) -> list[dict]:
        rows: list[dict] = []
        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            relation = normalise_relation_for_storage(triple.get("relation", ""))
            object_ = str(triple.get("object", "")).strip()
            if not subject or not relation or not object_:
                continue
            try:
                confidence = float(triple.get("confidence", DEFAULT_CONFIDENCE) or DEFAULT_CONFIDENCE)
            except (TypeError, ValueError):
                confidence = DEFAULT_CONFIDENCE
            rows.append(
                {
                    "document_id": document_id,
                    "subject": subject,
                    "subject_lower": subject.lower(),
                    "relation": relation,
                    "relation_lower": relation.lower(),
                    "object": object_,
                    "object_lower": object_.lower(),
                    "searchable_text": f"{subject} {relation} {object_}".lower(),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "chunk_id": str(triple.get("chunk_id", "") or ""),
                    "sentence": str(triple.get("sentence", "") or "")[:400],
                }
            )
        return rows

    @staticmethod
    def _match_any_term(text: str, terms: list[str]) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in terms if term)

    @staticmethod
    def _searchable_row(row: dict) -> str:
        return (
            row.get("searchable_text")
            or f"{row.get('subject', '')} {row.get('relation', '')} {row.get('object', '')}".lower()
        )

    def _label(self, name: str) -> str:
        return f"{self.backend.upper()}_{name}"

    # ------------------------------------------------------- shared searches
    def find_relevant_entities(self, terms: list[str], document_id: str, limit: int = 10) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        names: dict[str, str] = {}
        for row in self._load_document_triples(document_id):
            for key in ("subject", "object"):
                value = str(row.get(key, "")).strip()
                if value:
                    names[value.lower()] = value

        def relevance(lowered: str) -> float:
            score = 0.0
            for term in cleaned_terms:
                if lowered == term:
                    score += 3.0
                elif lowered.startswith(term) or lowered.endswith(term):
                    score += 2.0
                elif term in lowered:
                    score += 1.0
            return score

        matches = [
            {"entity_name": original, "_score": relevance(lowered)}
            for lowered, original in names.items()
            if cleaned_terms and any(term in lowered for term in cleaned_terms)
        ]
        if not matches:
            matches = [{"entity_name": original, "_score": 0.0} for original in names.values()]
        matches = sorted(matches, key=lambda row: -row["_score"])[:limit]
        for match in matches:
            match.pop("_score", None)
        return self._label("ENTITY_LOOKUP"), matches

    def get_entity_neighborhood(self, entity_name: str, document_id: str, limit: int = 20) -> tuple[str, list[dict]]:
        lowered = entity_name.strip().lower()
        rows = []
        for row in self._load_document_triples(document_id):
            subject, object_ = row.get("subject", ""), row.get("object", "")
            if subject.lower() == lowered or object_.lower() == lowered:
                rows.append(
                    {
                        "center_name": entity_name,
                        "from_name": subject,
                        "relation_type": row.get("relation", ""),
                        "to_name": object_,
                        "neighbor_name": object_ if subject.lower() == lowered else subject,
                    }
                )
        return self._label("NEIGHBORHOOD"), rows[:limit]

    def search_related(self, terms: list[str], document_id: str, limit: int = 25) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        rows = [
            {"from_name": row.get("subject", ""), "relation_type": row.get("relation", ""), "to_name": row.get("object", "")}
            for row in self._load_document_triples(document_id)
            if self._match_any_term(row.get("subject", ""), cleaned_terms)
            or self._match_any_term(row.get("object", ""), cleaned_terms)
        ]
        return self._label("RELATED_SEARCH"), rows[:limit]

    def search_semantic(self, terms: list[str], document_id: str, limit: int = 40) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        rows = []
        for row in self._load_document_triples(document_id):
            searchable_text = self._searchable_row(row)
            if self._match_any_term(searchable_text, cleaned_terms):
                rows.append(
                    {
                        "from_name": row.get("subject", ""),
                        "relation_type": row.get("relation", ""),
                        "to_name": row.get("object", ""),
                        "searchable_text": searchable_text,
                    }
                )
        return self._label("SEMANTIC_SEARCH"), rows[:limit]

    def search_paths(self, terms: list[str], document_id: str, max_hops: int = 2, limit: int = 20) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        if not cleaned_terms:
            return self._label("PATH_SEARCH"), []

        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        display_names: dict[str, str] = {}
        anchors: set[str] = set()
        for row in self._load_document_triples(document_id):
            subject, relation, object_ = row.get("subject", ""), row.get("relation", ""), row.get("object", "")
            if not subject or not relation or not object_:
                continue
            subject_key, object_key = subject.lower(), object_.lower()
            display_names[subject_key] = subject
            display_names[object_key] = object_
            adjacency[subject_key].append((relation, object_key))
            adjacency[object_key].append((relation, subject_key))
            if any(term in subject_key for term in cleaned_terms) or any(term in object_key for term in cleaned_terms):
                anchors.add(subject_key)
                anchors.add(object_key)

        paths: list[dict] = []
        seen_paths = set()
        for anchor in anchors:
            queue = deque([(anchor, [anchor], [])])
            while queue and len(paths) < limit:
                node, path_nodes, path_relations = queue.popleft()
                if 0 < len(path_relations) <= max_hops:
                    token = (tuple(path_nodes), tuple(path_relations))
                    if token not in seen_paths:
                        seen_paths.add(token)
                        paths.append(
                            {
                                "path_nodes": [display_names.get(item, item) for item in path_nodes],
                                "path_relationships": path_relations[:],
                            }
                        )
                if len(path_relations) >= max_hops:
                    continue
                for relation, neighbor in adjacency.get(node, []):
                    if neighbor not in path_nodes:
                        queue.append((neighbor, path_nodes + [neighbor], path_relations + [relation]))
        return self._label("PATH_SEARCH"), paths[:limit]

    def get_graph_overview(self, document_id: str, limit: int = 12) -> tuple[str, list[dict]]:
        rows = sorted(self._load_document_triples(document_id), key=lambda row: -float(row.get("confidence", 0) or 0))
        return self._label("OVERVIEW"), [
            {"from_name": row.get("subject", ""), "relation_type": row.get("relation", ""), "to_name": row.get("object", "")}
            for row in rows[:limit]
        ]

    def get_graph_data(self, document_id: str, limit: int = 150) -> dict:
        """Nodes + edges for the frontend visualisation."""
        rows = sorted(self._load_document_triples(document_id), key=lambda row: -float(row.get("confidence", 0) or 0))
        degree: dict[str, int] = defaultdict(int)
        display: dict[str, str] = {}
        edges = []
        for row in rows[:limit]:
            source, target = canonical_key(row["subject"]), canonical_key(row["object"])
            display.setdefault(source, row["subject"])
            display.setdefault(target, row["object"])
            degree[source] += 1
            degree[target] += 1
            edges.append({"source": source, "target": target, "relation": row["relation"]})
        nodes = [{"id": key, "label": display[key], "degree": degree[key]} for key in display]
        return {"nodes": nodes, "edges": edges, "backend": self.backend}


class EmbeddedGraphStore(BaseGraphStore):
    """Single JSON file. Good for local dev without Docker and for small deploys."""

    backend = "embedded"
    _lock = threading.RLock()

    def __init__(self, path: Path | None = None):
        self.path = Path(path or EMBEDDED_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"documents": {}, "triples": {}, "mentions": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for key in ("documents", "triples", "mentions"):
            payload.setdefault(key, {})
        return payload

    def _write(self, payload: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def store_document(self, document_id: str, file_name: str, summary: str) -> None:
        with self._lock:
            payload = self._read()
            payload["documents"][document_id] = {
                "file_name": file_name,
                "summary": summary,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write(payload)

    def insert_triples(self, triples: list[dict], document_id: str) -> int:
        rows = self._normalise_rows(triples, document_id)
        if not rows:
            return 0
        with self._lock:
            payload = self._read()
            existing = {
                (row["subject_lower"], row["relation"], row["object_lower"]): row
                for row in payload["triples"].get(document_id, [])
            }
            for row in rows:
                key = (row["subject_lower"], row["relation"], row["object_lower"])
                if key in existing:
                    existing[key]["confidence"] = max(existing[key].get("confidence", 0), row["confidence"])
                else:
                    existing[key] = row
            payload["triples"][document_id] = list(existing.values())
            self._write(payload)
        return len(rows)

    def store_mentions(self, document_id: str, mentions: dict[str, list[str]]) -> int:
        with self._lock:
            payload = self._read()
            payload["mentions"][document_id] = {chunk_id: list(names) for chunk_id, names in mentions.items()}
            self._write(payload)
        return sum(len(names) for names in mentions.values())

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            payload = self._read()
            for key in ("documents", "triples", "mentions"):
                payload[key].pop(document_id, None)
            self._write(payload)

    def count_triples(self, document_id: str | None = None) -> int:
        triples = self._read()["triples"]
        if document_id:
            return len(triples.get(document_id, []))
        return sum(len(rows) for rows in triples.values())

    def get_document_summary(self, document_id: str) -> str:
        return str(self._read()["documents"].get(document_id, {}).get("summary", "")).strip()

    def _load_document_triples(self, document_id: str) -> list[dict]:
        return list(self._read()["triples"].get(document_id, []))

    def _load_mentions(self, document_id: str) -> dict[str, list[str]]:
        return dict(self._read()["mentions"].get(document_id, {}))


def active_backend_name() -> str:
    return "embedded"


def create_graph_store() -> BaseGraphStore:
    return EmbeddedGraphStore()


class GraphStore(BaseGraphStore):
    """Factory kept under the old name so existing imports keep working."""

    def __new__(cls, *args, **kwargs):
        return create_graph_store()
