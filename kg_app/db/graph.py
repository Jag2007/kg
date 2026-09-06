import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Iterable

from kg_app.core.utils import normalise_relation_for_storage
from kg_app.db.mongo import ENV_PATH, get_database

try:
    from pymongo import ASCENDING, UpdateOne
except Exception:  # pragma: no cover - dependency may be missing locally before install
    ASCENDING = 1
    UpdateOne = None


def _safe_uri(uri: str) -> str:
    if "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    host = rest.split("@", 1)[-1]
    return f"{scheme}://{host}"


def _current_mongo_uri() -> str:
    from kg_app.db.mongo import _read_env_value

    try:
        return _read_env_value("MONGO_URI")
    except Exception:
        return ""


def _current_db_name() -> str:
    from kg_app.db.mongo import _read_env_value

    try:
        return _read_env_value("DB_NAME")
    except Exception:
        return ""


class GraphStore:
    def __init__(self):
        try:
            self.client, self.db = get_database()
            self.documents = self.db["documents"]
            self.triples = self.db["triples"]
            self._ensure_schema()
        except Exception as error:
            print(f"Failed to connect to MongoDB at {_safe_uri(_current_mongo_uri())}: {error}")
            raise RuntimeError(
                "MongoDB driver not initialized. "
                f"Check MONGO_URI/DB_NAME in {ENV_PATH}. "
                f"Current URI: {_safe_uri(_current_mongo_uri())} "
                f"(database: {_current_db_name()})"
            ) from error

    def _ensure_schema(self) -> None:
        self.documents.create_index([("document_id", ASCENDING)], unique=True)
        self.triples.create_index(
            [
                ("document_id", ASCENDING),
                ("subject", ASCENDING),
                ("relation", ASCENDING),
                ("object", ASCENDING),
            ],
            unique=True,
        )
        self.triples.create_index([("document_id", ASCENDING), ("subject_lower", ASCENDING)])
        self.triples.create_index([("document_id", ASCENDING), ("object_lower", ASCENDING)])
        self.triples.create_index([("document_id", ASCENDING), ("relation_lower", ASCENDING)])

    def close(self):
        if getattr(self, "client", None) is not None:
            self.client.close()

    def store_document(self, document_id: str, file_name: str, summary: str) -> None:
        self.documents.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "file_name": file_name,
                    "summary": summary,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    def count_triples(self, document_id: str | None = None) -> int:
        query = {"document_id": document_id} if document_id else {}
        return int(self.triples.count_documents(query))

    @staticmethod
    def _normalise_rows(triples: Iterable[dict], document_id: str) -> list[dict]:
        rows: list[dict] = []
        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            relation = normalise_relation_for_storage(triple.get("relation", ""))
            object_ = str(triple.get("object", "")).strip()
            if not subject or not relation or not object_:
                continue
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
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        return rows

    def insert_triples(self, triples: list[dict], document_id: str) -> int:
        rows = self._normalise_rows(triples, document_id)
        if not rows:
            return 0

        if UpdateOne is None:
            raise RuntimeError(
                "pymongo bulk operations are unavailable. "
                "Run `pip install -r requirements.txt` after activating your virtual environment."
            )

        operations = [
            UpdateOne(
                {
                    "document_id": row["document_id"],
                    "subject": row["subject"],
                    "relation": row["relation"],
                    "object": row["object"],
                },
                {"$set": row},
                upsert=True,
            )
            for row in rows
        ]
        result = self.triples.bulk_write(operations, ordered=False)
        return int(result.upserted_count + result.modified_count + result.matched_count)

    def get_document_summary(self, document_id: str) -> str:
        record = self.documents.find_one({"document_id": document_id}, {"summary": 1, "_id": 0})
        return str(record.get("summary", "")).strip() if record else ""

    def _load_document_triples(self, document_id: str) -> list[dict]:
        cursor = self.triples.find(
            {"document_id": document_id},
            {
                "_id": 0,
                "subject": 1,
                "relation": 1,
                "object": 1,
                "searchable_text": 1,
                "subject_lower": 1,
                "object_lower": 1,
                "relation_lower": 1,
            },
        )
        return list(cursor)

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

    def find_relevant_entities(self, terms: list[str], document_id: str, limit: int = 10) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        triples = self._load_document_triples(document_id)
        names: dict[str, str] = {}
        for row in triples:
            for key in ("subject", "object"):
                value = str(row.get(key, "")).strip()
                if value:
                    names[value.lower()] = value

        def _relevance_score(lowered: str, original: str) -> float:
            if not cleaned_terms:
                return 0.0
            score = 0.0
            for term in cleaned_terms:
                if lowered == term:
                    score += 3.0
                elif lowered.startswith(term) or lowered.endswith(term):
                    score += 2.0
                elif term in lowered:
                    score += 1.0
            return score

        matches = []
        for lowered, original in names.items():
            if cleaned_terms and any(term in lowered for term in cleaned_terms):
                score = _relevance_score(lowered, original)
                matches.append({"entity_name": original, "_score": score})

        if not matches:
            matches = [{"entity_name": original, "_score": 0.0} for _, original in names.items()]

        matches = sorted(matches, key=lambda row: -row["_score"])[:limit]
        for m in matches:
            m.pop("_score", None)
        return "MONGO_ENTITY_LOOKUP", matches

    def get_entity_neighborhood(self, entity_name: str, document_id: str, limit: int = 20) -> tuple[str, list[dict]]:
        lowered = entity_name.strip().lower()
        rows = []
        for row in self._load_document_triples(document_id):
            subject = row.get("subject", "")
            object_ = row.get("object", "")
            relation = row.get("relation", "")
            if subject.lower() == lowered or object_.lower() == lowered:
                rows.append(
                    {
                        "center_name": entity_name,
                        "from_name": subject,
                        "relation_type": relation,
                        "to_name": object_,
                        "neighbor_name": object_ if subject.lower() == lowered else subject,
                    }
                )
        return "MONGO_NEIGHBORHOOD", rows[:limit]

    def search_related(self, terms: list[str], document_id: str, limit: int = 25) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        rows = []
        for row in self._load_document_triples(document_id):
            if self._match_any_term(row.get("subject", ""), cleaned_terms) or self._match_any_term(row.get("object", ""), cleaned_terms):
                rows.append(
                    {
                        "from_name": row.get("subject", ""),
                        "relation_type": row.get("relation", ""),
                        "to_name": row.get("object", ""),
                    }
                )
        return "MONGO_RELATED_SEARCH", rows[:limit]

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
        return "MONGO_SEMANTIC_SEARCH", rows[:limit]

    def search_paths(self, terms: list[str], document_id: str, max_hops: int = 2, limit: int = 20) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        triples = self._load_document_triples(document_id)
        if not cleaned_terms:
            return "MONGO_PATH_SEARCH", []

        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        display_names: dict[str, str] = {}
        anchors: set[str] = set()

        for row in triples:
            subject = str(row.get("subject", "")).strip()
            relation = str(row.get("relation", "")).strip()
            object_ = str(row.get("object", "")).strip()
            if not subject or not relation or not object_:
                continue

            subject_key = subject.lower()
            object_key = object_.lower()
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
                    if neighbor in path_nodes:
                        continue
                    queue.append((neighbor, path_nodes + [neighbor], path_relations + [relation]))

        return "MONGO_PATH_SEARCH", paths[:limit]

    def get_graph_overview(self, document_id: str, limit: int = 12) -> tuple[str, list[dict]]:
        rows = [
            {
                "from_name": row.get("subject", ""),
                "relation_type": row.get("relation", ""),
                "to_name": row.get("object", ""),
            }
            for row in self._load_document_triples(document_id)[:limit]
        ]
        return "MONGO_OVERVIEW", rows
