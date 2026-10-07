"""
Graph storage for the knowledge graph.

Two interchangeable backends share one interface:

* Neo4jGraphStore    - Neo4j (Community edition works) queried with Cypher.
* EmbeddedGraphStore - a JSON file on disk; no external services needed.

`GraphStore()` picks the backend from KG_GRAPH_BACKEND (auto | neo4j | embedded).
In auto mode Neo4j is used when NEO4J_URI is set and reachable, otherwise the
embedded store is used so the app keeps working (e.g. on a free Render instance).

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

# Multi-hop expansion settings.
HOP_LIMITS = {1: 25, 2: 12, 3: 5}
HOP_DECAY = 0.6
CO_MENTION_WEIGHT = 0.35
DEFAULT_CONFIDENCE = 0.7


def canonical_key(name: str) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"('s|’s)$", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def hop_decay(distance: int) -> float:
    return 1.0 / (1.0 + HOP_DECAY * max(distance - 1, 0))


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

    # ---------------------------------------------------------- multi-hop
    def _typed_edges(self, document_id: str) -> list[dict]:
        return self._load_document_triples(document_id)

    def expand_entities(
        self,
        seeds: list[str],
        document_id: str,
        depth: int = 2,
        query_terms: list[str] | None = None,
    ) -> tuple[str, dict]:
        """
        Walk outward from the seed entities, hop by hop.

        Typed relation edges carry their confidence; entities that are only
        mentioned in the same chunk are linked by a weaker co-mention edge.
        Each node keeps the shortest distance at which it was reached and the
        best path score, decayed by distance, so a fact two hops away can
        still support an answer without drowning out direct facts.
        """
        depth = max(1, min(3, int(depth)))
        query_terms = [term.lower() for term in (query_terms or []) if term]
        adjacency: dict[str, list[dict]] = defaultdict(list)
        names: dict[str, str] = {}

        for row in self._typed_edges(document_id):
            subject_key, object_key = canonical_key(row["subject"]), canonical_key(row["object"])
            if not subject_key or not object_key or subject_key == object_key:
                continue
            names.setdefault(subject_key, row["subject"])
            names.setdefault(object_key, row["object"])
            edge = {
                "from": row["subject"],
                "relation": row["relation"],
                "to": row["object"],
                "weight": float(row.get("confidence", DEFAULT_CONFIDENCE) or DEFAULT_CONFIDENCE),
                "sentence": row.get("sentence", ""),
                "chunk_id": row.get("chunk_id", ""),
            }
            adjacency[subject_key].append({**edge, "target": object_key})
            adjacency[object_key].append({**edge, "target": subject_key})

        # Co-mention edges: entities that appear in the same chunk.
        shared: dict[tuple[str, str], set[str]] = defaultdict(set)
        for chunk_id, entity_names in self._load_mentions(document_id).items():
            keys = sorted({canonical_key(name) for name in entity_names if canonical_key(name)})[:40]
            for name in entity_names:
                names.setdefault(canonical_key(name), name)
            for i, left in enumerate(keys):
                for right in keys[i + 1 :]:
                    shared[(left, right)].add(chunk_id)
        for (left, right), chunk_ids in shared.items():
            weight = CO_MENTION_WEIGHT * min(1.0, len(chunk_ids) / 2)
            for source, target in ((left, right), (right, left)):
                adjacency[source].append(
                    {
                        "from": names.get(left, left),
                        "relation": "CO_MENTIONED",
                        "to": names.get(right, right),
                        "weight": weight,
                        "sentence": "",
                        "chunk_id": sorted(chunk_ids)[0],
                        "target": target,
                    }
                )

        seed_keys = [canonical_key(seed) for seed in seeds if canonical_key(seed) in names]
        nodes: dict[str, dict] = {
            key: {"name": names[key], "distance": 0, "score": 1.0, "via": []} for key in seed_keys
        }
        edges: list[dict] = []
        frontier = list(seed_keys)
        for distance in range(1, depth + 1):
            next_frontier: list[str] = []
            for source in frontier:
                candidates = []
                for edge in adjacency.get(source, []):
                    target = edge["target"]
                    if target in seed_keys:
                        continue
                    term_bonus = 0.5 if any(term in target for term in query_terms) else 0.0
                    path_score = nodes[source]["score"] * edge["weight"]
                    candidates.append((path_score + term_bonus * hop_decay(distance), edge, path_score))
                candidates.sort(key=lambda item: -item[0])
                for rank_score, edge, path_score in candidates[: HOP_LIMITS[distance]]:
                    target = edge["target"]
                    score = path_score * hop_decay(distance)
                    known = nodes.get(target)
                    if known is None:
                        nodes[target] = {
                            "name": names.get(target, target),
                            "distance": distance,
                            "score": score,
                            "via": nodes[source]["via"] + [nodes[source]["name"]],
                        }
                        next_frontier.append(target)
                    elif score > known["score"] and known["distance"] >= distance:
                        known["score"] = score
                    edges.append(
                        {
                            "from": edge["from"],
                            "relation": edge["relation"],
                            "to": edge["to"],
                            "distance": distance,
                            "score": round(score, 4),
                            "sentence": edge["sentence"],
                            "chunk_id": edge["chunk_id"],
                        }
                    )
            frontier = next_frontier
            if not frontier:
                break

        unique_edges: dict[tuple, dict] = {}
        for edge in edges:
            key = (edge["from"].lower(), edge["relation"], edge["to"].lower())
            if key not in unique_edges or edge["score"] > unique_edges[key]["score"]:
                unique_edges[key] = edge
        ranked_nodes = sorted(nodes.values(), key=lambda node: (node["distance"], -node["score"]))
        return self._label(f"EXPAND_DEPTH_{depth}"), {
            "nodes": ranked_nodes,
            "edges": sorted(unique_edges.values(), key=lambda edge: (edge["distance"], -edge["score"])),
        }

    def entity_chunks(self, document_id: str) -> dict[str, list[str]]:
        """canonical entity key -> chunk ids that mention it."""
        index: dict[str, list[str]] = defaultdict(list)
        for chunk_id, entity_names in self._load_mentions(document_id).items():
            for name in entity_names:
                key = canonical_key(name)
                if key and chunk_id not in index[key]:
                    index[key].append(chunk_id)
        return index

    def get_graph_data(self, document_id: str, limit: int = 150) -> dict:
        """Best-connected part of the graph for the frontend visualisation."""
        rows = self._load_document_triples(document_id)
        degree: dict[str, int] = defaultdict(int)
        display: dict[str, str] = {}
        for row in rows:
            for name in (row["subject"], row["object"]):
                key = canonical_key(name)
                display.setdefault(key, name)
                degree[key] += 1
        # Keep the hubs: isolated pairs make the picture unreadable.
        max_nodes = max(10, min(limit, 400)) // 2
        keep = set(sorted(degree, key=lambda key: -degree[key])[:max_nodes])
        edges, seen = [], set()
        for row in sorted(rows, key=lambda row: -float(row.get("confidence", 0) or 0)):
            source, target = canonical_key(row["subject"]), canonical_key(row["object"])
            if source in keep and target in keep and (source, target) not in seen:
                seen.add((source, target))
                edges.append({"source": source, "target": target, "relation": row["relation"]})
            if len(edges) >= limit:
                break
        connected = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
        nodes = [{"id": key, "label": display[key], "degree": degree[key]} for key in connected]
        return {
            "nodes": nodes,
            "edges": edges,
            "backend": self.backend,
            "total_entities": len(degree),
            "total_relations": len(rows),
        }

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


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j backend. Relations are stored as typed relationships between Entity nodes."""

    backend = "neo4j"

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import GraphDatabase

        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"), password or os.environ.get("NEO4J_PASSWORD", "")),
            connection_timeout=5,
        )
        self.driver.verify_connectivity()
        self._triple_cache: dict[str, list[dict]] = {}
        self._ensure_schema()

    def _run(self, cypher: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(cypher, params, database_=self.database)
        return [record.data() for record in records]

    def _ensure_schema(self) -> None:
        for statement in (
            "CREATE CONSTRAINT kg_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT kg_entity_uid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE",
            "CREATE CONSTRAINT kg_chunk_uid IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uid IS UNIQUE",
            "CREATE INDEX kg_entity_document IF NOT EXISTS FOR (e:Entity) ON (e.document_id)",
        ):
            self._run(statement)

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def _rel_type(relation: str) -> str:
        value = re.sub(r"[^A-Z0-9_]", "_", relation.upper()).strip("_") or "RELATED"
        if value[0].isdigit():
            value = f"R_{value}"
        return value[:60]

    def store_document(self, document_id: str, file_name: str, summary: str) -> None:
        self._run(
            "MERGE (d:Document {document_id: $document_id}) "
            "SET d.file_name = $file_name, d.summary = $summary, d.updated_at = datetime()",
            document_id=document_id,
            file_name=file_name,
            summary=summary,
        )

    def insert_triples(self, triples: list[dict], document_id: str) -> int:
        rows = self._normalise_rows(triples, document_id)
        if not rows:
            return 0
        by_type: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            row["subject_uid"] = f"{document_id}::{canonical_key(row['subject'])}"
            row["object_uid"] = f"{document_id}::{canonical_key(row['object'])}"
            by_type[self._rel_type(row["relation"])].append(row)
        for rel_type, batch in by_type.items():
            # Relationship types cannot be parameters in Cypher; rel_type is sanitised to [A-Z0-9_].
            self._run(
                "UNWIND $rows AS row "
                "MATCH (d:Document {document_id: $document_id}) "
                "MERGE (s:Entity {uid: row.subject_uid}) "
                "  ON CREATE SET s.name = row.subject, s.document_id = $document_id "
                "MERGE (o:Entity {uid: row.object_uid}) "
                "  ON CREATE SET o.name = row.object, o.document_id = $document_id "
                "MERGE (d)-[:HAS_ENTITY]->(s) MERGE (d)-[:HAS_ENTITY]->(o) "
                f"MERGE (s)-[r:`{rel_type}`]->(o) "
                "SET r.document_id = $document_id, r.relation = row.relation, "
                "    r.confidence = CASE WHEN r.confidence IS NULL OR row.confidence > r.confidence "
                "                        THEN row.confidence ELSE r.confidence END, "
                "    r.chunk_id = row.chunk_id, r.sentence = row.sentence",
                rows=batch,
                document_id=document_id,
            )
        self._triple_cache.pop(document_id, None)
        return len(rows)

    def store_mentions(self, document_id: str, mentions: dict[str, list[str]]) -> int:
        rows = [
            {
                "chunk_uid": f"{document_id}::{chunk_id}",
                "chunk_id": chunk_id,
                "entity_uid": f"{document_id}::{canonical_key(name)}",
                "name": name,
            }
            for chunk_id, names in mentions.items()
            for name in names
            if canonical_key(name)
        ]
        if not rows:
            return 0
        self._run(
            "UNWIND $rows AS row "
            "MATCH (d:Document {document_id: $document_id}) "
            "MERGE (c:Chunk {uid: row.chunk_uid}) ON CREATE SET c.chunk_id = row.chunk_id, c.document_id = $document_id "
            "MERGE (e:Entity {uid: row.entity_uid}) ON CREATE SET e.name = row.name, e.document_id = $document_id "
            "MERGE (d)-[:HAS_CHUNK]->(c) "
            "MERGE (c)-[:MENTIONS]->(e)",
            rows=rows,
            document_id=document_id,
        )
        return len(rows)

    def delete_document(self, document_id: str) -> None:
        self._run(
            "MATCH (n) WHERE (n:Entity OR n:Chunk) AND n.document_id = $document_id DETACH DELETE n",
            document_id=document_id,
        )
        self._run("MATCH (d:Document {document_id: $document_id}) DETACH DELETE d", document_id=document_id)
        self._triple_cache.pop(document_id, None)

    def count_triples(self, document_id: str | None = None) -> int:
        if document_id:
            rows = self._run(
                "MATCH (:Entity {document_id: $document_id})-[r]->(:Entity) RETURN count(r) AS total",
                document_id=document_id,
            )
        else:
            rows = self._run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS total")
        return int(rows[0]["total"]) if rows else 0

    def get_document_summary(self, document_id: str) -> str:
        rows = self._run("MATCH (d:Document {document_id: $document_id}) RETURN d.summary AS summary", document_id=document_id)
        return str(rows[0].get("summary") or "").strip() if rows else ""

    def _load_document_triples(self, document_id: str) -> list[dict]:
        if document_id not in self._triple_cache:
            rows = self._run(
                "MATCH (s:Entity {document_id: $document_id})-[r]->(o:Entity) "
                "RETURN s.name AS subject, coalesce(r.relation, type(r)) AS relation, o.name AS object, "
                "       r.confidence AS confidence, r.chunk_id AS chunk_id, r.sentence AS sentence",
                document_id=document_id,
            )
            for row in rows:
                row["searchable_text"] = f"{row['subject']} {row['relation']} {row['object']}".lower()
                row["confidence"] = row.get("confidence") if row.get("confidence") is not None else DEFAULT_CONFIDENCE
            self._triple_cache[document_id] = rows
        return list(self._triple_cache[document_id])

    def _load_mentions(self, document_id: str) -> dict[str, list[str]]:
        rows = self._run(
            "MATCH (c:Chunk {document_id: $document_id})-[:MENTIONS]->(e:Entity) "
            "RETURN c.chunk_id AS chunk_id, collect(e.name) AS names",
            document_id=document_id,
        )
        return {row["chunk_id"]: row["names"] for row in rows}

    def get_entity_neighborhood(self, entity_name: str, document_id: str, limit: int = 20) -> tuple[str, list[dict]]:
        cypher = (
            "MATCH (c:Entity {uid: $uid})-[r]-(n:Entity) "
            "RETURN c.name AS center_name, startNode(r).name AS from_name, "
            "       coalesce(r.relation, type(r)) AS relation_type, endNode(r).name AS to_name, "
            "       n.name AS neighbor_name "
            "ORDER BY r.confidence DESC LIMIT $limit"
        )
        rows = self._run(cypher, uid=f"{document_id}::{canonical_key(entity_name)}", limit=limit)
        if not rows:
            return super().get_entity_neighborhood(entity_name, document_id, limit)
        for row in rows:
            row["center_name"] = entity_name
        return cypher, rows

    def search_paths(self, terms: list[str], document_id: str, max_hops: int = 2, limit: int = 20) -> tuple[str, list[dict]]:
        cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
        if not cleaned_terms:
            return "NEO4J_PATH_SEARCH", []
        max_hops = max(1, min(3, int(max_hops)))
        cypher = (
            "MATCH (a:Entity {document_id: $document_id}) "
            "WHERE any(term IN $terms WHERE toLower(a.name) CONTAINS term) "
            f"MATCH p = (a)-[*1..{max_hops}]-(b:Entity) "
            "WHERE all(n IN nodes(p) WHERE n:Entity) "
            "RETURN [n IN nodes(p) | n.name] AS path_nodes, "
            "       [r IN relationships(p) | coalesce(r.relation, type(r))] AS path_relationships "
            "LIMIT $limit"
        )
        rows = self._run(cypher, document_id=document_id, terms=cleaned_terms, limit=limit)
        unique, seen = [], set()
        for row in rows:
            token = (tuple(row["path_nodes"]), tuple(row["path_relationships"]))
            if token not in seen and len(set(n.lower() for n in row["path_nodes"])) == len(row["path_nodes"]):
                seen.add(token)
                unique.append(row)
        return cypher, unique


_NEO4J_FAILED_AT = 0.0
_NEO4J_RETRY_SECONDS = 30


def active_backend_name() -> str:
    mode = os.environ.get("KG_GRAPH_BACKEND", "auto").strip().lower()
    if mode == "embedded":
        return "embedded"
    if mode == "neo4j":
        return "neo4j"
    return "neo4j" if os.environ.get("NEO4J_URI") and time.time() - _NEO4J_FAILED_AT > _NEO4J_RETRY_SECONDS else "embedded"


def create_graph_store() -> BaseGraphStore:
    global _NEO4J_FAILED_AT
    mode = os.environ.get("KG_GRAPH_BACKEND", "auto").strip().lower()
    if mode == "embedded":
        return EmbeddedGraphStore()
    if mode == "neo4j":
        return Neo4jGraphStore()
    if os.environ.get("NEO4J_URI") and time.time() - _NEO4J_FAILED_AT > _NEO4J_RETRY_SECONDS:
        try:
            return Neo4jGraphStore()
        except Exception as error:
            _NEO4J_FAILED_AT = time.time()
            print(f"[graph] Neo4j unavailable ({error}); using embedded graph store.", flush=True)
    return EmbeddedGraphStore()


class GraphStore(BaseGraphStore):
    """Factory kept under the old name so existing imports keep working."""

    def __new__(cls, *args, **kwargs):
        return create_graph_store()
