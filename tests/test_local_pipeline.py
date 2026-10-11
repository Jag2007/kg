import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from kg_app.api import server as app_server
from kg_app.core import local_extractor, query_engine
from kg_app.db import graph as graph_module
from kg_app.db.graph import EmbeddedGraphStore, canonical_key
from kg_app.state import chunk_store, document_store

COMPANY_TEXT = (
    "Maya Rao founded Helix Labs in 2019. Helix Labs is headquartered in Berlin. "
    "Berlin is the capital of Germany. Arjun Mehta works at Helix Labs. "
    "He leads the Orion project, which runs on the Atlas platform. "
    "The Atlas platform is hosted in the Frankfurt region. "
    "Helix Labs was acquired by Nimbus Corp in 2023."
)


def _triples(text: str) -> set[tuple[str, str, str]]:
    return {
        (t["subject"], t["relation"], t["object"])
        for t in local_extractor.extract_chunk_knowledge(text)["triples"]
    }


@unittest.skipUnless(local_extractor.is_available(), "spaCy model en_core_web_sm is not installed")
class LocalExtractorTests(unittest.TestCase):
    def test_subject_verb_object(self):
        self.assertIn(("Maya Rao", "FOUNDED", "Helix Labs"), _triples("Maya Rao founded Helix Labs."))

    def test_verb_with_preposition(self):
        self.assertIn(("Arjun Mehta", "WORKS_AT", "Helix Labs"), _triples("Arjun Mehta works at Helix Labs."))

    def test_passive_voice_keeps_agent(self):
        self.assertIn(
            ("Helix Labs", "ACQUIRED_BY", "Nimbus Corp"),
            _triples("Helix Labs was acquired by Nimbus Corp in 2023."),
        )

    def test_relative_clause_uses_head_noun(self):
        triples = _triples("He leads the Orion project, which runs on the Atlas platform.")
        self.assertIn(("Orion project", "RUNS_ON", "Atlas platform"), triples)

    def test_pronoun_takes_previous_subject(self):
        triples = _triples("Arjun Mehta works at Helix Labs. He leads the Orion project.")
        self.assertIn(("Arjun Mehta", "LEADS", "Orion project"), triples)

    def test_negated_statement_is_skipped(self):
        triples = _triples("Backend A does not use Redis.")
        self.assertFalse(any(t[2] == "Redis" for t in triples))

    def test_numeric_identifiers_stay_attached(self):
        self.assertIn(("Backend A", "RUNS_ON", "port 3001"), _triples("Backend A runs on port 3001."))

    def test_person_name_is_not_used_as_relation(self):
        triples = _triples("Step 3 is the DNS server (Dhanvin on Mac 1).")
        self.assertFalse(any(t[1].startswith("DHANVIN") for t in triples))

    def test_chunk_reports_entity_mentions(self):
        entities = local_extractor.extract_chunk_knowledge(COMPANY_TEXT)["entities"]
        for name in ("Maya Rao", "Helix Labs", "Berlin", "Atlas platform"):
            self.assertIn(name, entities)


class EmbeddedGraphStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EmbeddedGraphStore(path=Path(self.tmp.name) / "graph.json")
        self.store.store_document("doc", "company.pdf", "summary")
        self.store.insert_triples(
            [
                {"subject": "Arjun Mehta", "relation": "WORKS_AT", "object": "Helix Labs", "confidence": 0.8},
                {"subject": "Helix Labs", "relation": "HEADQUARTERED_IN", "object": "Berlin", "confidence": 0.8},
                {"subject": "Berlin", "relation": "CAPITAL_OF", "object": "Germany", "confidence": 0.75},
                {"subject": "Maya Rao", "relation": "FOUNDED", "object": "Helix Labs", "confidence": 0.8},
            ],
            "doc",
        )
        self.store.store_mentions("doc", {"chunk_1": ["Arjun Mehta", "Helix Labs"], "chunk_2": ["Helix Labs", "Berlin"]})

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_and_summary(self):
        self.assertEqual(self.store.count_triples("doc"), 4)
        self.assertEqual(self.store.get_document_summary("doc"), "summary")

    def test_reinserting_triples_does_not_duplicate(self):
        self.store.insert_triples([{"subject": "Arjun Mehta", "relation": "WORKS_AT", "object": "Helix Labs"}], "doc")
        self.assertEqual(self.store.count_triples("doc"), 4)

    def test_expansion_reaches_two_hops_with_decayed_score(self):
        _, result = self.store.expand_entities(["Arjun Mehta"], "doc", depth=2)
        nodes = {node["name"]: node for node in result["nodes"]}
        self.assertEqual(nodes["Arjun Mehta"]["distance"], 0)
        self.assertEqual(nodes["Helix Labs"]["distance"], 1)
        self.assertEqual(nodes["Berlin"]["distance"], 2)
        self.assertEqual(nodes["Berlin"]["via"], ["Arjun Mehta", "Helix Labs"])
        self.assertLess(nodes["Berlin"]["score"], nodes["Helix Labs"]["score"])

    def test_depth_one_stops_at_direct_neighbours(self):
        _, result = self.store.expand_entities(["Arjun Mehta"], "doc", depth=1)
        self.assertNotIn("Berlin", {node["name"] for node in result["nodes"]})

    def test_entity_chunk_index_uses_canonical_keys(self):
        index = self.store.entity_chunks("doc")
        self.assertEqual(sorted(index[canonical_key("Helix Labs")]), ["chunk_1", "chunk_2"])

    def test_delete_document_removes_everything(self):
        self.store.delete_document("doc")
        self.assertEqual(self.store.count_triples("doc"), 0)
        self.assertEqual(self.store.entity_chunks("doc"), {})

    def test_graph_data_for_visualisation(self):
        data = self.store.get_graph_data("doc")
        self.assertEqual(data["backend"], "embedded")
        self.assertEqual(len(data["edges"]), 4)
        self.assertIn("helix labs", {node["id"] for node in data["nodes"]})


def _make_pdf(text: str) -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    return document.tobytes()


@unittest.skipUnless(local_extractor.is_available(), "spaCy model en_core_web_sm is not installed")
class LocalPipelineEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {key: os.environ.get(key) for key in ("KG_EXTRACTOR", "KG_ANSWER_LLM", "KG_GRAPH_BACKEND")}
        os.environ.update(KG_EXTRACTOR="spacy", KG_ANSWER_LLM="none", KG_GRAPH_BACKEND="embedded")
        self.paths = (graph_module.EMBEDDED_PATH, chunk_store.STATE_PATH, document_store.STATE_PATH)
        graph_module.EMBEDDED_PATH = Path(self.tmp.name) / "graph.json"
        chunk_store.STATE_PATH = Path(self.tmp.name) / "chunks.json"
        document_store.STATE_PATH = Path(self.tmp.name) / "active.json"
        self.client = TestClient(app_server.app)

    def tearDown(self):
        graph_module.EMBEDDED_PATH, chunk_store.STATE_PATH, document_store.STATE_PATH = self.paths
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def _upload(self):
        response = self.client.post(
            "/upload_pdf",
            files={"file": ("company.pdf", _make_pdf(COMPANY_TEXT), "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("error", payload)
        return payload

    def test_upload_builds_graph_with_mentions(self):
        payload = self._upload()
        self.assertGreaterEqual(payload["triples_added"], 6)
        self.assertEqual(payload["debug"]["extractor"], "spacy")
        self.assertEqual(payload["debug"]["graph_backend"], "embedded")
        self.assertGreater(payload["debug"]["entity_mentions"], 0)

    def test_single_hop_question(self):
        self._upload()
        answer = self.client.post("/ask", json={"question": "Who founded Helix Labs?"}).json()["answer"]
        self.assertIn("Maya Rao", answer)

    def test_multi_hop_question_returns_graph_evidence(self):
        self._upload()
        payload = self.client.post(
            "/ask", json={"question": "Where is the company that Arjun Mehta works at headquartered?"}
        ).json()
        evidence = payload["graph_evidence"]
        self.assertIn("Arjun Mehta", evidence["anchors"])
        reached = {node["name"]: node["distance"] for node in evidence["nodes"]}
        # One-page PDF: Berlin is reachable by the typed path (2 hops) or a same-chunk co-mention (1 hop).
        self.assertIn(reached.get("Berlin"), {1, 2})
        self.assertIn("Berlin", payload["answer"])

    def test_chain_answer_without_llm(self):
        evidence = {
            "anchors": ["Arjun Mehta"],
            "nodes": [
                {"name": "Arjun Mehta", "distance": 0, "score": 1.0, "via": []},
                {"name": "Helix Labs", "distance": 1, "score": 0.8, "via": ["Arjun Mehta"]},
                {"name": "Berlin", "distance": 2, "score": 0.5, "via": ["Arjun Mehta", "Helix Labs"]},
            ],
            "edges": [
                {"from": "Arjun Mehta", "relation": "WORKS_AT", "to": "Helix Labs", "distance": 1},
                {"from": "Helix Labs", "relation": "HEADQUARTERED_IN", "to": "Berlin", "distance": 2},
            ],
        }
        answer = query_engine._chain_answer("Where is Arjun Mehta based?", evidence)
        self.assertEqual(answer, "Arjun Mehta works at Helix Labs. Helix Labs headquartered in Berlin.")

    def test_health_and_graph_endpoints(self):
        self._upload()
        health = self.client.get("/health").json()
        self.assertEqual(health["graph_backend"], "embedded")
        self.assertTrue(health["spacy_ready"])
        graph = self.client.get("/graph").json()
        self.assertGreater(len(graph["nodes"]), 3)
        self.assertEqual(graph["file_name"], "company.pdf")


if __name__ == "__main__":
    unittest.main()
