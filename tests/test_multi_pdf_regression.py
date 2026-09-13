import os
import unittest

from fastapi.testclient import TestClient

import app
from kg_app.api import server as app_server
from kg_app.core import query_engine
from kg_app.state import chunk_store

os.environ.setdefault("KG_EMBEDDING_BACKEND", "local")


class FakeGraph:
    stored = {}
    summaries = {}

    def __init__(self):
        self.driver = object()

    def store_document(self, document_id, file_name, summary):
        FakeGraph.summaries[document_id] = {
            "file_name": file_name,
            "summary": summary,
        }

    def insert_triples(self, triples, document_id):
        FakeGraph.stored[document_id] = list(triples)
        return len(triples)

    def count_triples(self, document_id=None):
        if not document_id:
            return sum(len(value) for value in FakeGraph.stored.values())
        return len(FakeGraph.stored.get(document_id, []))

    def search_related(self, terms, document_id, limit=25):
        rows = []
        for triple in FakeGraph.stored.get(document_id, []):
            if any(
                term.lower() in triple["subject"].lower() or term.lower() in triple["object"].lower()
                for term in terms
            ):
                rows.append(
                    {
                        "from_name": triple["subject"],
                        "relation_type": triple["relation"],
                        "to_name": triple["object"],
                    }
                )
        return "MATCH (a)-[r]-(b)", rows[:limit]

    def search_paths(self, terms, document_id, max_hops=2, limit=20):
        triples = FakeGraph.stored.get(document_id, [])
        paths = []
        for first in triples:
            if terms and not any(
                term.lower() in first["subject"].lower() or term.lower() in first["object"].lower()
                for term in terms
            ):
                continue
            for second in triples:
                if first["object"].lower() != second["subject"].lower():
                    continue
                paths.append(
                    {
                        "path_nodes": [
                            first["subject"],
                            first["object"],
                            second["object"],
                        ],
                        "path_relationships": [
                            first["relation"],
                            second["relation"],
                        ],
                    }
                )
        if paths:
            return "MATCH p=(a)-[r*1..2]-(b)", paths[:limit]
        return "MATCH p=(a)-[r*1..2]-(b)", []

    def search_semantic(self, terms, document_id, limit=40):
        rows = []
        for triple in FakeGraph.stored.get(document_id, []):
            searchable = f"{triple['subject']} {triple['relation']} {triple['object']}".lower()
            if any(term.lower() in searchable for term in terms):
                rows.append(
                    {
                        "from_name": triple["subject"],
                        "relation_type": triple["relation"],
                        "to_name": triple["object"],
                        "searchable_text": searchable,
                    }
                )
        return "MATCH (a)-[r]-(b) semantic", rows[:limit]

    def get_graph_overview(self, document_id, limit=12):
        rows = [
            {
                "from_name": triple["subject"],
                "relation_type": triple["relation"],
                "to_name": triple["object"],
            }
            for triple in FakeGraph.stored.get(document_id, [])[:limit]
        ]
        return "MATCH (a)-[r]->(b) overview", rows

    def get_document_summary(self, document_id):
        payload = FakeGraph.summaries.get(document_id, {})
        return payload.get("summary", "")

    def find_relevant_entities(self, terms, document_id, limit=10):
        candidates = []
        seen = set()
        for triple in FakeGraph.stored.get(document_id, []):
            for value in (triple["subject"], triple["object"]):
                lowered = value.lower()
                if lowered in seen:
                    continue
                if any(term.lower() in lowered for term in terms):
                    seen.add(lowered)
                    candidates.append({"entity_name": value})
        if not candidates:
            for triple in FakeGraph.stored.get(document_id, []):
                for value in (triple["subject"], triple["object"]):
                    lowered = value.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    candidates.append({"entity_name": value})
        return "MATCH (d)-[:MENTIONS]->(e)", candidates[:limit]

    def get_entity_neighborhood(self, entity_name, document_id, limit=20):
        rows = []
        lowered = entity_name.lower()
        for triple in FakeGraph.stored.get(document_id, []):
            if triple["subject"].lower() == lowered or triple["object"].lower() == lowered:
                rows.append(
                    {
                        "center_name": entity_name,
                        "from_name": triple["subject"],
                        "relation_type": triple["relation"],
                        "to_name": triple["object"],
                        "neighbor_name": triple["object"] if triple["subject"].lower() == lowered else triple["subject"],
                    }
                )
        return "MATCH (e)-[r]-(other)", rows[:limit]

    def close(self):
        pass


REGRESSION_PDFS = [
    {
        "file_name": "deep-learning-primer.pdf",
        "text": (
            "Deep learning is a subset of machine learning that uses multilayer neural networks to learn complex "
            "patterns from data. Backpropagation is the algorithm used to train multilayer neural networks by "
            "propagating error derivatives backward through the network. Reinforcement learning is learning by "
            "interacting with an environment and receiving rewards. A convolutional neural network is a deep "
            "learning model designed for images. LSTM is a recurrent neural network architecture that captures "
            "long-term dependencies in sequential data."
        ),
        "triples": [
            {"subject": "Deep Learning", "relation": "SUBSET_OF", "object": "Machine Learning"},
            {"subject": "Deep Learning", "relation": "USES", "object": "Multilayer Neural Networks"},
            {"subject": "Backpropagation", "relation": "TRAINS", "object": "Multilayer Neural Networks"},
            {"subject": "Reinforcement Learning", "relation": "LEARNS_FROM", "object": "Rewards"},
            {"subject": "Convolutional Neural Network", "relation": "TYPE_OF", "object": "Deep Learning Model"},
            {"subject": "Convolutional Neural Network", "relation": "DESIGNED_FOR", "object": "Images"},
            {"subject": "LSTM", "relation": "TYPE_OF", "object": "Recurrent Neural Network"},
            {"subject": "LSTM", "relation": "CAPTURES", "object": "Long-term Dependencies"},
        ],
        "cases": [
            {
                "question": "what is deep learning",
                "contains": ["Deep Learning", "machine learning"],
            },
            {
                "question": "what is backpropagation",
                "contains": ["Backpropagation", "train"],
            },
            {
                "question": "what is convolutional neural network",
                "contains": ["Convolutional Neural Network", "images"],
            },
            {
                "question": "what is lstm",
                "contains": ["LSTM", "recurrent neural network"],
            },
            {
                "question": "what is hebbian learning",
                "exact_answer": "It is not in the uploaded document. Please check the text.",
            },
        ],
    },
    {
        "file_name": "constitutional-law-notes.pdf",
        "text": (
            "The Indian Constitution was adopted on 26 November 1949. Dr. B. R. Ambedkar is known as the father "
            "of the Indian Constitution. The Constitution guarantees fundamental rights and separates power among "
            "the legislature, executive, and judiciary."
        ),
        "triples": [
            {"subject": "Indian Constitution", "relation": "ADOPTED_ON", "object": "26 November 1949"},
            {"subject": "Dr. B. R. Ambedkar", "relation": "KNOWN_AS", "object": "Father of the Indian Constitution"},
            {"subject": "Indian Constitution", "relation": "GUARANTEES", "object": "Fundamental Rights"},
            {"subject": "Indian Constitution", "relation": "SEPARATES_INTO", "object": "Legislature"},
            {"subject": "Indian Constitution", "relation": "SEPARATES_INTO", "object": "Executive"},
            {"subject": "Indian Constitution", "relation": "SEPARATES_INTO", "object": "Judiciary"},
        ],
        "cases": [
            {
                "question": "when was the indian constitution adopted",
                "contains": ["26 November 1949"],
            },
            {
                "question": "who is the father of the indian constitution",
                "contains": ["Dr. B. R. Ambedkar"],
            },
            {
                "question": "what does the indian constitution guarantee",
                "contains": ["Fundamental Rights"],
            },
            {
                "question": "what is this pdf about",
                "contains": ["Indian Constitution"],
            },
        ],
    },
    {
        "file_name": "space-company-profile.pdf",
        "text": (
            "SpaceX was founded by Elon Musk. SpaceX is based in the United States. Starship is developed by "
            "SpaceX. Falcon 9 is operated by SpaceX."
        ),
        "triples": [
            {"subject": "SpaceX", "relation": "FOUNDED_BY", "object": "Elon Musk"},
            {"subject": "SpaceX", "relation": "BASED_IN", "object": "United States"},
            {"subject": "Starship", "relation": "DEVELOPED_BY", "object": "SpaceX"},
            {"subject": "Falcon 9", "relation": "OPERATED_BY", "object": "SpaceX"},
        ],
        "cases": [
            {
                "question": "who founded spacex",
                "contains": ["Elon Musk"],
            },
            {
                "question": "where is the company founded by elon musk based",
                "contains": ["United States"],
            },
            {
                "question": "what do you know about spacex",
                "contains": ["SpaceX", "United States"],
            },
        ],
    },
    {
        "file_name": "biology-basics.pdf",
        "text": (
            "Plants perform photosynthesis. Photosynthesis uses sunlight, water, and carbon dioxide. Chlorophyll "
            "absorbs light energy. Mitochondria produce ATP in cells."
        ),
        "triples": [
            {"subject": "Plants", "relation": "PERFORM", "object": "Photosynthesis"},
            {"subject": "Photosynthesis", "relation": "USES", "object": "Sunlight"},
            {"subject": "Photosynthesis", "relation": "USES", "object": "Water"},
            {"subject": "Photosynthesis", "relation": "USES", "object": "Carbon Dioxide"},
            {"subject": "Chlorophyll", "relation": "ABSORBS", "object": "Light Energy"},
            {"subject": "Mitochondria", "relation": "PRODUCE", "object": "ATP"},
        ],
        "cases": [
            {
                "question": "what do plants perform",
                "contains": ["Photosynthesis"],
            },
            {
                "question": "what does photosynthesis use",
                "contains": ["Sunlight"],
            },
            {
                "question": "what absorbs light energy",
                "contains": ["Chlorophyll"],
            },
            {
                "question": "what do mitochondria produce",
                "contains": ["ATP"],
            },
        ],
    },
    {
        "file_name": "world-history-summary.pdf",
        "text": (
            "The French Revolution began in 1789. Bastille Day commemorates the storming of the Bastille. World "
            "War II ended in 1945. The Treaty of Versailles was signed in 1919."
        ),
        "triples": [
            {"subject": "French Revolution", "relation": "BEGAN_IN", "object": "1789"},
            {"subject": "Bastille Day", "relation": "COMMEMORATES", "object": "Storming of the Bastille"},
            {"subject": "World War II", "relation": "ENDED_IN", "object": "1945"},
            {"subject": "Treaty of Versailles", "relation": "SIGNED_IN", "object": "1919"},
        ],
        "cases": [
            {
                "question": "when did the french revolution begin",
                "contains": ["1789"],
            },
            {
                "question": "what does bastille day commemorate",
                "contains": ["Storming of the Bastille"],
            },
            {
                "question": "when did world war ii end",
                "contains": ["1945"],
            },
            {
                "question": "when was the treaty of versailles signed",
                "contains": ["1919"],
            },
        ],
    },
    {
        "file_name": "database-systems-guide.pdf",
        "text": (
            "SQL stores data in tables. An index improves query performance. ACID guarantees consistency and "
            "durability for transactions. PostgreSQL is a relational database system."
        ),
        "triples": [
            {"subject": "SQL", "relation": "STORES", "object": "Data in Tables"},
            {"subject": "Index", "relation": "IMPROVES", "object": "Query Performance"},
            {"subject": "ACID", "relation": "GUARANTEES", "object": "Consistency"},
            {"subject": "ACID", "relation": "GUARANTEES", "object": "Durability"},
            {"subject": "PostgreSQL", "relation": "TYPE_OF", "object": "Relational Database System"},
        ],
        "cases": [
            {
                "question": "where does sql store data",
                "contains": ["tables"],
            },
            {
                "question": "what improves query performance",
                "contains": ["Index"],
            },
            {
                "question": "what does acid guarantee",
                "contains": ["Consistency"],
            },
            {
                "question": "what is postgresql",
                "contains": ["Relational Database System"],
            },
        ],
    },
]


class MultiPdfRegressionTests(unittest.TestCase):
    def setUp(self):
        self.original_hebbrix_native_mode = os.environ.get("HEBBRIX_NATIVE_MODE")
        os.environ["HEBBRIX_NATIVE_MODE"] = "0"
        self.original_graph_app = app_server.GraphStore
        self.original_graph_query = query_engine.GraphStore
        self.original_extract_pages = app_server.extract_pdf_pages
        self.original_extract_text = app_server.extract_text_from_pdf
        self.original_extract_triples = app_server.extract_triples_llm
        self.original_precompute_metadata = app_server.precompute_chunk_metadata
        self.original_set_active_document = app_server.set_active_document
        self.original_get_active_document = app_server.get_active_document

        app_server.GraphStore = FakeGraph
        query_engine.GraphStore = FakeGraph
        app_server.precompute_chunk_metadata = lambda chunks: chunks
        FakeGraph.stored = {}
        FakeGraph.summaries = {}
        chunk_store.clear_all_document_chunks()
        self.active_document = {}
        app_server.set_active_document = self._set_active_document
        app_server.get_active_document = self._get_active_document
        self.client = TestClient(app.app)

    def tearDown(self):
        if self.original_hebbrix_native_mode is None:
            os.environ.pop("HEBBRIX_NATIVE_MODE", None)
        else:
            os.environ["HEBBRIX_NATIVE_MODE"] = self.original_hebbrix_native_mode
        app_server.GraphStore = self.original_graph_app
        query_engine.GraphStore = self.original_graph_query
        app_server.extract_pdf_pages = self.original_extract_pages
        app_server.extract_text_from_pdf = self.original_extract_text
        app_server.extract_triples_llm = self.original_extract_triples
        app_server.precompute_chunk_metadata = self.original_precompute_metadata
        app_server.set_active_document = self.original_set_active_document
        app_server.get_active_document = self.original_get_active_document
        chunk_store.clear_all_document_chunks()

    def _set_active_document(self, document_id, file_name):
        self.active_document = {"document_id": document_id, "file_name": file_name}

    def _get_active_document(self):
        return dict(self.active_document)

    def _upload_fixture(self, fixture):
        app_server.extract_pdf_pages = lambda _: [{"page": 1, "text": fixture["text"]}]
        app_server.extract_text_from_pdf = lambda _: fixture["text"]
        app_server.extract_triples_llm = lambda chunk: list(fixture["triples"])
        response = self.client.post(
            "/upload_pdf",
            files={"file": (fixture["file_name"], b"%PDF", "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(payload.get("triples_added", 0), 0)
        return payload

    def _assert_case(self, fixture_name, case, payload):
        answer = payload.get("answer", "")
        if "exact_answer" in case:
            self.assertEqual(
                answer,
                case["exact_answer"],
                msg=f"{fixture_name} :: {case['question']}",
            )
            return

        lowered_answer = answer.lower()
        for token in case.get("contains", []):
            self.assertIn(
                token.lower(),
                lowered_answer,
                msg=f"{fixture_name} :: {case['question']} :: missing '{token}' in '{answer}'",
            )

    def test_multi_pdf_question_regression_suite(self):
        total_cases = 0
        for fixture in REGRESSION_PDFS:
            with self.subTest(pdf=fixture["file_name"], phase="upload"):
                self._upload_fixture(fixture)
            for case in fixture["cases"]:
                total_cases += 1
                with self.subTest(pdf=fixture["file_name"], question=case["question"]):
                    response = self.client.post("/ask", json={"question": case["question"]})
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self._assert_case(fixture["file_name"], case, payload)

        self.assertGreaterEqual(total_cases, 20)


if __name__ == "__main__":
    unittest.main()
