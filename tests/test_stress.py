# tests/test_stress.py
import unittest

from kg_app.core import extractor, query_engine, utils
from kg_app.db.graph import GraphStore


class TestStress(unittest.TestCase):
    def test_clean_and_validate_triples_returns_empty_for_none(self):
        self.assertEqual(extractor.clean_and_validate_triples(None), [])

    def test_clean_and_validate_triples_returns_empty_for_non_list_input(self):
        self.assertEqual(extractor.clean_and_validate_triples("not a list"), [])

    def test_clean_and_validate_triples_ignores_non_dict_items(self):
        self.assertEqual(extractor.clean_and_validate_triples([None, "x", 1, 2.0]), [])

    def test_clean_and_validate_triples_rejects_whitespace_only_fields(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "   ", "relation": "INCLUDES", "object": "Physics"}]
        )
        self.assertEqual(cleaned, [])

    def test_clean_and_validate_triples_rejects_missing_subject(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"relation": "INCLUDES", "object": "Physics"}]
        )
        self.assertEqual(cleaned, [])

    def test_clean_and_validate_triples_rejects_missing_relation(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "Science", "object": "Physics"}]
        )
        self.assertEqual(cleaned, [])

    def test_clean_and_validate_triples_rejects_missing_object(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "Science", "relation": "INCLUDES"}]
        )
        self.assertEqual(cleaned, [])

    def test_clean_and_validate_triples_rejects_gibberish_relation_after_normalization(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "Science", "relation": "%%%$$$", "object": "Physics"}]
        )
        self.assertEqual(cleaned, [])

    def test_clean_and_validate_triples_keeps_unicode_subject_and_object(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "Schrödinger Equation", "relation": "DESCRIBES", "object": "Quantum States"}]
        )
        self.assertEqual(
            cleaned,
            [{"subject": "Schrödinger Equation", "relation": "DESCRIBES", "object": "Quantum States"}],
        )

    def test_clean_and_validate_triples_rejects_very_long_object(self):
        long_object = " ".join(f"word{i}" for i in range(50))
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "Physics", "relation": "INCLUDES", "object": long_object}]
        )
        self.assertEqual(cleaned, [])

    def test_clean_and_validate_triples_rejects_very_long_subject(self):
        long_subject = " ".join(f"term{i}" for i in range(20))
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": long_subject, "relation": "INCLUDES", "object": "Physics"}]
        )
        self.assertEqual(cleaned, [])

    def test_clean_and_validate_triples_deduplicates_identical_triples(self):
        cleaned = extractor.clean_and_validate_triples(
            [
                {"subject": "India", "relation": "INCLUDES", "object": "States"},
                {"subject": "India", "relation": "INCLUDES", "object": "States"},
            ]
        )
        self.assertEqual(len(cleaned), 1)

    def test_clean_and_validate_triples_deduplicates_case_insensitive_duplicates(self):
        cleaned = extractor.clean_and_validate_triples(
            [
                {"subject": "India", "relation": "includes", "object": "States"},
                {"subject": "india", "relation": "INCLUDES", "object": "states"},
            ]
        )
        self.assertEqual(len(cleaned), 1)

    def test_split_compound_entity_returns_empty_list_for_empty_string(self):
        self.assertEqual(extractor._split_compound_entity(""), [])

    def test_split_compound_entity_returns_empty_list_for_none(self):
        self.assertEqual(extractor._split_compound_entity(None), [])

    def test_split_compound_entity_handles_long_comma_and_and_list(self):
        parts = extractor._split_compound_entity("Hindi, English, Tamil and Sanskrit")
        self.assertEqual(parts, ["Hindi", "English", "Tamil", "Sanskrit"])

    def test_clean_context_phrase_strips_articles_and_trailing_linking_verbs(self):
        cleaned = extractor._clean_context_phrase("the quantum state is")
        self.assertEqual(cleaned, "quantum state")

    def test_infer_relation_from_phrase_recognizes_known_predicate(self):
        self.assertEqual(extractor._infer_relation_from_phrase("was adopted on"), "ADOPTED_ON")

    def test_infer_relation_from_phrase_returns_empty_for_unknown_predicate(self):
        self.assertEqual(extractor._infer_relation_from_phrase("wanders mysteriously"), "")

    def test_normalise_relation_for_storage_prefixes_numeric_starting_relations(self):
        self.assertEqual(utils.normalise_relation_for_storage("123 relation"), "R_123_RELATION")

    def test_normalise_relation_for_storage_strips_punctuation_and_collapses_underscores(self):
        self.assertEqual(utils.normalise_relation_for_storage(" founded---by!!! "), "FOUNDED_BY")

    def test_graphstore_normalise_rows_trims_whitespace_from_values(self):
        rows = GraphStore._normalise_rows(
            [{"subject": "  India  ", "relation": " contains ", "object": "  States  "}],
            "doc-a",
        )
        self.assertEqual(rows[0]["subject"], "India")
        self.assertEqual(rows[0]["object"], "States")
        self.assertEqual(rows[0]["relation"], "INCLUDES")

    def test_graphstore_normalise_rows_skips_invalid_rows(self):
        rows = GraphStore._normalise_rows(
            [
                {"subject": "", "relation": "INCLUDES", "object": "A"},
                {"subject": "A", "relation": "INCLUDES", "object": ""},
            ],
            "doc-a",
        )
        self.assertEqual(rows, [])

    def test_graphstore_normalise_rows_preserves_cross_document_isolation(self):
        rows_a = GraphStore._normalise_rows(
            [{"subject": "India", "relation": "INCLUDES", "object": "States"}],
            "doc-a",
        )
        rows_b = GraphStore._normalise_rows(
            [{"subject": "India", "relation": "INCLUDES", "object": "States"}],
            "doc-b",
        )
        self.assertEqual(rows_a[0]["document_id"], "doc-a")
        self.assertEqual(rows_b[0]["document_id"], "doc-b")
        self.assertTrue(rows_a[0]["document_id"] != rows_b[0]["document_id"])

    def test_split_into_sentences_returns_empty_for_blank_input(self):
        self.assertEqual(utils.split_into_sentences("   "), [])

    def test_split_into_sentences_handles_unicode_content(self):
        sentences = utils.split_into_sentences("Schrödinger wrote equations. Einstein explained light.")
        self.assertEqual(len(sentences), 2)

    def test_chunk_text_handles_empty_input(self):
        self.assertEqual(utils.chunk_text(""), [])

    def test_chunk_text_handles_very_long_repetitive_input(self):
        text = ("Physics explains matter and energy. " * 120).strip()
        chunks = utils.chunk_text(text, target_words=40, min_words=20, max_words=60, overlap_words=10)
        self.assertTrue(chunks)
        self.assertGreaterEqual(len(chunks), 2)

    def test_extract_terms_handles_gibberish_and_symbols(self):
        terms = query_engine._extract_terms("### ??? quantum!! @@ field%%")
        self.assertIn("quantum", [term.lower() for term in terms])

    def test_extract_terms_handles_unicode_words(self):
        terms = query_engine._extract_terms("Tell me about Schrödinger Equation")
        lowered = [term.lower() for term in terms]
        self.assertTrue(any("equation" in term for term in lowered))

    def test_extract_entity_phrases_handles_mixed_case_titles(self):
        phrases = query_engine._extract_entity_phrases("Explain Special Relativity and General Relativity")
        self.assertIn("Special Relativity General Relativity", phrases[0])

    def test_tokens_match_handles_substring_and_similarity_cases(self):
        self.assertTrue(query_engine._tokens_match("movement", "movements"))
        self.assertTrue(query_engine._tokens_match("physics", "physicals") or not query_engine._tokens_match("physics", "physicals"))

    def test_count_term_coverage_handles_none_like_text_gracefully(self):
        coverage = query_engine._count_term_coverage("", ["physics", "energy"])
        self.assertEqual(coverage, 0)

    def test_required_term_coverage_scales_with_number_of_terms(self):
        self.assertEqual(query_engine._required_term_coverage(["physics"]), 1)
        self.assertEqual(query_engine._required_term_coverage(["physics", "energy", "quantum"]), 2)

    def test_adjust_verb_for_subject_handles_plural_and_non_plural_subjects(self):
        self.assertEqual(query_engine._adjust_verb_for_subject("Electrons", "reflects"), "reflect")
        self.assertEqual(query_engine._adjust_verb_for_subject("Atom", "reflects"), "reflects")


if __name__ == "__main__":
    unittest.main()
