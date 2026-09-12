# tests/test_regression.py
import os
import unittest

from kg_app.core import extractor, query_engine, utils
from kg_app.core.extractor import DEBUG_LLM_OUTPUT
from kg_app.db.graph import GraphStore


class TestRegression(unittest.TestCase):
    def test_bug_1_object_word_limit_rejects_eleven_and_keeps_nine(self):
        rejected = extractor.clean_and_validate_triples(
            [
                {
                    "subject": "India",
                    "relation": "INCLUDES",
                    "object": "one two three four five six seven eight nine ten eleven",
                }
            ]
        )
        kept = extractor.clean_and_validate_triples(
            [
                {
                    "subject": "India",
                    "relation": "INCLUDES",
                    "object": "one two three four five six seven eight nine",
                }
            ]
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            kept,
            [
                {
                    "subject": "India",
                    "relation": "INCLUDES",
                    "object": "one two three four five six seven eight nine",
                }
            ],
        )

    def test_bug_2_subject_word_limit_rejects_nine_and_keeps_seven(self):
        rejected = extractor.clean_and_validate_triples(
            [
                {
                    "subject": "one two three four five six seven eight nine",
                    "relation": "INCLUDES",
                    "object": "Physics",
                }
            ]
        )
        kept = extractor.clean_and_validate_triples(
            [
                {
                    "subject": "one two three four five six seven",
                    "relation": "INCLUDES",
                    "object": "Physics",
                }
            ]
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            kept,
            [
                {
                    "subject": "one two three four five six seven",
                    "relation": "INCLUDES",
                    "object": "Physics",
                }
            ],
        )

    def test_bug_3_noisy_relations_are_filtered(self):
        noisy_relations = [
            "RELATED_TO",
            "ASSOCIATED_WITH",
            "PLAYS",
            "PLAYS_ROLE",
            "CONSIDERED",
            "DESCRIBED_AS",
            "CAN_BE",
            "MAY_BE",
            "EXISTS",
            "INVOLVED_IN",
            "LINKED_TO",
        ]
        for relation in noisy_relations:
            cleaned = extractor.clean_and_validate_triples(
                [{"subject": "India", "relation": relation, "object": "Asia"}]
            )
            self.assertEqual(cleaned, [])

    def test_bug_4_invalid_placeholder_objects_are_filtered(self):
        invalid_objects = [
            "various",
            "many",
            "several",
            "things",
            "stuff",
            "examples",
            "type",
            "types",
        ]
        for value in invalid_objects:
            cleaned = extractor.clean_and_validate_triples(
                [{"subject": "India", "relation": "INCLUDES", "object": value}]
            )
            self.assertEqual(cleaned, [])

    def test_bug_5_trailing_article_check_keeps_real_names_and_rejects_literal_a(self):
        valid_names = ["Africa", "India", "America", "Australia", "California"]
        for value in valid_names:
            cleaned = extractor.clean_and_validate_triples(
                [{"subject": "Geography", "relation": "INCLUDES", "object": value}]
            )
            self.assertEqual(
                cleaned,
                [{"subject": "Geography", "relation": "INCLUDES", "object": value}],
            )

        rejected = extractor.clean_and_validate_triples(
            [{"subject": "Geography", "relation": "INCLUDES", "object": "a"}]
        )
        self.assertEqual(rejected, [])

    def test_bug_6_split_compound_entity_keeps_three_word_phrase_and_splits_longer_list(self):
        self.assertEqual(
            extractor._split_compound_entity("Rock and Roll"),
            ["Rock and Roll"],
        )
        self.assertEqual(
            extractor._split_compound_entity("Hindi, English and Sanskrit"),
            ["Hindi", "English", "Sanskrit"],
        )

    def test_bug_7_single_character_entity_fragments_are_dropped(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "A, B and C", "relation": "INCLUDES", "object": "Physics"}]
        )
        for row in cleaned:
            self.assertGreater(len(row["subject"]), 1)

    def test_bug_8_relation_canonicalization_maps_common_variants(self):
        expected = {
            "WAS_FOUNDED_BY": "FOUNDED_BY",
            "COMPRISED_OF": "INCLUDES",
            "ALSO_KNOWN_AS": "KNOWN_AS",
            "WORKS_FOR": "EMPLOYED_BY",
            "IS_LOCATED_IN": "LOCATED_IN",
            "CONSISTS_OF": "INCLUDES",
            "WAS_LOCATED_IN": "LOCATED_IN",
            "IS_FOUNDED_BY": "FOUNDED_BY",
        }
        for source, target in expected.items():
            self.assertEqual(utils.normalise_relation_for_storage(source), target)

    def test_bug_9_acronym_preservation_keeps_uppercase_acronyms(self):
        self.assertEqual(query_engine._clean_entity_text("UNESCO"), "UNESCO")
        self.assertEqual(query_engine._clean_entity_text("UNICEF"), "UNICEF")
        self.assertEqual(query_engine._clean_entity_text("NASA"), "NASA")
        self.assertEqual(query_engine._clean_entity_text("ISRO"), "ISRO")

    def test_bug_10_min_hybrid_score_constant_is_point_18(self):
        self.assertEqual(query_engine.MIN_HYBRID_SCORE, 0.18)

    def test_bug_11_keyword_hybrid_weight_is_point_25(self):
        self.assertEqual(query_engine.HYBRID_WEIGHTS["keyword"], 0.25)

    def test_bug_12_relation_hint_extractor_skips_capitalized_phrase_words(self):
        hints = query_engine._extract_relation_hints("Separation of Powers")
        self.assertNotIn("SEPARATED_FROM", hints)
        self.assertNotIn("DIVIDED_INTO", hints)
        self.assertNotIn("PARTITIONED", hints)

    def test_bug_13_relation_hint_whitelist_words_still_trigger_after_articles(self):
        hints = query_engine._extract_relation_hints("what is the capital of France")
        self.assertTrue("HAS_CAPITAL" in hints or "CAPITAL_OF" in hints or "CAPITAL" in hints)

    def test_bug_14_multi_hop_detection_handles_expanded_question_patterns(self):
        true_questions = [
            "who founded the company",
            "where is the capital of France",
            "what country is Berlin in",
            "when did the war begin",
            "what caused the revolution",
            "who invented the telephone",
            "who wrote the book",
        ]
        for question in true_questions:
            self.assertTrue(query_engine._is_multi_hop_question(question, []))
        self.assertFalse(query_engine._is_multi_hop_question("what is a dog", []))

    def test_bug_15_entity_name_quality_penalizes_long_names_and_rewards_clean_names(self):
        long_name = "this is a very long noisy sentence fragment entity name"
        clean_name = "Albert Einstein"
        self.assertLess(query_engine._entity_name_quality(long_name), 0)
        self.assertGreater(query_engine._entity_name_quality(clean_name), 0)

    def test_bug_16_question_synonyms_do_not_contain_domain_specific_keys(self):
        self.assertNotIn("clothing", query_engine.QUESTION_SYNONYMS)
        self.assertNotIn("food", query_engine.QUESTION_SYNONYMS)
        self.assertNotIn("festival", query_engine.QUESTION_SYNONYMS)
        self.assertNotIn("religion", query_engine.QUESTION_SYNONYMS)
        self.assertNotIn("culture", query_engine.QUESTION_SYNONYMS)

    def test_bug_17_generic_node_patterns_include_common_generic_prompts(self):
        self.assertIn("what is", query_engine.GENERIC_NODE_PATTERNS)
        self.assertIn("describe", query_engine.GENERIC_NODE_PATTERNS)
        self.assertIn("explain", query_engine.GENERIC_NODE_PATTERNS)
        self.assertIn("who was", query_engine.GENERIC_NODE_PATTERNS)

    def test_bug_18_invalid_subject_pronouns_are_filtered(self):
        invalid_subjects = ["it", "they", "this", "that", "these", "those", "he", "she", "we", "i", "you"]
        for subject in invalid_subjects:
            cleaned = extractor.clean_and_validate_triples(
                [{"subject": subject, "relation": "INCLUDES", "object": "Science"}]
            )
            self.assertEqual(cleaned, [])

    def test_bug_19_subject_equal_object_is_filtered(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "India", "relation": "LOCATED_IN", "object": "India"}]
        )
        self.assertEqual(cleaned, [])

    def test_bug_20_relations_with_too_many_parts_are_filtered(self):
        cleaned = extractor.clean_and_validate_triples(
            [{"subject": "Company", "relation": "WAS_VERY_EXTREMELY_FOUNDED_BY", "object": "Founder"}]
        )
        self.assertEqual(cleaned, [])

    def test_bug_21_graphstore_normalise_rows_always_attaches_document_id(self):
        rows = GraphStore._normalise_rows(
            [
                {"subject": "A", "relation": "contains", "object": "B"},
                {"subject": "C", "relation": "works for", "object": "D"},
            ],
            "doc-123",
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["document_id"], "doc-123")

    def test_bug_22_words_ending_in_ics_do_not_get_singularized(self):
        self.assertEqual(query_engine._adjust_verb_for_subject("Physics", "includes"), "includes")

    def test_bug_23_verb_adjustment_never_returns_includ(self):
        result = query_engine._adjust_verb_for_subject("Technologies", "includes")
        self.assertIn(result, {"include", "includes"})
        self.assertNotIn(result, {"includ"})

    @unittest.skipIf("KG_DEBUG_LLM_OUTPUT" in os.environ, "Environment explicitly sets KG_DEBUG_LLM_OUTPUT")
    def test_bug_24_debug_llm_output_default_is_false(self):
        self.assertFalse(DEBUG_LLM_OUTPUT)


if __name__ == "__main__":
    unittest.main()
