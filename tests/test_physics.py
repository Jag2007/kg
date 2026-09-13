# tests/test_physics.py
import unittest

from kg_app.core import extractor, query_engine, utils


PHYSICS_TRIPLES = [
    {"subject": "Bohr Model", "relation": "EXPLAINS", "object": "Hydrogen Spectrum"},
    {"subject": "Bohr Model", "relation": "INTRODUCES", "object": "Quantized Orbits"},
    {"subject": "Electrons", "relation": "OCCUPY", "object": "Energy Levels"},
    {"subject": "Electron Transitions", "relation": "PRODUCE", "object": "Spectral Lines"},
    {"subject": "Balmer Series", "relation": "INCLUDES", "object": "Visible Lines"},
    {"subject": "Balmer Series", "relation": "CORRESPONDS_TO", "object": "Level n 2"},
    {"subject": "Lyman Series", "relation": "INCLUDES", "object": "Ultraviolet Lines"},
    {"subject": "Lyman Series", "relation": "CORRESPONDS_TO", "object": "Level n 1"},
    {"subject": "Paschen Series", "relation": "INCLUDES", "object": "Infrared Lines"},
    {"subject": "Hydrogen Atom", "relation": "CONTAINS", "object": "One Electron"},
    {"subject": "Albert Einstein", "relation": "EXPLAINS", "object": "Photoelectric Effect"},
    {"subject": "Albert Einstein", "relation": "PROPOSED", "object": "Special Relativity"},
    {"subject": "Albert Einstein", "relation": "PROPOSED", "object": "General Relativity"},
    {"subject": "Albert Einstein", "relation": "WON", "object": "Nobel Prize"},
    {"subject": "Photoelectric Effect", "relation": "DEPENDS_ON", "object": "Threshold Frequency"},
    {"subject": "Photoelectric Effect", "relation": "EMITS", "object": "Electrons"},
    {"subject": "Photoelectric Effect", "relation": "SUPPORTS", "object": "Quantum Theory"},
    {"subject": "Planck Constant", "relation": "RELATES", "object": "Energy Frequency"},
    {"subject": "Planck Constant", "relation": "SYMBOL_IS", "object": "Constant h"},
    {"subject": "Wave Particle Duality", "relation": "DESCRIBES", "object": "Light"},
    {"subject": "Wave Particle Duality", "relation": "DESCRIBES", "object": "Matter"},
    {"subject": "De Broglie", "relation": "PROPOSED", "object": "Matter Waves"},
    {"subject": "De Broglie", "relation": "RELATES", "object": "Momentum Wavelength"},
    {"subject": "Schrodinger Equation", "relation": "DESCRIBES", "object": "Quantum States"},
    {"subject": "Schrodinger Equation", "relation": "PREDICTS", "object": "Probability Density"},
    {"subject": "Heisenberg Uncertainty Principle", "relation": "LIMITS", "object": "Measurement Precision"},
    {"subject": "Heisenberg Uncertainty Principle", "relation": "RELATES", "object": "Position Momentum"},
    {"subject": "General Relativity", "relation": "DESCRIBES", "object": "Gravity"},
    {"subject": "General Relativity", "relation": "PREDICTS", "object": "Spacetime Curvature"},
    {"subject": "Special Relativity", "relation": "RELATES", "object": "Space Time"},
    {"subject": "Special Relativity", "relation": "LIMITS", "object": "Speed of Light"},
    {"subject": "Speed of Light", "relation": "APPROXIMATES", "object": "300000 km/s"},
    {"subject": "Atomic Spectral Lines", "relation": "REVEAL", "object": "Energy Levels"},
    {"subject": "Emission Spectrum", "relation": "SHOWS", "object": "Bright Lines"},
    {"subject": "Absorption Spectrum", "relation": "SHOWS", "object": "Dark Lines"},
    {"subject": "Rutherford Model", "relation": "PRECEDES", "object": "Bohr Model"},
    {"subject": "Bohr Model", "relation": "PRECEDES", "object": "Quantum Mechanics"},
    {"subject": "Quantum Mechanics", "relation": "INCLUDES", "object": "Probability Waves"},
    {"subject": "Photon", "relation": "CARRIES", "object": "Light Energy"},
    {"subject": "Photon", "relation": "POSSESSES", "object": "Zero Rest Mass"},
    {"subject": "Photon", "relation": "EXHIBITS", "object": "Wave Nature"},
    {"subject": "Photon", "relation": "EXHIBITS", "object": "Particle Nature"},
    {"subject": "Nuclear Atom", "relation": "CONTAINS", "object": "Dense Nucleus"},
    {"subject": "Nucleus", "relation": "CONTAINS", "object": "Protons"},
    {"subject": "Nucleus", "relation": "CONTAINS", "object": "Neutrons"},
    {"subject": "Spectral Lines", "relation": "IDENTIFY", "object": "Elements"},
]


class TestPhysics(unittest.TestCase):
    def test_all_physics_triples_pass_validation(self):
        cleaned = extractor.clean_and_validate_triples(PHYSICS_TRIPLES)
        self.assertEqual(len(cleaned), 46)

    def test_bohr_model_triples_survive_validation(self):
        cleaned = extractor.clean_and_validate_triples(PHYSICS_TRIPLES)
        bohr_rows = [row for row in cleaned if row["subject"] == "Bohr Model"]
        self.assertGreaterEqual(len(bohr_rows), 3)
        objects = [row["object"] for row in bohr_rows]
        self.assertIn("Hydrogen Spectrum", objects)
        self.assertIn("Quantized Orbits", objects)

    def test_balmer_series_is_not_confused_with_lyman_series(self):
        cleaned = extractor.clean_and_validate_triples(PHYSICS_TRIPLES)
        balmer_objects = [row["object"] for row in cleaned if row["subject"] == "Balmer Series"]
        lyman_objects = [row["object"] for row in cleaned if row["subject"] == "Lyman Series"]
        self.assertIn("Visible Lines", balmer_objects)
        self.assertIn("Ultraviolet Lines", lyman_objects)
        self.assertNotIn("Ultraviolet Lines", balmer_objects)
        self.assertNotIn("Visible Lines", lyman_objects)

    def test_einstein_triple_count_is_four(self):
        cleaned = extractor.clean_and_validate_triples(PHYSICS_TRIPLES)
        einstein_rows = [row for row in cleaned if row["subject"] == "Albert Einstein"]
        self.assertEqual(len(einstein_rows), 4)

    def test_relation_canonicalization_handles_physics_style_relation_variants(self):
        self.assertEqual(utils.normalise_relation_for_storage("contains"), "INCLUDES")
        self.assertEqual(utils.normalise_relation_for_storage("composed of"), "INCLUDES")
        self.assertEqual(utils.normalise_relation_for_storage("is located in"), "LOCATED_IN")

    def test_acronym_preservation_for_physics_related_acronyms(self):
        self.assertEqual(query_engine._clean_entity_text("NASA"), "NASA")
        self.assertEqual(query_engine._clean_entity_text("ISRO"), "ISRO")
        self.assertEqual(query_engine._clean_entity_text("CERN"), "CERN")

    def test_entity_quality_scores_for_clean_physics_names_are_positive(self):
        self.assertGreater(query_engine._entity_name_quality("Bohr Model"), 0)
        self.assertGreater(query_engine._entity_name_quality("Planck Constant"), 0)
        self.assertGreater(query_engine._entity_name_quality("Schrodinger Equation"), 0)

    def test_multi_hop_detection_for_physics_questions(self):
        self.assertTrue(query_engine._is_multi_hop_question("who discovered the photoelectric effect", []))
        self.assertTrue(query_engine._is_multi_hop_question("who wrote the theory", []))
        self.assertFalse(query_engine._is_multi_hop_question("what is a photon", []))

    def test_relation_hints_for_physics_questions(self):
        discover_hints = query_engine._extract_relation_hints("who discovered the photoelectric effect")
        author_hints = query_engine._extract_relation_hints("who wrote the paper")
        self.assertIn("DISCOVERED_BY", discover_hints)
        self.assertIn("WRITTEN_BY", author_hints)

    def test_entity_name_quality_prefers_short_clean_names_over_noisy_fragments(self):
        clean_score = query_engine._entity_name_quality("Albert Einstein")
        noisy_score = query_engine._entity_name_quality(
            "the long descriptive fragment about the atomic spectral observation process"
        )
        self.assertGreater(clean_score, noisy_score)


if __name__ == "__main__":
    unittest.main()
