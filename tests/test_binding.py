import json
import os
import tempfile
import unittest

from textvoxel.binding import (
    DatasetConfig,
    SemanticInput,
    VocabManifest,
    build_semantic_signature,
    build_s1_canonical_bytes,
    compute_voxel_identity,
    quantize_time,
    quantize_space_ecef,
    round_half_away_from_zero,
    SigilWriter,
    QuantizedRegion,
)


class RoundHalfAwayFromZeroTests(unittest.TestCase):
    def test_rounding_behavior(self):
        self.assertEqual(round_half_away_from_zero(1.2), 1)
        self.assertEqual(round_half_away_from_zero(1.5), 2)
        self.assertEqual(round_half_away_from_zero(-1.2), -1)
        self.assertEqual(round_half_away_from_zero(-1.5), -2)
        self.assertEqual(round_half_away_from_zero(0.0), 0)


class ConfigValidationTests(unittest.TestCase):
    def test_missing_config_raises(self):
        config = DatasetConfig(
            dataset_id="",
            earth_model_id="WGS84",
            Q_space_m=1.0,
            Q_time=1.0,
            canonicalization_rules_id="rules",
            schema_version="voxel-bind-v1",
        )
        with self.assertRaises(ValueError):
            config.validate()


class SemanticSignatureTests(unittest.TestCase):
    def setUp(self):
        self.vocab = VocabManifest({"opA": {}}, operator_aliases={"opAlias": "opA"})

    def test_unknown_operator_emits_unknown_sigil(self):
        semantic_input = SemanticInput(
            operators=["opA", "opUnknown"],
            entities=[{"type": "STABLE", "id": "e1"}],
            relations=[("STABLE:e1", "STABLE:e2")],
            constraints=[],
            context={"appearance": "ignore", "note": "keep"},
            trace_id="trace123",
        )
        signature = build_semantic_signature(semantic_input, self.vocab)
        self.assertIn("operator:opUnknown", signature.unknown_components)
        self.assertTrue(signature.context_commitment)

    def test_s1_sorting_and_dedup(self):
        canonical = build_s1_canonical_bytes(
            self.vocab,
            operators=["beta", "alpha", "alpha"],
            entities=["STABLE:e2", "STABLE:e1"],
            relations=["r2", "r1", "r1"],
            constraints=["c2", "c1", "c1"],
        )
        # After header, operator block should list alpha then beta
        # Each record is 4 byte length + bytes; check substring order.
        header = canonical[:2]
        self.assertEqual(header, b"S1")
        remainder = canonical[2:]
        # Decode first block length (vocab), then operators block starts.
        vocab_len = int.from_bytes(remainder[:4], "big")
        operator_block_offset = 4 + vocab_len
        operator_len_first = int.from_bytes(
            remainder[operator_block_offset : operator_block_offset + 4], "big"
        )
        first_operator = remainder[
            operator_block_offset + 4 : operator_block_offset + 4 + operator_len_first
        ]
        self.assertEqual(first_operator, b"alpha")


class IdentityComputationTests(unittest.TestCase):
    def setUp(self):
        self.vocab = VocabManifest({"opA": {}})
        semantic_input = SemanticInput(
            operators=["opA"],
            entities=[{"type": "STABLE", "id": "e1"}],
            relations=[],
            constraints=[],
        )
        self.signature = build_semantic_signature(semantic_input, self.vocab)
        self.config = DatasetConfig(
            dataset_id="dataset",
            earth_model_id="WGS84",
            Q_space_m=1.0,
            Q_time=1.0,
            canonicalization_rules_id="rules",
            schema_version="voxel-bind-v1",
        )
        self.region = QuantizedRegion.from_bounds((0, 0, 0), (1, 1, 1), 0.0, 1.0, 1.0, 1.0)

    def test_voxel_identity_determinism(self):
        identity = compute_voxel_identity(
            self.config, self.region, self.vocab, self.signature
        )
        identity_2 = compute_voxel_identity(
            self.config, self.region, self.vocab, self.signature
        )
        self.assertEqual(identity.voxel_id, identity_2.voxel_id)


class SigilWriterTests(unittest.TestCase):
    def test_unknown_sigil_written(self):
        vocab = VocabManifest({"opA": {}})
        config = DatasetConfig(
            dataset_id="dataset",
            earth_model_id="WGS84",
            Q_space_m=1.0,
            Q_time=1.0,
            canonicalization_rules_id="rules",
            schema_version="voxel-bind-v1",
        )
        signature = build_semantic_signature(
            SemanticInput(
                operators=["opUnknown"],
                entities=[{"type": "UNKNOWN", "id": "eX"}],
                relations=[],
                constraints=[],
            ),
            vocab,
        )
        region = QuantizedRegion.from_bounds((0, 0, 0), (0, 0, 0), 0, 0, 1.0, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            pure_path = os.path.join(tmpdir, "sigils_pure.jsonl")
            unknown_path = os.path.join(tmpdir, "sigils_unknown.jsonl")
            writer = SigilWriter(pure_path, unknown_path)
            writer.write_unknown(config, vocab, region, signature, trace_id="trace1")
            with open(unknown_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["dataset_id"], "dataset")
        self.assertIn("operator:opUnknown", parsed["unknown_components"])


class QuantizationTests(unittest.TestCase):
    def test_quantization_to_ints(self):
        coords = (1.1, -1.1, 0.4)
        q = quantize_space_ecef(coords, 0.5)
        self.assertEqual(q, (2, -2, 1))
        self.assertEqual(quantize_time(1.2, 0.5), 2)


if __name__ == "__main__":
    unittest.main()
