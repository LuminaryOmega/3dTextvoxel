"""Deterministic Earth-ECEF voxel binding and guardian enforcement primitives.

This module implements the enforcement rules described in the problem statement,
including quantization, canonical byte formatting, identity hashing, unknown
sigil handling, and guardrails for voxel payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


def round_half_away_from_zero(value: float) -> int:
    """Round-half-away-from-zero with no reliance on language defaults."""
    if value == 0:
        return 0
    sign = 1 if value > 0 else -1
    magnitude = math.floor(abs(value) + 0.5)
    return int(sign * magnitude)


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    earth_model_id: str
    Q_space_m: float
    Q_time: float
    canonicalization_rules_id: str
    schema_version: str

    def validate(self) -> None:
        missing = []
        for field_name in (
            "dataset_id",
            "earth_model_id",
            "Q_space_m",
            "Q_time",
            "canonicalization_rules_id",
            "schema_version",
        ):
            value = getattr(self, field_name)
            if value in (None, ""):
                missing.append(field_name)
        if missing:
            raise ValueError(
                f"Missing required dataset config keys: {', '.join(sorted(missing))}"
            )


def convert_to_ecef(
    coordinates_m: Tuple[float, float, float],
    earth_model_id: str,
    converter: Optional[Mapping[str, Callable[[Tuple[float, float, float]], Tuple[float, float, float]]]] = None,
) -> Tuple[float, float, float]:
    """Convert to ECEF using declared earth_model_id.

    A converter must be supplied by the caller; this module will not guess.
    """
    if converter and earth_model_id in converter:
        return converter[earth_model_id](coordinates_m)
    raise NotImplementedError(
        f"TODO: supply deterministic ECEF conversion for earth_model_id={earth_model_id}"
    )


def quantize_space_ecef(
    coordinates_ecef_m: Tuple[float, float, float], Q_space_m: float
) -> Tuple[int, int, int]:
    x_m, y_m, z_m = coordinates_ecef_m
    return (
        round_half_away_from_zero(x_m / Q_space_m),
        round_half_away_from_zero(y_m / Q_space_m),
        round_half_away_from_zero(z_m / Q_space_m),
    )


def quantize_time(time_value: float, Q_time: float) -> int:
    return round_half_away_from_zero(time_value / Q_time)


@dataclass(frozen=True)
class QuantizedRegion:
    min_x_q: int
    min_y_q: int
    min_z_q: int
    max_x_q: int
    max_y_q: int
    max_z_q: int
    t0_q: int
    t1_q: int

    def as_tuple(self) -> Tuple[int, int, int, int, int, int, int, int]:
        return (
            self.min_x_q,
            self.min_y_q,
            self.min_z_q,
            self.max_x_q,
            self.max_y_q,
            self.max_z_q,
            self.t0_q,
            self.t1_q,
        )

    @staticmethod
    def from_bounds(
        min_coord_ecef_m: Tuple[float, float, float],
        max_coord_ecef_m: Tuple[float, float, float],
        t0: float,
        t1: float,
        Q_space_m: float,
        Q_time: float,
    ) -> "QuantizedRegion":
        min_x, min_y, min_z = quantize_space_ecef(min_coord_ecef_m, Q_space_m)
        max_x, max_y, max_z = quantize_space_ecef(max_coord_ecef_m, Q_space_m)
        return QuantizedRegion(
            min_x, min_y, min_z, max_x, max_y, max_z, quantize_time(t0, Q_time), quantize_time(t1, Q_time)
        )


def _hash_bytes(parts: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _encode_block(records: Iterable[bytes]) -> bytes:
    """Encode a block with bytewise lexicographic sorting and deduplication."""
    unique_sorted = sorted(set(records))
    output = bytearray()
    for rec in unique_sorted:
        output.extend(len(rec).to_bytes(4, "big"))
        output.extend(rec)
    return bytes(output)


@dataclass(frozen=True)
class VocabManifest:
    manifest: Mapping[str, object]
    operator_aliases: Mapping[str, str] = field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def version(self) -> str:
        return _hash_bytes([self.canonical_bytes()])

    def is_operator_known(self, operator: str) -> bool:
        return operator in self.manifest or operator in self.operator_aliases


def build_s1_canonical_bytes(
    vocab_manifest: VocabManifest,
    operators: Sequence[str],
    entities: Sequence[str],
    relations: Sequence[str],
    constraints: Sequence[str],
) -> bytes:
    header = b"S1"
    vocab_block = _encode_block([vocab_manifest.canonical_bytes()])
    operator_block = _encode_block(op.encode("utf-8") for op in operators)
    entity_block = _encode_block(ent.encode("utf-8") for ent in entities)
    relation_block = _encode_block(rel.encode("utf-8") for rel in relations)
    constraint_block = _encode_block(con.encode("utf-8") for con in constraints)
    return b"".join(
        [header, vocab_block, operator_block, entity_block, relation_block, constraint_block]
    )


def _contextual_commitment(context: Mapping[str, object]) -> str:
    safe_context = {k: v for k, v in context.items() if k != "appearance"}
    return _hash_bytes(
        [json.dumps(safe_context, sort_keys=True, separators=(",", ":")).encode("utf-8")]
    )


@dataclass(frozen=True)
class SemanticInput:
    operators: Sequence[str]
    entities: Sequence[Mapping[str, str]]
    relations: Sequence[Tuple[str, ...]]
    constraints: Sequence[str]
    context: Mapping[str, object] = field(default_factory=dict)
    trace_id: Optional[str] = None


@dataclass(frozen=True)
class SemanticSignatureResult:
    pure_bytes: bytes
    unknown_components: Sequence[str]
    context_commitment: str


def _encode_entity(entity: Mapping[str, str]) -> str:
    entity_type = entity.get("type")
    entity_id = entity.get("id")
    return f"{entity_type}:{entity_id}"


def build_semantic_signature(
    semantic_input: SemanticInput, vocab_manifest: VocabManifest
) -> SemanticSignatureResult:
    pure_operators: List[str] = []
    unknown_components: List[str] = []

    for op in semantic_input.operators:
        if vocab_manifest.is_operator_known(op):
            pure_operators.append(op)
        else:
            unknown_components.append(f"operator:{op}")

    pure_entities: List[str] = []
    unknown_entities: Set[str] = set()
    for ent in semantic_input.entities:
        ent_type = ent.get("type")
        ent_id = ent.get("id")
        if ent_type in {"STABLE", "DERIVED"} and ent_id:
            encoded = _encode_entity(ent)
            pure_entities.append(encoded)
        else:
            unknown_entities.add(_encode_entity(ent))
            unknown_components.append(f"entity:{_encode_entity(ent)}")

    pure_relations: List[str] = []
    for rel in semantic_input.relations:
        if any(item in unknown_entities for item in rel):
            unknown_components.append(f"relation:{'|'.join(rel)}")
            continue
        pure_relations.append("|".join(rel))

    pure_constraints: List[str] = []
    for constraint in semantic_input.constraints:
        if constraint:
            pure_constraints.append(constraint)

    pure_bytes = build_s1_canonical_bytes(
        vocab_manifest,
        pure_operators,
        pure_entities,
        pure_relations,
        pure_constraints,
    )

    context_commitment = _contextual_commitment(
        semantic_input.context or {
            "operators": list(semantic_input.operators),
            "entities": [dict(e) for e in semantic_input.entities],
            "relations": [list(r) for r in semantic_input.relations],
            "constraints": list(semantic_input.constraints),
        }
    )

    return SemanticSignatureResult(
        pure_bytes=pure_bytes,
        unknown_components=tuple(sorted(set(unknown_components))),
        context_commitment=context_commitment,
    )


@dataclass(frozen=True)
class VoxelIdentity:
    region_id: str
    semantic_id_pure: str
    voxel_id: str


def compute_voxel_identity(
    dataset_config: DatasetConfig,
    quantized_region: QuantizedRegion,
    vocab_manifest: VocabManifest,
    semantic_signature: SemanticSignatureResult,
) -> VoxelIdentity:
    dataset_config.validate()
    region_id = _hash_bytes(
        [
            dataset_config.schema_version.encode("utf-8"),
            b"EARTH_ECEF",
            dataset_config.earth_model_id.encode("utf-8"),
            str(dataset_config.Q_space_m).encode("utf-8"),
            str(dataset_config.Q_time).encode("utf-8"),
            b"RHAZ",
            json.dumps(quantized_region.as_tuple()).encode("utf-8"),
        ]
    )
    semantic_id_pure = _hash_bytes(
        [
            dataset_config.dataset_id.encode("utf-8"),
            vocab_manifest.version().encode("utf-8"),
            dataset_config.canonicalization_rules_id.encode("utf-8"),
            semantic_signature.pure_bytes,
        ]
    )
    voxel_id = _hash_bytes([region_id.encode("utf-8"), semantic_id_pure.encode("utf-8")])
    return VoxelIdentity(region_id=region_id, semantic_id_pure=semantic_id_pure, voxel_id=voxel_id)


@dataclass
class VoxelRecord:
    identity: VoxelIdentity
    payload: Mapping[str, object]
    semantic_payload: SemanticSignatureResult


class VoxelStore:
    """In-memory append-only voxel store with deterministic indexes."""

    def __init__(self) -> None:
        self._voxels: Dict[str, VoxelRecord] = {}
        self._by_region: Dict[str, Set[str]] = {}
        self._by_semantic: Dict[str, Set[str]] = {}

    def add(self, voxel: VoxelRecord) -> None:
        if voxel.identity.voxel_id in self._voxels:
            return
        self._voxels[voxel.identity.voxel_id] = voxel
        self._by_region.setdefault(voxel.identity.region_id, set()).add(voxel.identity.voxel_id)
        self._by_semantic.setdefault(voxel.identity.semantic_id_pure, set()).add(
            voxel.identity.voxel_id
        )

    def get_by_region(self, region_id: str) -> Sequence[VoxelRecord]:
        return [self._voxels[vid] for vid in sorted(self._by_region.get(region_id, set()))]

    def get_by_semantic(self, semantic_id: str) -> Sequence[VoxelRecord]:
        return [self._voxels[vid] for vid in sorted(self._by_semantic.get(semantic_id, set()))]

    def get(self, voxel_id: str) -> Optional[VoxelRecord]:
        return self._voxels.get(voxel_id)


class SigilWriter:
    """Write pure and unknown sigils deterministically."""

    def __init__(self, pure_path: Optional[str], unknown_path: Optional[str]) -> None:
        if not pure_path or not unknown_path:
            raise ValueError(
                "TODO: configurable sigil output paths must be provided (sigils_pure.<ext>, sigils_unknown.<ext>)"
            )
        self.pure_path = pure_path
        self.unknown_path = unknown_path

    def write_pure(self, voxels: Sequence[VoxelIdentity]) -> None:
        ordered = sorted(voxels, key=lambda v: v.voxel_id)
        records = [
            {
                "region_id": v.region_id,
                "semantic_id_pure": v.semantic_id_pure,
                "voxel_id": v.voxel_id,
            }
            for v in ordered
        ]
        self._write_json_lines(self.pure_path, records)

    def write_unknown(
        self,
        dataset_config: DatasetConfig,
        vocab_manifest: VocabManifest,
        quantized_region: QuantizedRegion,
        signature: SemanticSignatureResult,
        trace_id: Optional[str] = None,
    ) -> None:
        record = {
            "dataset_id": dataset_config.dataset_id,
            "vocab_version": vocab_manifest.version(),
            "region_bounds": quantized_region.as_tuple(),
            "unknown_components": list(signature.unknown_components),
            "context_commitment": signature.context_commitment,
            "trace_id": trace_id,
        }
        self._write_json_lines(self.unknown_path, [record])

    @staticmethod
    def _write_json_lines(path: str, records: Sequence[Mapping[str, object]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")))
                f.write("\n")


class Guardian:
    """Guardian enforcement for voxel creation and mutation."""

    FORBIDDEN_PAYLOAD_KEYS = {"pixels", "frames", "meshes", "textures", "camera", "style", "appearance"}

    def assert_creation_allowed(
        self, payload: Mapping[str, object], operators: Sequence[str], vocab_manifest: VocabManifest
    ) -> None:
        self._assert_payload_allowed(payload)
        unknown_ops = [op for op in operators if not vocab_manifest.is_operator_known(op)]
        if unknown_ops:
            raise PermissionError(f"Unknown operators not permitted for pure creation: {unknown_ops}")

    def assert_mutation_allowed(
        self,
        prior_payload: Mapping[str, object],
        new_payload: Mapping[str, object],
        authorization_token: Optional[str] = None,
    ) -> None:
        if prior_payload != new_payload and not authorization_token:
            raise PermissionError("Mutation altering semantic intent requires explicit authorization token")
        self._assert_payload_allowed(new_payload)

    def assert_overwrite_forbidden(self, authorization_token: Optional[str]) -> None:
        if not authorization_token:
            raise PermissionError("Overwrite forbidden without explicit authorization token")

    def _assert_payload_allowed(self, payload: Mapping[str, object]) -> None:
        forbidden_present = [k for k in payload if k in self.FORBIDDEN_PAYLOAD_KEYS]
        if forbidden_present:
            raise PermissionError(f"Forbidden payload keys present: {forbidden_present}")


class NestBindingAdapter:
    """Append-only nest bindings."""

    def __init__(self) -> None:
        self._nest_to_voxels: Dict[str, Set[str]] = {}
        self._cluster_descriptors: Dict[str, Mapping[str, object]] = {}

    def bind(self, nest_id: str, voxel_ids: Sequence[str], cluster_descriptor: Optional[Mapping[str, object]] = None) -> None:
        self._nest_to_voxels.setdefault(nest_id, set()).update(voxel_ids)
        if cluster_descriptor:
            descriptor_key = self._descriptor_key(cluster_descriptor)
            self._cluster_descriptors[descriptor_key] = cluster_descriptor

    @staticmethod
    def _descriptor_key(descriptor: Mapping[str, object]) -> str:
        filtered = {k: v for k, v in descriptor.items() if k != "appearance"}
        return _hash_bytes([json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")])
