"""Sub-slice 4.2: the `Skill` object and its stored form (docs/graph.md sec 2).

The graph's nodes live in `skill_vocab`; this module is the typed view over
them plus the vector codec. Storage (`hiredra.storage`) only ever sees vectors
as opaque bytes; numpy stays on this side of the boundary.

Vector storage format: float32, explicit little-endian, one flat row, already
L2-normalized by the embedder, so `dim * 4` bytes (3072 at 768 dims) and the
nearest-skill query is a plain dot product.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.storage import iter_skill_vocab, list_skill_aliases, set_skill_vectors


@dataclass(frozen=True)
class Skill:
    """One graph node (docs/graph.md sec 2: name, aliases, mine, level, vector, edges).

    `vector` is None when the node has not been embedded yet OR was embedded
    under a different version tag (stale) -- see `load_skills`. It is excluded
    from equality/repr (ndarray `==` is elementwise, and 768 floats are noise).
    `edges` is always `()` through Phase A; the field exists so Phase B is a
    matter of filling it, not a schema change.
    """

    name: str
    aliases: tuple[str, ...]
    mine: bool
    level: str | None
    vector: np.ndarray | None = field(default=None, compare=False, repr=False)
    edges: tuple[dict, ...] = ()


def vector_to_blob(vector: np.ndarray) -> bytes:
    """One 1-D vector -> little-endian float32 bytes."""
    v = np.asarray(vector, dtype="<f4")
    if v.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {v.shape}")
    return v.tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    """Inverse of `vector_to_blob`. Returns a writable native-float32 copy.
    Raises `ValueError` if the blob isn't exactly `dim` float32s -- a wrong-length
    blob under a matching tag is corruption, never something to guess around."""
    if len(blob) != dim * 4:
        raise ValueError(f"vector blob is {len(blob)} bytes, expected {dim * 4} ({dim} float32)")
    return np.frombuffer(blob, dtype="<f4").astype(np.float32)


def save_vectors(conn, names: Sequence[str], matrix: np.ndarray, embed_tag: str) -> None:
    """Persist `matrix[i]` as the vector of `names[i]`, stamped with `embed_tag`,
    in one transaction (a chunk lands whole or not at all)."""
    if len(names) != len(matrix):
        raise ValueError(f"{len(names)} names but {len(matrix)} vectors")
    set_skill_vectors(conn, [(n, vector_to_blob(v)) for n, v in zip(names, matrix)], embed_tag)


def load_skills(conn, cfg: GraphConfig) -> list[Skill]:
    """Every node as a `Skill`, ordered by name, with its aliases attached.

    A vector is decoded only if the row's `embed_tag` equals `cfg.embed_tag`;
    a row made under any other tag comes back with `vector=None`. So a stale
    index can never leak old-model vectors into a similarity query by accident --
    it just looks un-embedded, which the version gate then repairs.
    """
    aliases = list_skill_aliases(conn)
    skills: list[Skill] = []
    for row in iter_skill_vocab(conn):
        blob = row["vector"]
        current = blob is not None and row["embed_tag"] == cfg.embed_tag
        skills.append(
            Skill(
                name=row["name"],
                aliases=tuple(aliases.get(row["name"], ())),
                mine=bool(row["mine"]),
                level=row["level"],
                vector=blob_to_vector(blob, cfg.dimensions) if current else None,
                edges=tuple(row["edges"] or ()),
            )
        )
    return skills
