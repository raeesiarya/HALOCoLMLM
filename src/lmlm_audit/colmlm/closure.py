from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from lmlm_audit.colmlm.answers import _default_support_judge
from lmlm_audit.colmlm.errors import CoLMLMIntegrationError
from lmlm_audit.colmlm.index_filter import (
    _candidate_id,
    _candidate_score,
    _candidate_source_id,
    _candidate_text,
)
from lmlm_audit.core.examples import AuditExample, DeletionManifest

VALID_PREDICATES = ("geometric", "semantic", "provenance")


@dataclass(frozen=True)
class ClosureConfig:
    predicates: tuple[str, ...] = VALID_PREDICATES
    radius: float = 0.85
    envelope_top_k: int = 500
    max_closure_size: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "predicates", tuple(dict.fromkeys(self.predicates))
        )
        unknown = [
            predicate
            for predicate in self.predicates
            if predicate not in VALID_PREDICATES
        ]
        if unknown:
            raise ValueError(
                f"Unknown closure predicates {unknown!r}; "
                f"valid predicates are {list(VALID_PREDICATES)!r}."
            )
        if not self.predicates:
            raise ValueError("At least one closure predicate is required.")
        if not -1.0 <= self.radius <= 1.0:
            raise ValueError("radius must be a cosine similarity in [-1, 1].")
        if self.envelope_top_k < 1:
            raise ValueError("envelope_top_k must be at least 1.")
        if self.max_closure_size < 1:
            raise ValueError("max_closure_size must be at least 1.")

    def is_active(self, predicate: str) -> bool:
        return predicate in self.predicates


@dataclass(frozen=True)
class ClosureEntry:
    entry_id: str
    score: float | None
    source_id: str | None
    value: str
    caught_by: tuple[str, ...]


@dataclass(frozen=True)
class ClosureResult:
    example_key: str
    entries: tuple[ClosureEntry, ...]
    source_ids: tuple[str, ...]
    truncated: bool
    config: ClosureConfig
    index_nprobe: int | None = None

    def to_manifest(self) -> DeletionManifest:
        entry_counts = {
            predicate: sum(
                1 for entry in self.entries if predicate in entry.caught_by
            )
            for predicate in (*VALID_PREDICATES, "oracle")
        }
        # Per-entry attribution stays out of the manifest (it is embedded in
        # every JSONL result row); the full listing goes to the closure
        # artifact instead.
        return DeletionManifest(
            entry_ids=tuple(entry.entry_id for entry in self.entries),
            source_ids=self.source_ids,
            strategy="closure",
            metadata={
                "predicates_active": list(self.config.predicates),
                "radius": self.config.radius,
                "envelope_top_k": self.config.envelope_top_k,
                "max_closure_size": self.config.max_closure_size,
                "truncated": self.truncated,
                "entry_counts": entry_counts,
                "index_nprobe": self.index_nprobe,
            },
        )

    def as_dict(self) -> dict[str, Any]:
        manifest = self.to_manifest()
        return {
            "example_key": self.example_key,
            "manifest_id": manifest.manifest_id,
            "strategy": "closure",
            "predicates_active": list(self.config.predicates),
            "radius": self.config.radius,
            "envelope_top_k": self.config.envelope_top_k,
            "max_closure_size": self.config.max_closure_size,
            "truncated": self.truncated,
            "index_nprobe": self.index_nprobe,
            "source_ids": list(self.source_ids),
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "score": entry.score,
                    "source_id": entry.source_id,
                    "value": entry.value,
                    "caught_by": list(entry.caught_by),
                }
                for entry in self.entries
            ],
        }


def _flatten_single_query(raw: Any) -> list[Any]:
    if raw and isinstance(raw, list) and isinstance(raw[0], list):
        if len(raw) != 1:
            raise CoLMLMIntegrationError(
                "The closure builder issued a single query but the index "
                f"returned {len(raw)} result lists."
            )
        return list(raw[0])
    return list(raw or [])


def _index_nprobe(index: Any) -> int | None:
    config = getattr(index, "config", None)
    value = getattr(config, "ivf_nprobe", None)
    return int(value) if isinstance(value, int) else None


def build_closure(
    *,
    index: Any,
    example: AuditExample,
    query_vector: Any | None,
    config: ClosureConfig,
    seed_candidates: tuple[Any, ...] = (),
    seed_source_ids: tuple[str, ...] = (),
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]] = (
        _default_support_judge
    ),
    example_key: str = "",
) -> ClosureResult:
    needs_search = config.is_active("geometric") or config.is_active("semantic")
    if needs_search and query_vector is None:
        raise ValueError(
            "Geometric/semantic closure predicates need the FULL query "
            "embedding, but none was captured for this fact."
        )

    caught: dict[str, set[str]] = {}
    details: dict[str, dict[str, Any]] = {}

    def note(candidate: Any, predicate: str) -> None:
        entry_id = _candidate_id(candidate)
        if not entry_id:
            return
        caught.setdefault(entry_id, set()).add(predicate)
        details.setdefault(
            entry_id,
            {
                "score": _candidate_score(candidate),
                "source_id": _candidate_source_id(candidate),
                "value": _candidate_text(candidate),
            },
        )

    query = (
        np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if needs_search
        else None
    )

    truncated = False
    if config.is_active("geometric"):
        hits = _flatten_single_query(
            index.search(
                query,
                top_k=config.max_closure_size,
                similarity_threshold=config.radius,
            )
        )
        # A full page means the radius may hold more entries than we fetched;
        # the closure must never silently read as complete.
        truncated = len(hits) >= config.max_closure_size
        for candidate in hits:
            note(candidate, "geometric")

    envelope: list[Any] = []
    if config.is_active("semantic"):
        envelope = _flatten_single_query(
            index.search(
                query,
                top_k=config.envelope_top_k,
                similarity_threshold=None,
            )
        )
        for candidate in envelope:
            if dict(support_judge(candidate, example)).get("supports_target"):
                note(candidate, "semantic")

    source_ids = (
        tuple(dict.fromkeys(str(value) for value in seed_source_ids))
        if config.is_active("provenance")
        else ()
    )
    if source_ids:
        # Attribution only: the actual provenance exclusion happens by
        # source ID at search time, unbounded by what we saw here.
        for candidate in envelope:
            candidate_source = _candidate_source_id(candidate)
            if candidate_source is not None and candidate_source in source_ids:
                note(candidate, "provenance")

    for candidate in seed_candidates:
        note(candidate, "oracle")

    def sort_key(entry_id: str) -> tuple[int, float, str]:
        score = details[entry_id]["score"]
        return (0 if score is not None else 1, -(score or 0.0), entry_id)

    entries = tuple(
        ClosureEntry(
            entry_id=entry_id,
            score=details[entry_id]["score"],
            source_id=details[entry_id]["source_id"],
            value=details[entry_id]["value"],
            caught_by=tuple(sorted(caught[entry_id])),
        )
        for entry_id in sorted(caught, key=sort_key)
    )
    return ClosureResult(
        example_key=example_key,
        entries=entries,
        source_ids=source_ids,
        truncated=truncated,
        config=config,
        index_nprobe=_index_nprobe(index),
    )


def _example_key(example: AuditExample) -> str:
    for value in (example.prompt_id, example.fact_id):
        if value is not None:
            return str(value)
    return "fact"


def _safe_filename(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", key)


def write_closure_artifact(closure: ClosureResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(closure.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_closure_manifest_from_full(
    *,
    index: Any,
    example: AuditExample,
    full_result: Mapping[str, Any],
    config: ClosureConfig,
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]] = (
        _default_support_judge
    ),
    artifact_dir: Path | None = None,
) -> DeletionManifest:
    trace = full_result.get("retrieval_trace") or {}
    selected = trace.get("selected_candidate") or {}
    seed_entry_id = selected.get("entry_id")
    if not seed_entry_id:
        raise ValueError(
            "FULL produced no selected entry; cannot build a closure."
        )

    selected_event_index = None
    for event in trace.get("retrieval_events") or []:
        if event.get("selected_candidate"):
            selected_event_index = event.get("event_index")
            break

    query_vector = None
    for item in full_result.get("_query_embeddings") or []:
        if (
            selected_event_index is None
            or item.get("event_index") == selected_event_index
        ):
            query_vector = item.get("vector")
            break

    seed_source = selected.get("source_id")
    key = _example_key(example)
    closure = build_closure(
        index=index,
        example=example,
        query_vector=query_vector,
        config=config,
        seed_candidates=(selected,),
        seed_source_ids=(str(seed_source),) if seed_source is not None else (),
        support_judge=support_judge,
        example_key=key,
    )
    if artifact_dir is not None:
        write_closure_artifact(
            closure, artifact_dir / f"{_safe_filename(key)}.json"
        )
    return closure.to_manifest()
