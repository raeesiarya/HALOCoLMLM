"""Audit backend for the public Co-LMLM release."""

from __future__ import annotations

import argparse
from typing import Any

from lmlm_audit.core.backend import AuditBackend
from lmlm_audit.registry import BackendSpec, register_backend


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    from pathlib import Path

    group = parser.add_argument_group("Co-LMLM backend")
    group.add_argument(
        "--co-lmlm-model-path",
        type=Path,
        default=None,
        help="Local Co-LMLM checkpoint directory.",
    )
    group.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Local Co-LMLM retrieval-index directory.",
    )
    group.add_argument(
        "--entries-db-path",
        type=Path,
        default=None,
        help="Optional Co-LMLM entries.db path used to resolve index results.",
    )
    group.add_argument(
        "--co-lmlm-source-path",
        type=Path,
        default=None,
        help="Path to a public lil-lab/Co-LMLM checkout (or its src directory).",
    )
    group.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help=(
            "Retrieval similarity threshold. Defaults to None to match the "
            "released model's eval configs (always splice top-1); set a value "
            "only as a deliberate choice, since it shifts FULL toward "
            "parametric decoding and moves L(f)/R(f)."
        ),
    )
    group.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help=(
            "IVF nprobe for the index. Higher values raise geometric-closure "
            "recall (approximate IVFPQ search) at some latency cost; recorded "
            "in closure manifests."
        ),
    )


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.co_lmlm.backend import CoLMLMAuditBackend

    return CoLMLMAuditBackend.from_public_release(
        model_path=args.co_lmlm_model_path,
        index_path=args.index_path,
        db_path=args.entries_db_path,
        source_path=args.co_lmlm_source_path,
        similarity_threshold=args.similarity_threshold,
        nprobe=args.nprobe,
        max_new_tokens=args.max_new_tokens,
    )


def _search_index(backend: AuditBackend) -> Any:
    from models.co_lmlm.adapter import build_search_index

    return build_search_index(backend)


def _group_key(args: argparse.Namespace, _job: Any) -> Any:
    # One index serves every prompt file, so all jobs share one backend.
    return args.index_path


def _validate(args: argparse.Namespace) -> None:
    if args.prompt_files is None:
        raise ValueError("Co-LMLM runs require explicit --prompt-files.")
    if args.co_lmlm_model_path is None:
        raise ValueError("Co-LMLM runs require --co-lmlm-model-path.")
    if args.index_path is None:
        raise ValueError("Co-LMLM runs require --index-path.")


register_backend(
    BackendSpec(
        name="co-lmlm",
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        add_arguments=_add_arguments,
        validate=_validate,
    )
)
