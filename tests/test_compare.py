import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.cli.compare import (
    LegResult,
    build_leg_plans,
    main,
    merge_comparison,
    parse_args,
)


def _args(argv):
    return parse_args(argv)


BASE = [
    "--prompt-files",
    "p.jsonl",
    "--output-dir",
    "out",
    "--colmlm-env",
    "/path/Co-LMLM",
]


def test_both_legs_planned_with_shared_flags() -> None:
    args = _args(
        BASE
        + [
            "--radius-grid",
            "0.9:0.5:0.1",
            "--closure",
            "geometric,semantic",
            "--bootstrap-oracle-from-full",
            "--colmlm-model-path",
            "/models/co",
            "--index-path",
            "/idx",
        ]
    )
    plans = build_leg_plans(args)
    names = {plan.name for plan in plans}
    assert names == {"rel-lmlm", "colmlm"}

    rel = next(plan for plan in plans if plan.name == "rel-lmlm")
    colmlm = next(plan for plan in plans if plan.name == "colmlm")

    # Shared evaluation flags reach both legs.
    for plan in (rel, colmlm):
        assert "--radius-grid" in plan.command
        assert plan.command[plan.command.index("--radius-grid") + 1] == "0.9:0.5:0.1"
        assert "geometric,semantic" in plan.command

    # Backend selection and connection flags land on the right leg.
    assert rel.command[rel.command.index("--backend") + 1] == "rel-lmlm"
    assert "--bootstrap-oracle-from-full" not in rel.command
    assert colmlm.command[colmlm.command.index("--backend") + 1] == "colmlm"
    assert "--bootstrap-oracle-from-full" in colmlm.command
    assert colmlm.command[colmlm.command.index("--colmlm-model-path") + 1] == (
        "/models/co"
    )

    # Output dirs are per-leg subdirectories.
    assert rel.output_dir == Path("out/rel")
    assert colmlm.output_dir == Path("out/colmlm")

    # Our src is exposed so the Co-LMLM env can import lmlm_audit.
    assert "lmlm_audit" not in colmlm.env["PYTHONPATH"]  # it's the src dir
    assert colmlm.env["PYTHONPATH"].split(":")[0].endswith("src")


def test_colmlm_only_flags_forwarded() -> None:
    args = _args(BASE + ["--nprobe", "128", "--faiss-mmap"])
    plans = build_leg_plans(args)
    rel = next(plan for plan in plans if plan.name == "rel-lmlm")
    colmlm = next(plan for plan in plans if plan.name == "colmlm")

    assert colmlm.command[colmlm.command.index("--nprobe") + 1] == "128"
    assert "--faiss-mmap" in colmlm.command
    # Co-LMLM-only knobs never leak to the rel leg.
    assert "--nprobe" not in rel.command
    assert "--faiss-mmap" not in rel.command


def test_uv_vs_explicit_python_interpreter() -> None:
    uv_plan = build_leg_plans(_args(BASE))[0]
    assert uv_plan.command[:4] == ["uv", "run", "--project", "."]

    explicit = build_leg_plans(
        _args(BASE + ["--rel-python", "/envs/rel/bin/python"])
    )
    rel = next(plan for plan in explicit if plan.name == "rel-lmlm")
    assert rel.command[0] == "/envs/rel/bin/python"
    assert rel.command[1:3] == ["-m", "lmlm_audit.cli.run_audit"]


def test_skip_flags() -> None:
    plans = build_leg_plans(_args(BASE + ["--skip-colmlm"]))
    assert [plan.name for plan in plans] == ["rel-lmlm"]


def test_colmlm_requires_env_or_python() -> None:
    with pytest.raises(SystemExit):
        _args(["--prompt-files", "p.jsonl", "--output-dir", "out"])


def test_cannot_skip_both() -> None:
    with pytest.raises(SystemExit):
        _args(BASE + ["--skip-rel", "--skip-colmlm"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_comparison_tags_and_summarizes(tmp_path) -> None:
    rel_dir = tmp_path / "rel"
    colmlm_dir = tmp_path / "colmlm"
    # Nested like the real per-prompt-file sweep output.
    _write_csv(
        rel_dir / "d1_sweep" / "entanglement_gaps.csv",
        [
            {"target_key": "f1", "gap": "0.0", "gap_rho": "0.9"},
            {"target_key": "f2", "gap": "0.2", "gap_rho": "0.8"},
        ],
    )
    _write_csv(
        colmlm_dir / "d1_sweep" / "entanglement_gaps.csv",
        [
            {"target_key": "f1", "gap": "0.5", "gap_rho": "0.7"},
            {"target_key": "f2", "gap": "0.9", "gap_rho": "0.6"},
        ],
    )

    legs = [
        LegResult("rel-lmlm", 0, rel_dir),
        LegResult("colmlm", 0, colmlm_dir),
    ]
    comparison_dir = tmp_path / "comparison"
    report = merge_comparison(legs, comparison_dir)

    assert "entanglement_gaps.csv" in report["written"]
    with (comparison_dir / "entanglement_gaps.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["backend"] for row in rows} == {"rel-lmlm", "colmlm"}
    assert all(row["source"] == "d1_sweep/entanglement_gaps.csv" for row in rows)

    summary = report["gap_summary"]
    assert summary["rel-lmlm"]["gap_mean"] == pytest.approx(0.1)
    assert summary["colmlm"]["gap_mean"] == pytest.approx(0.7)
    # The paper's headline contrast: relational deletion is cleaner.
    assert summary["rel-lmlm"]["gap_mean"] < summary["colmlm"]["gap_mean"]


def test_merge_handles_no_outputs(tmp_path) -> None:
    legs = [LegResult("rel-lmlm", 0, tmp_path / "rel")]
    (tmp_path / "rel").mkdir()
    report = merge_comparison(legs, tmp_path / "comparison")
    assert report["written"] == []
    assert report["gap_summary"] == {}


def test_main_orchestrates_without_real_subprocesses(tmp_path, monkeypatch) -> None:
    launched = []

    def fake_run_legs(plans):
        results = []
        for plan in plans:
            plan.output_dir.mkdir(parents=True, exist_ok=True)
            launched.append(plan.name)
            if plan.name == "rel-lmlm":
                _write_csv(
                    plan.output_dir / "sweep" / "entanglement_gaps.csv",
                    [{"target_key": "f1", "gap": "0.0", "gap_rho": "0.9"}],
                )
            else:
                _write_csv(
                    plan.output_dir / "sweep" / "entanglement_gaps.csv",
                    [{"target_key": "f1", "gap": "0.6", "gap_rho": "0.7"}],
                )
            results.append(LegResult(plan.name, 0, plan.output_dir))
        return results

    monkeypatch.setattr("lmlm_audit.cli.compare._run_legs", fake_run_legs)

    code = main(
        [
            "--prompt-files",
            "p.jsonl",
            "--output-dir",
            str(tmp_path / "out"),
            "--colmlm-env",
            "/path/Co-LMLM",
            "--radius-grid",
            "0.9:0.7:0.1",
            "--closure",
            "geometric,semantic",
        ]
    )
    assert code == 0
    assert set(launched) == {"rel-lmlm", "colmlm"}
    assert (
        tmp_path / "out" / "comparison" / "entanglement_gaps.csv"
    ).is_file()


def test_main_reports_leg_failure(tmp_path, monkeypatch) -> None:
    def fake_run_legs(plans):
        return [
            LegResult(plan.name, 1 if plan.name == "colmlm" else 0, plan.output_dir)
            for plan in plans
        ]

    monkeypatch.setattr("lmlm_audit.cli.compare._run_legs", fake_run_legs)
    code = main(
        [
            "--prompt-files",
            "p.jsonl",
            "--output-dir",
            str(tmp_path / "out"),
            "--colmlm-env",
            "/path/Co-LMLM",
        ]
    )
    assert code == 1
