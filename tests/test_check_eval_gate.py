"""Tests for scripts/check_eval_gate.py, in particular NaN/Infinity handling."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_eval_gate.py"


def _run(report: dict) -> subprocess.CompletedProcess:
    report_path = Path(SCRIPT).parent.parent / "reports" / "_test_eval_gate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(report_path)],
            capture_output=True,
            text=True,
        )
    finally:
        report_path.unlink(missing_ok=True)


def test_passing_report_exits_zero():
    result = _run(
        {
            "retrieval_hit_rate": 0.9,
            "retrieval_mrr_at_k": 0.8,
            "citation_coverage": 0.9,
            "faithfulness": 0.7,
        }
    )
    assert result.returncode == 0
    assert "Quality gate passed." in result.stdout


def test_below_threshold_exits_nonzero():
    result = _run(
        {
            "retrieval_hit_rate": 0.1,
            "retrieval_mrr_at_k": 0.8,
            "citation_coverage": 0.9,
            "faithfulness": 0.7,
        }
    )
    assert result.returncode != 0
    assert "Quality gate FAILED" in result.stderr


def test_nan_metric_fails_instead_of_silently_passing():
    """Regression test: NaN < minimum is False, so a naive `<` check would
    incorrectly let a NaN metric pass the gate."""
    result = _run(
        {
            "retrieval_hit_rate": float("nan"),
            "retrieval_mrr_at_k": 0.8,
            "citation_coverage": 0.9,
            "faithfulness": 0.7,
        }
    )
    assert result.returncode != 0
    assert "Quality gate FAILED" in result.stderr
    assert "not a finite number" in result.stderr


def test_infinity_metric_fails_instead_of_silently_passing():
    result = _run(
        {
            "retrieval_hit_rate": float("inf"),
            "retrieval_mrr_at_k": 0.8,
            "citation_coverage": 0.9,
            "faithfulness": 0.7,
        }
    )
    assert result.returncode != 0
    assert "Quality gate FAILED" in result.stderr
    assert "not a finite number" in result.stderr
