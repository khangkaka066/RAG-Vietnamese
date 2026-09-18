"""Fail (non-zero exit) if an eval JSON report falls below quality thresholds.

Used as a CI "quality gate" step after ``scripts/run_eval.py``, so a
regression in retrieval/generation quality fails the pipeline instead of
silently shipping.

Usage:
    python scripts/check_eval_gate.py reports/eval.json \\
        --min-hit-rate 0.8 --min-mrr 0.7 --min-citation-coverage 0.8 \\
        --min-faithfulness 0.6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check an eval report against quality thresholds")
    parser.add_argument("report", type=Path, help="Path to the eval JSON report")
    parser.add_argument("--min-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.7)
    parser.add_argument("--min-citation-coverage", type=float, default=0.8)
    parser.add_argument("--min-faithfulness", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))

    checks = [
        ("retrieval_hit_rate", report["retrieval_hit_rate"], args.min_hit_rate),
        ("retrieval_mrr_at_k", report["retrieval_mrr_at_k"], args.min_mrr),
        ("citation_coverage", report["citation_coverage"], args.min_citation_coverage),
        ("faithfulness", report["faithfulness"], args.min_faithfulness),
    ]

    def is_valid_metric(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    failures = []
    for name, value, minimum in checks:
        if not is_valid_metric(value):
            failures.append(f"{name}={value!r} is not a finite number")
        elif value < minimum:
            failures.append(f"{name}={value:.4f} < min {minimum:.4f}")

    for name, value, minimum in checks:
        if not is_valid_metric(value):
            print(f"[FAIL] {name}={value!r} (min {minimum:.4f}) — not a finite number")
        else:
            status = "OK" if value >= minimum else "FAIL"
            print(f"[{status}] {name}={value:.4f} (min {minimum:.4f})")

    if failures:
        print("Quality gate FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        sys.exit(1)

    print("Quality gate passed.")


if __name__ == "__main__":
    main()
