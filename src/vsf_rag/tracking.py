"""Optional MLflow experiment tracking for evaluation reports.

``mlflow`` is an optional dependency (``pip install -e '.[tracking]'``); it
is imported lazily inside ``log_report`` so importing this module (and
running evaluation in general) never requires it to be installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def log_report(
    report: dict[str, Any],
    params: dict[str, Any] | None = None,
    run_name: str | None = None,
    artifacts: list[str | Path] | None = None,
) -> bool:
    """Log an evaluation report to MLflow. Returns ``False`` (no-op) if
    ``mlflow`` is not installed, instead of raising."""
    try:
        import mlflow
    except ImportError:
        print("mlflow chưa được cài; bỏ qua tracking. Cài bằng: pip install -e '.[tracking]'")
        return False

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("vsf-rag-eval")

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)

        metrics = {
            "retrieval_hit_rate": report["retrieval_hit_rate"],
            "retrieval_mrr_at_k": report["retrieval_mrr_at_k"],
            "faithfulness": report["faithfulness"],
            "answer_relevancy": report["answer_relevancy"],
            "citation_coverage": report["citation_coverage"],
            "citation_coverage_model_asserted": report["citation_coverage_model_asserted"],
            "answer_keyword_coverage": report["answer_keyword_coverage"],
            **{f"latency_ms_{name}": value for name, value in report["latency_ms"].items()},
        }
        mlflow.log_metrics(metrics)

        for artifact in artifacts or []:
            mlflow.log_artifact(str(artifact))

    return True
