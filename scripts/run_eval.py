import argparse
import json
import os
from pathlib import Path

from vsf_rag.answering import build_answer_engine
from vsf_rag.evaluation import default_eval_path, evaluate, load_eval_cases, write_csv_report, write_json_report
from vsf_rag.retrieval import build_retriever, load_documents


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation set")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--mlflow", action="store_true", default=False)
    parser.add_argument("--retriever", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
    retriever = build_retriever(documents, mode=args.retriever)
    engine = build_answer_engine(retriever)
    cases = load_eval_cases(default_eval_path())
    report = evaluate(engine, cases, top_k=args.top_k)

    if args.json_out is not None:
        write_json_report(report, args.json_out)
    if args.csv_out is not None:
        write_csv_report(report, args.csv_out)

    if args.mlflow:
        from vsf_rag.tracking import log_report

        params = {
            "retriever": type(retriever).__name__,
            "generator": "llm" if engine.llm is not None else "extractive",
            "top_k": args.top_k,
            "cases": report["cases"],
        }
        model = os.environ.get("OPENROUTER_MODEL")
        if model:
            params["openrouter_model"] = model
        artifacts = [path for path in (args.json_out, args.csv_out) if path is not None]
        log_report(report, params=params, artifacts=artifacts)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
