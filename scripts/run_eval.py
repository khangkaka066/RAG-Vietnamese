import json
from pathlib import Path

from vsf_rag.answering import GroundedAnswerEngine
from vsf_rag.evaluation import evaluate, load_eval_cases
from vsf_rag.retrieval import build_retriever, load_documents


ROOT = Path(__file__).resolve().parents[1]
documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
engine = GroundedAnswerEngine(build_retriever(documents))
report = evaluate(engine, load_eval_cases(ROOT / "data" / "eval.jsonl"))
print(json.dumps(report, ensure_ascii=False, indent=2))
