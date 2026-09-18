"""Recall layer: search errors, fixes, and lessons for a project.

Wraps `LexicalRetriever` from `vietnamese_rag.retrieval`, which is a TF-IDF-style
retriever (NOT BM25) and raises ValueError on an empty document list — so this
module always checks for zero candidates before ever constructing it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from vietnamese_rag.retrieval import Document, LexicalRetriever

from . import storage
from .models import FixSummary, Lesson

UNVERIFIED_SHARED_LABEL = "[tham khảo — chưa xác minh áp dụng được cho project này]"


@dataclass(frozen=True)
class ErrorMatch:
    error_id: int
    message: str
    score: float
    fix_summaries: list[FixSummary]


@dataclass(frozen=True)
class LessonMatch:
    lesson: Lesson
    score: float
    is_shared: bool
    applicability_label: str | None


@dataclass(frozen=True)
class RecallResult:
    query: str
    project_id: int
    error_matches: list[ErrorMatch]
    lesson_matches: list[LessonMatch]
    note: str | None = None


def _build_fix_summaries(conn: sqlite3.Connection, error_id: int) -> list[FixSummary]:
    summaries = []
    for fix in storage.list_fixes_for_error(conn, error_id):
        verifications = storage.list_verifications_for_fix(conn, fix.id)
        summaries.append(FixSummary(fix=fix, verifications=verifications))
    return summaries


def _search_documents(query: str, documents: list[Document], top_k: int) -> dict[str, float]:
    if not documents:
        return {}
    retriever = LexicalRetriever(documents)
    return {result.document.id: result.score for result in retriever.search(query, top_k=top_k)}


def recall(
    conn: sqlite3.Connection,
    project_id: int,
    query: str,
    top_k: int = 5,
    include_shared: bool = False,
) -> RecallResult:
    errors = storage.list_errors_for_project(conn, project_id)
    error_documents = [
        Document(id=str(error.id), title=error.message[:80], source="error", text=error.message)
        for error in errors
    ]
    error_scores = _search_documents(query, error_documents, top_k)
    error_matches = [
        ErrorMatch(
            error_id=error.id,
            message=error.message,
            score=error_scores[str(error.id)],
            fix_summaries=_build_fix_summaries(conn, error.id),
        )
        for error in errors
        if str(error.id) in error_scores
    ]
    error_matches.sort(key=lambda match: match.score, reverse=True)

    lessons = storage.list_lessons(conn, project_id, include_shared=include_shared)
    lesson_documents = [
        Document(id=str(lesson.id), title=lesson.title, source="lesson", text=f"{lesson.title} {lesson.body}")
        for lesson in lessons
    ]
    lesson_scores = _search_documents(query, lesson_documents, top_k)
    lesson_by_id = {lesson.id: lesson for lesson in lessons}
    lesson_matches = []
    for lesson_id_str, score in lesson_scores.items():
        lesson = lesson_by_id[int(lesson_id_str)]
        is_shared = lesson.project_id is None
        label = UNVERIFIED_SHARED_LABEL if is_shared else None
        lesson_matches.append(
            LessonMatch(lesson=lesson, score=score, is_shared=is_shared, applicability_label=label)
        )
    lesson_matches.sort(key=lambda match: match.score, reverse=True)

    note = None
    if not errors and not lessons:
        note = "Chưa có bản ghi lỗi hoặc lesson nào cho project này."

    return RecallResult(
        query=query,
        project_id=project_id,
        error_matches=error_matches,
        lesson_matches=lesson_matches,
        note=note,
    )
