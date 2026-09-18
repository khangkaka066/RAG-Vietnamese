from __future__ import annotations

import sqlite3

import pytest

from devmem import storage
from devmem.cli import main
from devmem.models import FixKind
from devmem.recall import UNVERIFIED_SHARED_LABEL, recall


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("DEVMEM_DB", str(path))
    return path


def _make_fix(conn, project_path):
    project = storage.get_or_create_project(conn, str(project_path))
    session = storage.start_session(conn, project.id, "task", None, "")
    error = storage.record_error(conn, project.id, session.id, "boom")
    fix = storage.record_fix(conn, error.id, session.id, "guard denominator", FixKind.ATTEMPTED)
    return project, fix


def test_record_lesson_requires_source_fix_id(db_path, tmp_path):
    with storage.connect() as conn:
        project, fix = _make_fix(conn, tmp_path)
        lesson = storage.record_lesson(
            conn, project.id, "Guard division by zero", "Check denominator before dividing",
            "python", fix.id,
        )
    assert lesson.source_fix_id == fix.id


def test_recall_default_excludes_shared_lessons_from_other_project(db_path, tmp_path):
    proj_a_dir = tmp_path / "a"
    proj_b_dir = tmp_path / "b"
    proj_a_dir.mkdir()
    proj_b_dir.mkdir()
    with storage.connect() as conn:
        project_a, fix_a = _make_fix(conn, proj_a_dir)
        storage.record_lesson(
            conn, None, "Guard division by zero", "Check denominator before dividing",
            "python", fix_a.id,
        )
        project_b = storage.get_or_create_project(conn, str(proj_b_dir))

        default_result = recall(conn, project_b.id, "denominator", include_shared=False)
        shared_result = recall(conn, project_b.id, "denominator", include_shared=True)

    assert default_result.lesson_matches == []
    assert len(shared_result.lesson_matches) == 1
    assert shared_result.lesson_matches[0].applicability_label == UNVERIFIED_SHARED_LABEL


def test_recall_on_empty_project_returns_empty_with_note_no_raise(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        result = recall(conn, project.id, "anything")
    assert result.error_matches == []
    assert result.lesson_matches == []
    assert result.note is not None


def test_lesson_promote_without_body_falls_back_to_fix_description(db_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["session", "start", "task"])
    main(["error", "record", "boom"])
    main(["fix", "1", "guard denominator before dividing"])

    exit_code = main(
        ["lesson", "promote", "1", "--title", "Guard division by zero", "--applies-when", "python"]
    )
    assert exit_code == 0

    with storage.connect() as conn:
        lessons = storage.list_lessons(conn, storage.get_or_create_project(conn, str(tmp_path)).id, include_shared=False)
    assert len(lessons) == 1
    assert lessons[0].body == "guard denominator before dividing"


def test_secret_in_lesson_body_is_redacted_in_raw_db_row(db_path, tmp_path):
    with storage.connect() as conn:
        project, fix = _make_fix(conn, tmp_path)
        secret = "api_key=sk-THIS-IS-A-SECRET-VALUE-1234567890"
        storage.record_lesson(
            conn, project.id, "Rotate keys", f"Remember to rotate {secret}", "always", fix.id,
        )

    raw_conn = sqlite3.connect(db_path)
    raw_conn.row_factory = sqlite3.Row
    row = raw_conn.execute("SELECT body FROM lessons").fetchone()
    raw_conn.close()
    assert "sk-THIS-IS-A-SECRET-VALUE-1234567890" not in row["body"]
