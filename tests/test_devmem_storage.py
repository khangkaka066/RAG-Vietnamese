from __future__ import annotations

import sqlite3

import pytest

from devmem import storage
from devmem.models import ActionStatus, FixKind


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("DEVMEM_DB", str(path))
    return path


def test_get_or_create_project_is_idempotent(db_path, tmp_path):
    with storage.connect() as conn:
        first = storage.get_or_create_project(conn, str(tmp_path))
        second = storage.get_or_create_project(conn, str(tmp_path))
    assert first.id == second.id
    assert first.path == second.path


def test_get_or_create_project_isolates_different_paths(db_path, tmp_path):
    proj_a_dir = tmp_path / "a"
    proj_b_dir = tmp_path / "b"
    proj_a_dir.mkdir()
    proj_b_dir.mkdir()
    with storage.connect() as conn:
        a = storage.get_or_create_project(conn, str(proj_a_dir))
        b = storage.get_or_create_project(conn, str(proj_b_dir))
    assert a.id != b.id


def test_start_session_rejects_second_open_session(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        storage.start_session(conn, project.id, "task one", "deadbeef", "")
        with pytest.raises(ValueError):
            storage.start_session(conn, project.id, "task two", "deadbeef", "")


def test_end_session_allows_new_session_afterward(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        first = storage.start_session(conn, project.id, "task one", None, "")
        storage.end_session(conn, first.id, "")
        second = storage.start_session(conn, project.id, "task two", None, "")
        still_open = storage.open_session(conn, project.id)
    assert second.id != first.id
    assert still_open is not None and still_open.id == second.id


def test_record_error_and_fix_and_verification_round_trip(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session = storage.start_session(conn, project.id, "fix a bug", None, "")
        error = storage.record_error(conn, project.id, session.id, "boom: division by zero")
        fix = storage.record_fix(
            conn, error.id, session.id, "guard denominator", FixKind.ATTEMPTED
        )
        verification = storage.record_verification(
            conn,
            fix.id,
            "pytest -q tests/test_math.py",
            0,
            "abc123",
            False,
            "2026-09-18T00:00:00+00:00",
            120,
            "unit test for divide()",
            False,
        )
    assert verification.fix_id == fix.id
    assert verification.exit_code == 0


def test_record_fix_has_no_verified_kind_value():
    # `fixes.kind` CHECK constraint only allows proposed/attempted — there is
    # deliberately no "verified" value anywhere in FixKind.
    assert {k.value for k in FixKind} == {"proposed", "attempted"}


def test_secret_in_error_message_is_redacted_in_raw_db_row(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session = storage.start_session(conn, project.id, "task", None, "")
        secret = "api_key=sk-THIS-IS-A-SECRET-VALUE-1234567890"
        storage.record_error(conn, project.id, session.id, f"failed auth: {secret}")

    raw_conn = sqlite3.connect(db_path)
    raw_conn.row_factory = sqlite3.Row
    row = raw_conn.execute("SELECT message FROM errors").fetchone()
    raw_conn.close()
    assert "sk-THIS-IS-A-SECRET-VALUE-1234567890" not in row["message"]
    assert "[REDACTED]" in row["message"]


def test_secret_in_fix_description_is_redacted_in_raw_db_row(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session = storage.start_session(conn, project.id, "task", None, "")
        error = storage.record_error(conn, project.id, session.id, "some error")
        secret = "token: ABCDEFGHIJ1234567890KLMNOP"
        storage.record_fix(
            conn, error.id, session.id, f"used {secret} to auth", FixKind.PROPOSED
        )

    raw_conn = sqlite3.connect(db_path)
    raw_conn.row_factory = sqlite3.Row
    row = raw_conn.execute("SELECT description FROM fixes").fetchone()
    raw_conn.close()
    assert "ABCDEFGHIJ1234567890KLMNOP" not in row["description"]


def test_record_action_and_get_action(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session = storage.start_session(conn, project.id, "task", None, "")
        action = storage.record_action(
            conn, session.id, "pytest -q", 1, ActionStatus.COMPLETED,
            "stdout here", "stderr here", "2026-09-18T00:00:00+00:00", 50,
        )
        fetched = storage.get_action(conn, action.id)
    assert fetched is not None
    assert fetched.exit_code == 1
    assert fetched.status == "completed"
