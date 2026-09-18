from __future__ import annotations

import pytest

from devmem import storage
from devmem.handoff import generate_handoff
from devmem.models import ActionStatus, FixKind


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("DEVMEM_DB", str(path))
    return path


def test_handoff_does_not_mix_two_parallel_sessions_same_project(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session_a = storage.start_session(conn, project.id, "task A", "commit1", "")
        storage.record_action(
            conn, session_a.id, "pytest -q test_a.py", 0, ActionStatus.COMPLETED,
            "ok", "", "2026-09-18T00:00:00+00:00", 10,
        )
        storage.end_session(conn, session_a.id, "")

        session_b = storage.start_session(conn, project.id, "task B", "commit2", "")
        storage.record_action(
            conn, session_b.id, "pytest -q test_b.py", 0, ActionStatus.COMPLETED,
            "ok", "", "2026-09-18T01:00:00+00:00", 10,
        )
        storage.end_session(conn, session_b.id, "")

        handoff_a = generate_handoff(conn, session_a.id)
        handoff_b = generate_handoff(conn, session_b.id)

    assert "test_a.py" in handoff_a
    assert "test_b.py" not in handoff_a
    assert "test_b.py" in handoff_b
    assert "test_a.py" not in handoff_b


def test_handoff_isolates_sessions_across_different_projects(db_path, tmp_path):
    dir_a = tmp_path / "proj_a"
    dir_b = tmp_path / "proj_b"
    dir_a.mkdir()
    dir_b.mkdir()
    with storage.connect() as conn:
        project_a = storage.get_or_create_project(conn, str(dir_a))
        project_b = storage.get_or_create_project(conn, str(dir_b))

        session_a = storage.start_session(conn, project_a.id, "task in A", None, "")
        error_a = storage.record_error(conn, project_a.id, session_a.id, "boom in project A")
        storage.record_fix(conn, error_a.id, session_a.id, "fix in A", FixKind.ATTEMPTED)
        storage.end_session(conn, session_a.id, "")

        session_b = storage.start_session(conn, project_b.id, "task in B", None, "")
        storage.end_session(conn, session_b.id, "")

        handoff_a = generate_handoff(conn, session_a.id)
        handoff_b = generate_handoff(conn, session_b.id)

    assert "boom in project A" in handoff_a
    assert "boom in project A" not in handoff_b


def test_session_id_required_for_actions_fixes_verifications(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session = storage.start_session(conn, project.id, "task", None, "")
        action = storage.record_action(
            conn, session.id, "echo hi", 0, ActionStatus.COMPLETED,
            "hi", "", "2026-09-18T00:00:00+00:00", 5,
        )
        error = storage.record_error(conn, project.id, session.id, "err", action_id=action.id)
        fix = storage.record_fix(conn, error.id, session.id, "fix", FixKind.PROPOSED)
        verification = storage.record_verification(
            conn, fix.id, "pytest -q", 0, "abc", False,
            "2026-09-18T00:00:00+00:00", 10, "scope", False,
        )
    assert action.session_id == session.id
    assert error.session_id == session.id
    assert fix.session_id == session.id
    assert verification.fix_id == fix.id
