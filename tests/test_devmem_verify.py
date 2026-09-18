from __future__ import annotations

from devmem import storage
from devmem.cli import main
from devmem.models import FixKind, FixSummary


import pytest


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("DEVMEM_DB", str(path))
    return path


def test_fix_is_not_verified_without_any_passing_verification(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session = storage.start_session(conn, project.id, "task", None, "")
        error = storage.record_error(conn, project.id, session.id, "boom")
        fix = storage.record_fix(conn, error.id, session.id, "attempt", FixKind.ATTEMPTED)
        storage.record_verification(
            conn, fix.id, "pytest -q", 1, "abc", False,
            "2026-09-18T00:00:00+00:00", 10, "scope", False,
        )
        verifications = storage.list_verifications_for_fix(conn, fix.id)
    summary = FixSummary(fix=fix, verifications=verifications)
    assert summary.is_verified is False


def test_fix_is_verified_only_after_a_passing_verification(db_path, tmp_path):
    with storage.connect() as conn:
        project = storage.get_or_create_project(conn, str(tmp_path))
        session = storage.start_session(conn, project.id, "task", None, "")
        error = storage.record_error(conn, project.id, session.id, "boom")
        fix = storage.record_fix(conn, error.id, session.id, "attempt", FixKind.ATTEMPTED)
        storage.record_verification(
            conn, fix.id, "pytest -q", 1, "abc", False,
            "2026-09-18T00:00:00+00:00", 10, "first run still failing", False,
        )
        storage.record_verification(
            conn, fix.id, "pytest -q", 0, "def", False,
            "2026-09-18T01:00:00+00:00", 10, "second run passing", True,
        )
        verifications = storage.list_verifications_for_fix(conn, fix.id)
    summary = FixSummary(fix=fix, verifications=verifications)
    assert summary.is_verified is True
    assert summary.verified_scope_notes == ["second run passing"]


def test_cmd_verify_appends_caveat_when_no_prior_failure(db_path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["session", "start", "task"])
    main(["error", "record", "boom"])
    main(["fix", "1", "guard denominator"])

    exit_code = main(
        ["verify", "1", "--scope", "unit test", "--", "python3", "-c", "import sys; sys.exit(0)"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Note: lệnh pass" in captured.out

    with storage.connect() as conn:
        verifications = storage.list_verifications_for_fix(conn, 1)
    assert len(verifications) == 1
    assert verifications[0].reproduced_failure_first is False
    assert "CẢNH BÁO" in verifications[0].scope_note


def test_cmd_verify_reproduced_first_true_only_after_prior_failing_verification(
    db_path, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    main(["session", "start", "task"])
    main(["error", "record", "boom"])
    main(["fix", "1", "guard denominator"])

    main(["verify", "1", "--scope", "unit test", "--", "python3", "-c", "import sys; sys.exit(1)"])
    exit_code = main(
        [
            "verify", "1", "--scope", "unit test", "--reproduced-first",
            "--", "python3", "-c", "import sys; sys.exit(0)",
        ]
    )
    assert exit_code == 0

    with storage.connect() as conn:
        verifications = storage.list_verifications_for_fix(conn, 1)
    assert verifications[0].exit_code == 1
    assert verifications[1].exit_code == 0
    assert verifications[1].reproduced_failure_first is True
    assert "CẢNH BÁO" not in verifications[1].scope_note


def test_reproduced_first_flag_ignored_without_any_prior_verification(db_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["session", "start", "task"])
    main(["error", "record", "boom"])
    main(["fix", "1", "guard denominator"])

    main(
        [
            "verify", "1", "--scope", "unit test", "--reproduced-first",
            "--", "python3", "-c", "import sys; sys.exit(0)",
        ]
    )

    with storage.connect() as conn:
        verifications = storage.list_verifications_for_fix(conn, 1)
    assert verifications[0].reproduced_failure_first is False
    assert "CẢNH BÁO" in verifications[0].scope_note


@pytest.mark.parametrize(
    "argv",
    (
        ["verify", "1", "--scope", "unit test"],
        ["verify", "1", "--scope", "unit test", "--"],
    ),
)
def test_verify_requires_a_command_after_separator(db_path, tmp_path, monkeypatch, capsys, argv):
    monkeypatch.chdir(tmp_path)
    exit_code = main(argv)
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires a command after `--`" in captured.err
