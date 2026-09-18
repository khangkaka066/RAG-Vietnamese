from __future__ import annotations

import pytest

from devmem import storage
from devmem.cli import main


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("DEVMEM_DB", str(path))
    return path


def test_full_flow_init_through_handoff(db_path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["init"]) == 0
    assert main(["session", "start", "fix division bug"]) == 0

    exit_code = main(["run", "--", "python3", "-c", "import sys; sys.exit(1)"])
    assert exit_code == 1

    assert main(["error", "record", "division by zero in calc()", "--action-id", "1"]) == 0
    assert main(["fix", "1", "guard denominator before dividing"]) == 0

    exit_code = main(
        ["verify", "1", "--scope", "unit test", "--", "python3", "-c", "import sys; sys.exit(1)"]
    )
    assert exit_code == 1

    exit_code = main(
        [
            "verify", "1", "--scope", "unit test", "--reproduced-first",
            "--", "python3", "-c", "import sys; sys.exit(0)",
        ]
    )
    assert exit_code == 0

    capsys.readouterr()
    assert main(["recall", "division"]) == 0
    recall_out = capsys.readouterr().out
    assert "Error #1" in recall_out
    assert "division by zero" in recall_out
    assert "VERIFIED" in recall_out

    assert main(["session", "end"]) == 0

    capsys.readouterr()
    assert main(["handoff"]) == 0
    handoff_out = capsys.readouterr().out
    assert "fix division bug" in handoff_out


def test_second_concurrent_session_start_is_rejected(db_path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["session", "start", "task one"]) == 0
    exit_code = main(["session", "start", "task two"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error" in captured.err


def test_two_projects_are_isolated_via_cli(db_path, tmp_path, monkeypatch, capsys):
    proj_a = tmp_path / "proj_a"
    proj_b = tmp_path / "proj_b"
    proj_a.mkdir()
    proj_b.mkdir()

    monkeypatch.chdir(proj_a)
    main(["session", "start", "task in A"])
    main(["error", "record", "boom only in project A"])
    main(["session", "end"])

    monkeypatch.chdir(proj_b)
    main(["session", "start", "task in B"])
    main(["session", "end"])

    capsys.readouterr()
    main(["recall", "boom"])
    recall_out_b = capsys.readouterr().out
    assert "boom only in project A" not in recall_out_b

    monkeypatch.chdir(proj_a)
    capsys.readouterr()
    main(["recall", "boom"])
    recall_out_a = capsys.readouterr().out
    assert "boom only in project A" in recall_out_a


def test_run_records_action_with_correct_exit_code(db_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["session", "start", "task"])
    exit_code = main(["run", "--", "python3", "-c", "import sys; sys.exit(3)"])
    assert exit_code == 3

    with storage.connect() as conn:
        action = storage.get_action(conn, 1)
    assert action.exit_code == 3
    assert action.status == "completed"


def test_run_without_open_session_fails_with_clear_message(db_path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    exit_code = main(["run", "--", "python3", "-c", "import sys; sys.exit(0)"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no open session" in captured.err


@pytest.mark.parametrize("argv", (["run"], ["run", "--"]))
def test_run_requires_a_command_after_separator(db_path, tmp_path, monkeypatch, capsys, argv):
    monkeypatch.chdir(tmp_path)
    exit_code = main(argv)
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires a command after `--`" in captured.err
