from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import (
    Action,
    ActionStatus,
    ErrorRecord,
    Fix,
    FixKind,
    Lesson,
    Project,
    Verification,
    WorkSession,
)
from .redact import redact_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS work_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    description TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    git_commit_start TEXT,
    git_dirty_files_start TEXT,
    git_dirty_files_end TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES work_sessions(id),
    command TEXT NOT NULL,
    exit_code INTEGER,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'interrupted', 'failed_to_start')),
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    duration_ms INTEGER
);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    session_id INTEGER NOT NULL REFERENCES work_sessions(id),
    action_id INTEGER REFERENCES actions(id),
    message TEXT NOT NULL,
    environment_note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fixes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_id INTEGER NOT NULL REFERENCES errors(id),
    session_id INTEGER NOT NULL REFERENCES work_sessions(id),
    description TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('proposed', 'attempted')),
    evidence TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fix_id INTEGER NOT NULL REFERENCES fixes(id),
    command TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    duration_ms INTEGER,
    scope_note TEXT NOT NULL,
    reproduced_failure_first INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    applies_when TEXT,
    source_fix_id INTEGER NOT NULL REFERENCES fixes(id),
    created_at TEXT NOT NULL
);
"""


def default_db_path() -> Path:
    override = os.environ.get("DEVMEM_DB")
    if override:
        return Path(override)
    return Path.home() / ".devmem" / "devmem.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_or_create_project(conn: sqlite3.Connection, path: str, name: str | None = None) -> Project:
    resolved_path = str(Path(path).resolve())
    row = conn.execute("SELECT * FROM projects WHERE path = ?", (resolved_path,)).fetchone()
    if row is not None:
        return Project(**dict(row))
    resolved_name = name or Path(resolved_path).name
    cursor = conn.execute(
        "INSERT INTO projects (path, name, created_at) VALUES (?, ?, ?)",
        (resolved_path, resolved_name, _now()),
    )
    return Project(id=cursor.lastrowid, path=resolved_path, name=resolved_name, created_at=_now())


def get_project_by_path(conn: sqlite3.Connection, path: str) -> Project | None:
    resolved_path = str(Path(path).resolve())
    row = conn.execute("SELECT * FROM projects WHERE path = ?", (resolved_path,)).fetchone()
    return Project(**dict(row)) if row is not None else None


def open_session(conn: sqlite3.Connection, project_id: int) -> WorkSession | None:
    row = conn.execute(
        "SELECT * FROM work_sessions WHERE project_id = ? AND ended_at IS NULL "
        "ORDER BY id DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    return WorkSession(**dict(row)) if row is not None else None


def start_session(
    conn: sqlite3.Connection,
    project_id: int,
    description: str,
    git_commit_start: str | None,
    git_dirty_files_start: str | None,
) -> WorkSession:
    if open_session(conn, project_id) is not None:
        raise ValueError(
            "A work session is already open for this project. "
            "Run `devmem session end` first, or pass --session explicitly."
        )
    started_at = _now()
    cursor = conn.execute(
        "INSERT INTO work_sessions "
        "(project_id, description, started_at, ended_at, git_commit_start, git_dirty_files_start) "
        "VALUES (?, ?, ?, NULL, ?, ?)",
        (project_id, redact_text(description), started_at, git_commit_start, git_dirty_files_start),
    )
    return WorkSession(
        id=cursor.lastrowid,
        project_id=project_id,
        description=description,
        started_at=started_at,
        ended_at=None,
        git_commit_start=git_commit_start,
        git_dirty_files_start=git_dirty_files_start,
    )


def end_session(conn: sqlite3.Connection, session_id: int, git_dirty_files_end: str | None) -> None:
    conn.execute(
        "UPDATE work_sessions SET ended_at = ?, git_dirty_files_end = ? WHERE id = ?",
        (_now(), git_dirty_files_end, session_id),
    )


def get_session(conn: sqlite3.Connection, session_id: int) -> WorkSession | None:
    row = conn.execute("SELECT * FROM work_sessions WHERE id = ?", (session_id,)).fetchone()
    return WorkSession(**dict(row)) if row is not None else None


def latest_ended_session(conn: sqlite3.Connection, project_id: int) -> WorkSession | None:
    row = conn.execute(
        "SELECT * FROM work_sessions WHERE project_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    return WorkSession(**dict(row)) if row is not None else None


def record_action(
    conn: sqlite3.Connection,
    session_id: int,
    command: str,
    exit_code: int | None,
    status: ActionStatus,
    stdout: str,
    stderr: str,
    started_at: str,
    duration_ms: int | None,
) -> Action:
    cursor = conn.execute(
        "INSERT INTO actions (session_id, command, exit_code, status, stdout, stderr, started_at, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            redact_text(command),
            exit_code,
            status.value,
            redact_text(stdout) or "",
            redact_text(stderr) or "",
            started_at,
            duration_ms,
        ),
    )
    return Action(
        id=cursor.lastrowid,
        session_id=session_id,
        command=command,
        exit_code=exit_code,
        status=status.value,
        stdout=stdout,
        stderr=stderr,
        started_at=started_at,
        duration_ms=duration_ms,
    )


def get_action(conn: sqlite3.Connection, action_id: int) -> Action | None:
    row = conn.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
    return Action(**dict(row)) if row is not None else None


def record_error(
    conn: sqlite3.Connection,
    project_id: int,
    session_id: int,
    message: str,
    action_id: int | None = None,
    environment_note: str | None = None,
) -> ErrorRecord:
    created_at = _now()
    cursor = conn.execute(
        "INSERT INTO errors (project_id, session_id, action_id, message, environment_note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, session_id, action_id, redact_text(message), redact_text(environment_note), created_at),
    )
    return ErrorRecord(
        id=cursor.lastrowid,
        project_id=project_id,
        session_id=session_id,
        action_id=action_id,
        message=message,
        environment_note=environment_note,
        created_at=created_at,
    )


def get_error(conn: sqlite3.Connection, error_id: int) -> ErrorRecord | None:
    row = conn.execute("SELECT * FROM errors WHERE id = ?", (error_id,)).fetchone()
    return ErrorRecord(**dict(row)) if row is not None else None


def list_errors_for_project(conn: sqlite3.Connection, project_id: int) -> list[ErrorRecord]:
    rows = conn.execute(
        "SELECT * FROM errors WHERE project_id = ? ORDER BY id", (project_id,)
    ).fetchall()
    return [ErrorRecord(**dict(row)) for row in rows]


def record_fix(
    conn: sqlite3.Connection,
    error_id: int,
    session_id: int,
    description: str,
    kind: FixKind,
    evidence: str | None = None,
) -> Fix:
    created_at = _now()
    cursor = conn.execute(
        "INSERT INTO fixes (error_id, session_id, description, kind, evidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (error_id, session_id, redact_text(description), kind.value, redact_text(evidence), created_at),
    )
    return Fix(
        id=cursor.lastrowid,
        error_id=error_id,
        session_id=session_id,
        description=description,
        kind=kind.value,
        evidence=evidence,
        created_at=created_at,
    )


def get_fix(conn: sqlite3.Connection, fix_id: int) -> Fix | None:
    row = conn.execute("SELECT * FROM fixes WHERE id = ?", (fix_id,)).fetchone()
    return Fix(**dict(row)) if row is not None else None


def list_fixes_for_error(conn: sqlite3.Connection, error_id: int) -> list[Fix]:
    rows = conn.execute("SELECT * FROM fixes WHERE error_id = ? ORDER BY id", (error_id,)).fetchall()
    return [Fix(**dict(row)) for row in rows]


def list_verifications_for_fix(conn: sqlite3.Connection, fix_id: int) -> list[Verification]:
    rows = conn.execute(
        "SELECT * FROM verifications WHERE fix_id = ? ORDER BY id", (fix_id,)
    ).fetchall()
    return [Verification(**_coerce_verification_row(dict(row))) for row in rows]


def _coerce_verification_row(row: dict) -> dict:
    row["git_dirty"] = bool(row["git_dirty"])
    row["reproduced_failure_first"] = bool(row["reproduced_failure_first"])
    return row


def record_verification(
    conn: sqlite3.Connection,
    fix_id: int,
    command: str,
    exit_code: int,
    git_commit: str | None,
    git_dirty: bool,
    started_at: str,
    duration_ms: int | None,
    scope_note: str,
    reproduced_failure_first: bool,
) -> Verification:
    cursor = conn.execute(
        "INSERT INTO verifications "
        "(fix_id, command, exit_code, git_commit, git_dirty, started_at, duration_ms, scope_note, reproduced_failure_first) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fix_id,
            redact_text(command),
            exit_code,
            git_commit,
            int(git_dirty),
            started_at,
            duration_ms,
            redact_text(scope_note),
            int(reproduced_failure_first),
        ),
    )
    return Verification(
        id=cursor.lastrowid,
        fix_id=fix_id,
        command=command,
        exit_code=exit_code,
        git_commit=git_commit,
        git_dirty=git_dirty,
        started_at=started_at,
        duration_ms=duration_ms,
        scope_note=scope_note,
        reproduced_failure_first=reproduced_failure_first,
    )


def record_lesson(
    conn: sqlite3.Connection,
    project_id: int | None,
    title: str,
    body: str,
    applies_when: str | None,
    source_fix_id: int,
) -> Lesson:
    created_at = _now()
    redacted_applies_when = redact_text(applies_when)
    cursor = conn.execute(
        "INSERT INTO lessons (project_id, title, body, applies_when, source_fix_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, redact_text(title), redact_text(body), redacted_applies_when, source_fix_id, created_at),
    )
    return Lesson(
        id=cursor.lastrowid,
        project_id=project_id,
        title=title,
        body=body,
        applies_when=redacted_applies_when,
        source_fix_id=source_fix_id,
        created_at=created_at,
    )


def list_lessons(
    conn: sqlite3.Connection, project_id: int | None, include_shared: bool
) -> list[Lesson]:
    if include_shared:
        rows = conn.execute(
            "SELECT * FROM lessons WHERE project_id = ? OR project_id IS NULL ORDER BY id",
            (project_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM lessons WHERE project_id = ? ORDER BY id", (project_id,)
        ).fetchall()
    return [Lesson(**dict(row)) for row in rows]
