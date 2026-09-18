from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED_TO_START = "failed_to_start"


class FixKind(str, Enum):
    PROPOSED = "proposed"
    ATTEMPTED = "attempted"


@dataclass(frozen=True)
class Project:
    id: int
    path: str
    name: str
    created_at: str


@dataclass(frozen=True)
class WorkSession:
    id: int
    project_id: int
    description: str
    started_at: str
    ended_at: str | None
    git_commit_start: str | None
    git_dirty_files_start: str | None
    git_dirty_files_end: str | None = None


@dataclass(frozen=True)
class Action:
    id: int
    session_id: int
    command: str
    exit_code: int | None
    status: str
    stdout: str
    stderr: str
    started_at: str
    duration_ms: int | None


@dataclass(frozen=True)
class ErrorRecord:
    id: int
    project_id: int
    session_id: int
    action_id: int | None
    message: str
    environment_note: str | None
    created_at: str


@dataclass(frozen=True)
class Fix:
    id: int
    error_id: int
    session_id: int
    description: str
    kind: str
    evidence: str | None
    created_at: str


@dataclass(frozen=True)
class Verification:
    id: int
    fix_id: int
    command: str
    exit_code: int
    git_commit: str | None
    git_dirty: bool
    started_at: str
    duration_ms: int | None
    scope_note: str
    reproduced_failure_first: bool


@dataclass(frozen=True)
class Lesson:
    id: int
    project_id: int | None
    title: str
    body: str
    applies_when: str | None
    source_fix_id: int
    created_at: str


@dataclass
class FixSummary:
    """Read-time view of a fix: whether it is verified is INFERRED, never stored."""

    fix: Fix
    verifications: list[Verification] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return any(v.exit_code == 0 for v in self.verifications)

    @property
    def verified_scope_notes(self) -> list[str]:
        return [v.scope_note for v in self.verifications if v.exit_code == 0]
