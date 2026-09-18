from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import handoff as handoff_mod
from . import recall as recall_mod
from . import storage
from .models import ActionStatus, FixKind
from .redact import curated_environment_snapshot, redact_text

MAX_OUTPUT_BYTES = 200_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_OUTPUT_BYTES:
        return text
    truncated = encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
    return truncated + f"\n[... output truncated, {len(encoded) - MAX_OUTPUT_BYTES} bytes dropped ...]"


def _git(args: list[str], cwd: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _current_project(conn, cwd: str) -> storage.Project:
    return storage.get_or_create_project(conn, cwd)


def cmd_init(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
    print(f"Project registered: id={project.id} path={project.path} name={project.name}")
    return 0


def cmd_session_start(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
        commit = _git(["rev-parse", "HEAD"], cwd)
        dirty = _git(["status", "--porcelain"], cwd)
        try:
            session = storage.start_session(conn, project.id, args.description, commit, dirty)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    print(f"Session started: id={session.id}")
    return 0


def cmd_session_end(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
        session = storage.open_session(conn, project.id)
        if session is None:
            print("Error: no open session for this project.", file=sys.stderr)
            return 1
        dirty = _git(["status", "--porcelain"], cwd)
        storage.end_session(conn, session.id, dirty)
    print(f"Session ended: id={session.id}")
    return 0


def _resolve_session(conn, project_id: int, session_arg: int | None) -> int | None:
    if session_arg is not None:
        return session_arg
    session = storage.open_session(conn, project_id)
    return session.id if session is not None else None


def cmd_run(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
        session_id = _resolve_session(conn, project.id, args.session)
        if session_id is None:
            print(
                "Error: no open session. Run `devmem session start \"<desc>\"` first, "
                "or pass --session <id>.",
                file=sys.stderr,
            )
            return 1

        command = args.command
        started_at = _now()
        start_time = time.monotonic()
        status = ActionStatus.RUNNING
        exit_code: int | None = None
        stdout = ""
        stderr = ""
        try:
            result = subprocess.run(
                command,
                shell=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            exit_code = result.returncode
            stdout = _truncate(result.stdout)
            stderr = _truncate(result.stderr)
            status = ActionStatus.COMPLETED
        except FileNotFoundError as exc:
            status = ActionStatus.FAILED_TO_START
            stderr = str(exc)
        except subprocess.TimeoutExpired as exc:
            status = ActionStatus.INTERRUPTED
            stdout = _truncate(exc.stdout or "")
            stderr = _truncate((exc.stderr or "") + "\n[timeout reached]")
        except KeyboardInterrupt:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            storage.record_action(
                conn, session_id, " ".join(command), None, ActionStatus.INTERRUPTED,
                stdout, stderr, started_at, duration_ms,
            )
            raise

        duration_ms = int((time.monotonic() - start_time) * 1000)
        action = storage.record_action(
            conn, session_id, " ".join(command), exit_code, status,
            stdout, stderr, started_at, duration_ms,
        )

    print(f"Action recorded: id={action.id} status={status.value} exit_code={exit_code}")
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return exit_code if exit_code is not None else 1


def cmd_error_record(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
        session_id = _resolve_session(conn, project.id, args.session)
        if session_id is None:
            print("Error: no open session. Pass --session or start one.", file=sys.stderr)
            return 1
        environment_note = curated_environment_snapshot() if args.with_environment else args.environment_note
        error = storage.record_error(
            conn, project.id, session_id, args.message,
            action_id=args.action_id, environment_note=environment_note,
        )
    print(f"Error recorded: id={error.id}")
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
        session_id = _resolve_session(conn, project.id, args.session)
        if session_id is None:
            print("Error: no open session. Pass --session or start one.", file=sys.stderr)
            return 1
        kind = FixKind.ATTEMPTED if args.attempted else FixKind.PROPOSED
        fix = storage.record_fix(
            conn, args.error_id, session_id, args.description, kind, evidence=args.evidence
        )
    print(f"Fix recorded: id={fix.id} kind={kind.value}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        fix = storage.get_fix(conn, args.fix_id)
        if fix is None:
            print(f"Error: no such fix: {args.fix_id}", file=sys.stderr)
            return 1

        prior = storage.list_verifications_for_fix(conn, args.fix_id)
        prior_failed = any(v.exit_code != 0 for v in prior)
        reproduced_failure_first = args.reproduced_first and prior_failed

        started_at = _now()
        try:
            result = subprocess.run(
                args.command, shell=False, capture_output=True, text=True, check=False
            )
            exit_code = result.returncode
        except FileNotFoundError as exc:
            print(f"Error: command not found: {exc}", file=sys.stderr)
            return 1
        duration_ms = None

        commit = _git(["rev-parse", "HEAD"], cwd)
        dirty_output = _git(["status", "--porcelain"], cwd)
        git_dirty = bool(dirty_output)

        scope_note = args.scope
        if not reproduced_failure_first:
            caveat = (
                "Chưa có lần verify nào trước đó fail cho fix này — chỉ có thể khẳng định "
                "lệnh hiện tại pass, KHÔNG thể khẳng định đã xác minh đúng nguyên nhân gốc."
            )
            scope_note = f"{scope_note}\n[CẢNH BÁO] {caveat}" if scope_note else f"[CẢNH BÁO] {caveat}"

        verification = storage.record_verification(
            conn, args.fix_id, " ".join(args.command), exit_code, commit, git_dirty,
            started_at, duration_ms, scope_note, reproduced_failure_first,
        )

    print(
        f"Verification recorded: id={verification.id} exit_code={exit_code} "
        f"reproduced_failure_first={reproduced_failure_first}"
    )
    if exit_code == 0 and not reproduced_failure_first:
        print(
            "Note: lệnh pass, nhưng chưa có bằng chứng đã tái hiện lỗi trước khi sửa — "
            "chỉ coi là 'lệnh hiện tại pass', không phải 'đã xác minh nguyên nhân gốc'."
        )
    return exit_code


def cmd_lesson_promote(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        fix = storage.get_fix(conn, args.fix_id)
        if fix is None:
            print(f"Error: no such fix: {args.fix_id}", file=sys.stderr)
            return 1
        project_id = None if args.shared else _current_project(conn, cwd).id
        lesson = storage.record_lesson(
            conn, project_id, args.title, args.body or fix.description,
            args.applies_when, args.fix_id,
        )
    print(f"Lesson recorded: id={lesson.id} shared={args.shared}")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
        result = recall_mod.recall(
            conn, project.id, args.query, top_k=args.top_k, include_shared=args.include_shared
        )

    if result.note:
        print(result.note)

    print(f"\n== Errors matching \"{result.query}\" ==")
    if not result.error_matches:
        print("(none)")
    for match in result.error_matches:
        print(f"\nError #{match.error_id} (score={match.score:.3f}): {redact_text(match.message)}")
        for summary in match.fix_summaries:
            status = "VERIFIED" if summary.is_verified else summary.fix.kind
            print(f"  - Fix #{summary.fix.id} [{status}]: {redact_text(summary.fix.description)}")
            if summary.is_verified:
                for note in summary.verified_scope_notes:
                    print(f"      scope: {redact_text(note)}")

    print(f"\n== Lessons matching \"{result.query}\" ==")
    if not result.lesson_matches:
        print("(none)")
    for lesson_match in result.lesson_matches:
        label = f" {lesson_match.applicability_label}" if lesson_match.applicability_label else ""
        print(
            f"- Lesson #{lesson_match.lesson.id} (score={lesson_match.score:.3f}){label}: "
            f"{redact_text(lesson_match.lesson.title)}"
        )
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    cwd = str(Path.cwd())
    with storage.connect() as conn:
        project = _current_project(conn, cwd)
        session_id = args.session
        if session_id is None:
            session = storage.open_session(conn, project.id) or storage.latest_ended_session(
                conn, project.id
            )
            if session is None:
                print("Error: no session found for this project.", file=sys.stderr)
                return 1
            session_id = session.id
        markdown = handoff_mod.generate_handoff(conn, session_id)
    print(markdown)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devmem", description="Vietnamese Dev Memory CLI")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("init", help="Register the current directory as a devmem project").set_defaults(func=cmd_init)

    p_session = sub.add_parser("session", help="Manage work sessions")
    session_sub = p_session.add_subparsers(dest="session_command", required=True)
    p_start = session_sub.add_parser("start", help="Start a new work session")
    p_start.add_argument("description")
    p_start.set_defaults(func=cmd_session_start)
    p_end = session_sub.add_parser("end", help="End the currently open work session")
    p_end.set_defaults(func=cmd_session_end)

    p_run = sub.add_parser("run", help="Run a command and record it as an action")
    p_run.add_argument("--session", type=int, default=None)
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.set_defaults(func=cmd_run)

    p_error = sub.add_parser("error", help="Manage errors")
    error_sub = p_error.add_subparsers(dest="error_command", required=True)
    p_error_record = error_sub.add_parser("record", help="Record a new error")
    p_error_record.add_argument("message")
    p_error_record.add_argument("--action-id", type=int, default=None)
    p_error_record.add_argument("--environment-note", default=None)
    p_error_record.add_argument("--with-environment", action="store_true", help="Attach a curated environment snapshot")
    p_error_record.add_argument("--session", type=int, default=None)
    p_error_record.set_defaults(func=cmd_error_record)

    p_fix = sub.add_parser("fix", help="Record a fix proposal/attempt for an error")
    p_fix.add_argument("error_id", type=int)
    p_fix.add_argument("description")
    p_fix.add_argument("--evidence", default=None)
    p_fix.add_argument("--attempted", action="store_true", help="Mark as attempted rather than merely proposed")
    p_fix.add_argument("--session", type=int, default=None)
    p_fix.set_defaults(func=cmd_fix)

    p_verify = sub.add_parser("verify", help="Run a command to verify a fix")
    p_verify.add_argument("fix_id", type=int)
    p_verify.add_argument("--scope", required=True, dest="scope", help="Description of what this verification proves")
    p_verify.add_argument("--reproduced-first", action="store_true", help="A prior verify of this fix failed before")
    p_verify.set_defaults(func=cmd_verify)

    p_lesson = sub.add_parser("lesson", help="Manage lessons")
    lesson_sub = p_lesson.add_subparsers(dest="lesson_command", required=True)
    p_promote = lesson_sub.add_parser("promote", help="Promote a fix into a reusable lesson")
    p_promote.add_argument("fix_id", type=int)
    p_promote.add_argument("--title", required=True)
    p_promote.add_argument("--body", default=None)
    p_promote.add_argument("--applies-when", dest="applies_when", default=None)
    p_promote.add_argument("--shared", action="store_true", help="Share across all projects (project_id=NULL)")
    p_promote.set_defaults(func=cmd_lesson_promote)

    p_recall = sub.add_parser("recall", help="Search errors, fixes, and lessons")
    p_recall.add_argument("query")
    p_recall.add_argument("--top-k", type=int, default=5)
    p_recall.add_argument("--include-shared", action="store_true")
    p_recall.set_defaults(func=cmd_recall)

    p_handoff = sub.add_parser("handoff", help="Generate a markdown handoff for a session")
    p_handoff.add_argument("--session", type=int, default=None)
    p_handoff.set_defaults(func=cmd_handoff)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # Split off the trailing "<command...>" ourselves before argparse sees it.
    # argparse's nargs=REMAINDER does not interleave correctly with a required
    # named option (e.g. --scope) placed before it on the command line, so we
    # never let REMAINDER actually consume real tokens.
    command_tail: list[str] | None = None
    if "--" in argv:
        sep_index = argv.index("--")
        command_tail = argv[sep_index + 1 :]
        argv = argv[:sep_index]

    parser = build_parser()
    args = parser.parse_args(argv)
    args.command = command_tail if command_tail is not None else []

    if args.subcommand in {"run", "verify"} and not args.command:
        print(
            f"Error: `devmem {args.subcommand}` requires a command after `--`.",
            file=sys.stderr,
        )
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
