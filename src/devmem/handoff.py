"""Generate a markdown handoff for a single work session.

Distinguishes git-observed changes (diff of dirty files at session start vs
end) from action-attributed changes (only what `devmem run`/`fix` recorded) —
the two are not the same thing and must not be presented as if they were.
"""

from __future__ import annotations

import sqlite3

from . import storage
from .models import FixSummary
from .redact import redact_text


def _parse_dirty_files(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {line.strip() for line in raw.splitlines() if line.strip()}


def _git_observed_changes(session) -> tuple[set[str], set[str], set[str]]:
    start = _parse_dirty_files(session.git_dirty_files_start)
    end = _parse_dirty_files(session.git_dirty_files_end)
    added = end - start
    removed = start - end
    unchanged = start & end
    return added, removed, unchanged


def generate_handoff(conn: sqlite3.Connection, session_id: int) -> str:
    session = storage.get_session(conn, session_id)
    if session is None:
        raise ValueError(f"No such session: {session_id}")

    lines: list[str] = []
    lines.append(f"# Handoff — session {session.id}")
    lines.append("")
    lines.append(f"**Task:** {redact_text(session.description)}")
    lines.append(f"**Started:** {session.started_at}")
    lines.append(f"**Ended:** {session.ended_at or '(chưa kết thúc)'}")
    lines.append(f"**Git commit at start:** {session.git_commit_start or 'unknown'}")
    lines.append("")

    lines.append("## Thay đổi quan sát được qua git (dirty files start → end)")
    lines.append("")
    if session.ended_at is None:
        lines.append("_Session chưa kết thúc — chưa có trạng thái git cuối để so sánh._")
    else:
        added, removed, unchanged = _git_observed_changes(session)
        if not added and not removed and not unchanged:
            lines.append("_Không có file nào dirty ở đầu hoặc cuối session._")
        else:
            if added:
                lines.append("Mới dirty trong session:")
                for path in sorted(added):
                    lines.append(f"- {path}")
            if removed:
                lines.append("Không còn dirty (đã commit/revert trong session):")
                for path in sorted(removed):
                    lines.append(f"- {path}")
            if unchanged:
                lines.append("Vẫn dirty từ trước session tới giờ:")
                for path in sorted(unchanged):
                    lines.append(f"- {path}")
    lines.append("")
    lines.append(
        "_Lưu ý: danh sách trên chỉ phản ánh trạng thái `git status --porcelain`, "
        "KHÔNG suy ra từ action nào gây ra thay đổi nào._"
    )
    lines.append("")

    cursor = conn.execute(
        "SELECT * FROM actions WHERE session_id = ? ORDER BY id", (session_id,)
    )
    actions = cursor.fetchall()
    lines.append("## Actions ghi nhận trong session (do devmem run gây ra)")
    lines.append("")
    if not actions:
        lines.append("_Không có action nào._")
    else:
        for row in actions:
            lines.append(
                f"- `{redact_text(row['command'])}` → status={row['status']}, "
                f"exit_code={row['exit_code']}"
            )
    lines.append("")

    cursor = conn.execute(
        "SELECT * FROM errors WHERE session_id = ? ORDER BY id", (session_id,)
    )
    errors = cursor.fetchall()
    lines.append("## Lỗi ghi nhận trong session")
    lines.append("")
    if not errors:
        lines.append("_Không có lỗi nào được ghi nhận._")
    else:
        for error_row in errors:
            lines.append(f"### Error #{error_row['id']}: {redact_text(error_row['message'])}")
            fixes = storage.list_fixes_for_error(conn, error_row["id"])
            session_fixes = [fix for fix in fixes if fix.session_id == session_id]
            if not session_fixes:
                lines.append("- _Chưa có fix nào trong session này._")
            for fix in session_fixes:
                verifications = storage.list_verifications_for_fix(conn, fix.id)
                summary = FixSummary(fix=fix, verifications=verifications)
                status = "VERIFIED (xem scope bên dưới)" if summary.is_verified else "CHƯA VERIFIED"
                lines.append(f"- Fix #{fix.id} [{fix.kind}] — {status}")
                lines.append(f"  - Mô tả: {redact_text(fix.description)}")
                if summary.is_verified:
                    for note in summary.verified_scope_notes:
                        lines.append(f"  - Scope đã pass: {redact_text(note)}")
                elif verifications:
                    lines.append("  - Có verification nhưng chưa lần nào exit_code == 0.")
            lines.append("")

    lines.append("## Còn dang dở (fix chưa có verification exit_code == 0)")
    lines.append("")
    unresolved = []
    for error_row in errors:
        for fix in storage.list_fixes_for_error(conn, error_row["id"]):
            if fix.session_id != session_id:
                continue
            verifications = storage.list_verifications_for_fix(conn, fix.id)
            summary = FixSummary(fix=fix, verifications=verifications)
            if not summary.is_verified:
                unresolved.append((error_row["id"], fix))
    if not unresolved:
        lines.append("_Không có fix nào còn dang dở trong session này._")
    else:
        for error_id, fix in unresolved:
            lines.append(f"- Error #{error_id} / Fix #{fix.id}: {redact_text(fix.description)}")

    return "\n".join(lines) + "\n"
