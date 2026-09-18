"""Shared redaction utility applied to every field before it is persisted or exported.

WARNING: this is a defense-in-depth layer, not a guarantee. Regex-based secret
detection and env-var substring replacement will miss secrets that don't match
a known shape (e.g. an unlabeled random string, a secret split across lines).
Callers must still avoid deliberately logging raw secrets.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

_LABELED_SECRET = re.compile(
    r"(?i)(api[_-]?key|access[_-]?key|secret|password|passwd|token|bearer)"
    r"(\s*[:=]\s*|\s+)"
    r"([^\s'\"]{4,})"
)

# Base64/hex-shaped runs only (>=20 chars) — deliberately excludes '-'/'_' so
# ordinary kebab-case/snake_case identifiers and file paths aren't swept up.
_LONG_OPAQUE_TOKEN = re.compile(r"\b(?:[A-Fa-f0-9]{20,}|[A-Za-z0-9+/]{20,}={0,2})\b")

_MAX_ENV_VALUE_LEN = 6


def _env_values() -> list[str]:
    values = []
    for name, value in os.environ.items():
        if len(value) < _MAX_ENV_VALUE_LEN:
            continue
        looks_sensitive = any(
            marker in name.upper()
            for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
        )
        if looks_sensitive:
            values.append(value)
    return values


def redact_text(text: str | None) -> str | None:
    """Redact likely secrets from `text`. Returns None unchanged (nullable fields)."""
    if text is None:
        return None
    if not text:
        return text

    redacted = text
    for value in _env_values():
        if value:
            redacted = redacted.replace(value, "[REDACTED]")

    def _replace_labeled(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}[REDACTED]"

    redacted = _LABELED_SECRET.sub(_replace_labeled, redacted)

    def _replace_opaque(match: re.Match[str]) -> str:
        token = match.group(0)
        if re.fullmatch(r"[0-9]+", token):
            return token
        return "[REDACTED]"

    redacted = _LONG_OPAQUE_TOKEN.sub(_replace_opaque, redacted)
    return redacted


def curated_environment_snapshot() -> str:
    """Allowlisted environment info only: OS, Python version, filtered `pip freeze`.

    Never includes a raw environment variable dump or any value keyed by name.
    """
    lines = [
        f"os={platform.platform()}",
        f"python={sys.version.split()[0]}",
    ]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        packages = sorted(
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("-e ") and "://" not in line
        )
        lines.append("packages=" + ",".join(packages))
    except (OSError, subprocess.SubprocessError):
        lines.append("packages=<unavailable>")
    return "\n".join(lines)
