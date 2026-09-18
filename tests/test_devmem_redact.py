from __future__ import annotations

from devmem.redact import curated_environment_snapshot, redact_text


def test_redact_text_masks_labeled_api_key():
    text = "config: api_key=sk-abcdef1234567890 loaded"
    redacted = redact_text(text)
    assert "sk-abcdef1234567890" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_text_masks_labeled_password():
    text = "login with password=hunter2SuperSecretValue now"
    redacted = redact_text(text)
    assert "hunter2SuperSecretValue" not in redacted


def test_redact_text_masks_bearer_token():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    redacted = redact_text(text)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted


def test_redact_text_leaves_ordinary_text_untouched():
    text = "pytest failed with 3 errors in test_math.py"
    assert redact_text(text) == text


def test_redact_text_masks_env_var_value(monkeypatch):
    monkeypatch.setenv("MY_SECRET_ENV", "leaked-value-should-not-appear")
    text = "using value leaked-value-should-not-appear in config"
    redacted = redact_text(text)
    assert "leaked-value-should-not-appear" not in redacted


def test_curated_environment_snapshot_has_no_raw_env_dump(monkeypatch):
    monkeypatch.setenv("SOME_SECRET_TOKEN", "should-never-appear-in-snapshot")
    snapshot = curated_environment_snapshot()
    assert "should-never-appear-in-snapshot" not in snapshot
    assert "SOME_SECRET_TOKEN" not in snapshot


def test_curated_environment_snapshot_includes_python_and_os_info():
    snapshot = curated_environment_snapshot()
    assert "python" in snapshot.lower()
