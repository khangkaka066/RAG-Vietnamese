from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_openrouter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no developer/CI machine's real OpenRouter credentials leak into
    tests, so the whole suite stays deterministic and offline by default."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
