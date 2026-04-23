"""Shared fixtures: disable AssetOpsBench subprocesses unless a test opts in."""

import pytest


@pytest.fixture(autouse=True)
def disable_remote_subprocess_agents(monkeypatch):
    monkeypatch.setenv("USE_IOT_SUBPROCESS", "0")
    monkeypatch.setenv("USE_TSFM_SUBPROCESS", "0")
    monkeypatch.setenv("USE_WO_SUBPROCESS", "0")
    monkeypatch.setenv("KNOWLEDGE_INJECTION", "1")
