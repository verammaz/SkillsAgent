"""Provider routing for ``_call_llm`` (gemini/anthropic/groq)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import skills  # noqa: E402


def _stub(name: str, value: str, calls: list):
    def fn(system, user, max_tokens=512):
        calls.append(name)
        return value
    return fn


def _fns(calls, values):
    return {name: _stub(name, val, calls) for name, val in values.items()}


def test_llm_provider_watsonx_first(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "watsonx")
    calls: list[str] = []
    monkeypatch.setattr(skills, "_PROVIDER_FNS", _fns(calls, {
        "watsonx": "ok-watsonx",
        "gemini": "ok-gemini",
        "anthropic": "ok-claude",
        "groq": "ok-groq",
    }))
    assert skills._call_llm("sys", "usr") == "ok-watsonx"
    assert calls == ["watsonx"]


def test_llm_provider_watsonx_falls_back_through_chain(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "watsonx")
    calls: list[str] = []
    monkeypatch.setattr(skills, "_PROVIDER_FNS", _fns(calls, {
        "watsonx": "",
        "gemini": "",
        "anthropic": "ok-claude",
        "groq": "ok-groq",
    }))
    assert skills._call_llm("sys", "usr") == "ok-claude"
    assert calls == ["watsonx", "gemini", "anthropic"]


def test_llm_provider_gemini_preferred(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    calls: list[str] = []
    monkeypatch.setattr(skills, "_PROVIDER_FNS", _fns(calls, {
        "watsonx": "ok-watsonx",
        "gemini": "ok-gemini",
        "anthropic": "ok-claude",
        "groq": "ok-groq",
    }))
    assert skills._call_llm("sys", "usr") == "ok-gemini"
    assert calls == ["gemini"]


def test_llm_provider_anthropic_preferred(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    calls: list[str] = []
    monkeypatch.setattr(skills, "_PROVIDER_FNS", _fns(calls, {
        "watsonx": "ok-watsonx",
        "gemini": "ok-gemini",
        "anthropic": "ok-claude",
        "groq": "ok-groq",
    }))
    assert skills._call_llm("sys", "usr") == "ok-claude"
    assert calls == ["anthropic"]


def test_gemini_helper_returns_empty_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert skills._call_gemini("sys", "usr") == ""


def test_watsonx_helper_returns_empty_without_creds(monkeypatch):
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    assert skills._call_watsonx("sys", "usr") == ""
