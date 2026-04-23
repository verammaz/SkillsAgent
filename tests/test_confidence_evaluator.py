"""confidence_evaluator: θ gating matches proposal Alg. 2."""

import pytest


def test_theta_below_invokes_deep():
    from confidence_evaluator import should_invoke_deep_tsfm

    assert should_invoke_deep_tsfm(0.5, theta=0.8, conditional_enabled=True) is True


def test_theta_above_skips_deep():
    from confidence_evaluator import should_invoke_deep_tsfm

    assert should_invoke_deep_tsfm(0.95, theta=0.8, conditional_enabled=True) is False


def test_conditional_disabled_never_invokes():
    from confidence_evaluator import should_invoke_deep_tsfm

    assert should_invoke_deep_tsfm(0.1, theta=0.8, conditional_enabled=False) is False


def test_theta_from_env(monkeypatch):
    from confidence_evaluator import theta_from_env

    monkeypatch.setenv("RCA_CONFIDENCE_THETA", "0.7")
    assert theta_from_env() == 0.7
