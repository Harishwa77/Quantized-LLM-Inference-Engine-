"""Shared fixtures for the llmopt test suite."""

from __future__ import annotations

import pytest

from _helpers import build_engine


@pytest.fixture
def engine():
    return build_engine()
