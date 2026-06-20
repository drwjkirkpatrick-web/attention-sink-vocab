"""
test_project.py
===============

pytest suite for the Attention Sink Vocab proof project.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))
from verify import check_theorem_1, check_theorem_2  # noqa: E402


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TestTheorem1:
    def test_pass(self, device):
        res = check_theorem_1()
        assert res.passed, f"Theorem 1 failed: {res.detail}"


class TestTheorem2:
    def test_pass(self, device):
        res = check_theorem_2()
        assert res.passed, f"Theorem 2 failed: {res.detail}"
