"""
test_vocab_sink.py
==================

pytest-compatible tests for Theorem 6: Vocabulary-Size Dependent
Attention Sink Strength.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Import from parent project.
PARENT_EMP = Path(__file__).parent.parent.parent / "attention-sink" / "empirical"
assert PARENT_EMP.exists()
sys.path.insert(0, str(PARENT_EMP))

from attention_sink import get_device, manual_seed  # noqa: E402

# Import from this project's empirical/.
EMP = Path(__file__).parent.parent / "empirical"
sys.path.insert(0, str(EMP))

from verify import (  # noqa: E402
    run_vocab_sweep,
    check_theorem_6a_lower_bound,
    check_theorem_6b_monotonicity,
    check_theorem_6c_gap_scaling,
)


@pytest.fixture(scope="module")
def device():
    return get_device()


# ---------------------------------------------------------------------------
# Theorem 6(a): Lower bound >= 1/T
# ---------------------------------------------------------------------------

class TestTheorem6a:
    def test_simple_lower_bound(self, device):
        """Quick sweep with small vocab sizes and short training."""
        results = run_vocab_sweep(
            vocab_sizes=[8, 16, 32],
            seq_len=8,
            d_x=16,
            head_dim=8,
            n_train_sequences=256,
            n_test_sequences=64,
            batch_size=16,
            n_steps=400,
            learning_rate=0.05,
            device=device,
        )
        res = check_theorem_6a_lower_bound(results, seq_len=8)
        assert res.passed, f"Theorem 6(a) failed: {res.detail}"


# ---------------------------------------------------------------------------
# Theorem 6(b): Monotonicity
# ---------------------------------------------------------------------------

class TestTheorem6b:
    def test_monotonicity_small(self, device):
        """Small sweep: V=8,16,32 should show monotonic decrease."""
        results = run_vocab_sweep(
            vocab_sizes=[8, 16, 32],
            seq_len=8,
            d_x=16,
            head_dim=8,
            n_train_sequences=256,
            n_test_sequences=64,
            batch_size=16,
            n_steps=400,
            learning_rate=0.05,
            device=device,
        )
        res = check_theorem_6b_monotonicity(results, strict=True)
        assert res.passed, f"Theorem 6(b) failed: {res.detail}"

    def test_monotonicity_medium(self, device):
        """Medium sweep with more points."""
        results = run_vocab_sweep(
            vocab_sizes=[8, 16, 32, 64],
            seq_len=12,
            d_x=24,
            head_dim=12,
            n_train_sequences=512,
            n_test_sequences=128,
            batch_size=24,
            n_steps=800,
            learning_rate=0.05,
            device=device,
        )
        res = check_theorem_6b_monotonicity(results, strict=False)
        assert res.passed, f"Theorem 6(b) medium failed: {res.detail}"


# ---------------------------------------------------------------------------
# Theorem 6(c): Gap scaling
# ---------------------------------------------------------------------------

class TestTheorem6c:
    def test_log_scaling_positive_slope(self, device):
        """Fit log-gap and verify positive slope c > 0."""
        results = run_vocab_sweep(
            vocab_sizes=[8, 16, 32, 64, 128],
            seq_len=12,
            d_x=24,
            head_dim=12,
            n_train_sequences=512,
            n_test_sequences=128,
            batch_size=24,
            n_steps=800,
            learning_rate=0.05,
            device=device,
        )
        res, r2 = check_theorem_6c_gap_scaling(results)
        assert res.passed, f"Theorem 6(c) failed: {res.detail}"
        assert r2 > 0.5, f"Poor fit: R^2 = {r2:.3f}"


# ---------------------------------------------------------------------------
# Unit tests on the sweep infrastructure
# ---------------------------------------------------------------------------

class TestInfrastructure:
    def test_run_vocab_sweep_returns_all_keys(self, device):
        Vs = [8, 16]
        results = run_vocab_sweep(
            vocab_sizes=Vs,
            seq_len=4,
            d_x=8,
            head_dim=4,
            n_train_sequences=32,
            n_test_sequences=8,
            batch_size=8,
            n_steps=50,
            learning_rate=0.05,
            device=device,
        )
        assert sorted(results.keys()) == Vs
        for V in Vs:
            assert results[V].vocab_size == V
            assert 0 <= results[V].final_sink_attn <= 1
            assert isinstance(results[V].b_history, list)
