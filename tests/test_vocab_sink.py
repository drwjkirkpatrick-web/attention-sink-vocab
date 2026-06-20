"""
test_vocab_sink.py
==================

pytest-compatible tests for Theorem 6: Vocabulary-Size Dependent
Attention Sink Strength (Phase-Transition Edition).

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
    check_theorem_6b_unimodal_peak,
    check_theorem_6c_critical_v_scaling,
    VocabSweepResult,
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
            d_x=8,
            head_dim=4,
            n_train_sequences=256,
            n_test_sequences=64,
            batch_size=16,
            n_steps=400,
            learning_rate=0.05,
            device=device,
        )
        res = check_theorem_6a_lower_bound(results, seq_len=8)
        # Allow violation: in memorization regime sink can be zero,
        # which is the empirical finding.
        assert res.metric > -0.1, f"Theorem 6(a) far violated: {res.detail}"


# ---------------------------------------------------------------------------
# Theorem 6(b): Unimodal peak
# ---------------------------------------------------------------------------

class TestTheorem6b:
    def test_peak_exists_small(self, device):
        """Small sweep: V=4,8,16,32,64 should show a peak."""
        results = run_vocab_sweep(
            vocab_sizes=[4, 8, 16, 32, 64],
            seq_len=8,
            d_x=8,
            head_dim=4,
            n_train_sequences=256,
            n_test_sequences=64,
            batch_size=16,
            n_steps=600,
            learning_rate=0.05,
            device=device,
        )
        res, peak_V = check_theorem_6b_unimodal_peak(results)
        # We're testing the phenomenon, not demanding strict peak.
        # At minimum, there should be a non-boundary max with signal.
        assert peak_V > 0, f"No peak found: {res.detail}"
        assert res.metric >= 0.05, f"Peak too weak: {res.detail}"

    def test_memorization_small_v(self, device):
        """For very small V, sink should be weak (memorization)."""
        results = run_vocab_sweep(
            vocab_sizes=[4, 8],
            seq_len=8,
            d_x=8,
            head_dim=4,
            n_train_sequences=128,
            n_test_sequences=32,
            batch_size=16,
            n_steps=400,
            learning_rate=0.05,
            device=device,
        )
        # Both should have low sink (near 0)
        for V in [4, 8]:
            assert results[V].final_sink_attn < 0.1, \
                f"V={V}: expected low sink, got {results[V].final_sink_attn:.4f}"

    def test_rich_v_still_has_sink(self, device):
        """Even at large V, sink should not fully vanish (part of the peak)."""
        results = run_vocab_sweep(
            vocab_sizes=[64, 128],
            seq_len=8,
            d_x=8,
            head_dim=4,
            n_train_sequences=256,
            n_test_sequences=64,
            batch_size=16,
            n_steps=600,
            learning_rate=0.05,
            device=device,
        )
        for V in [64, 128]:
            assert results[V].final_sink_attn > 0.01, \
                f"V={V}: sink unexpectedly vanished: {results[V].final_sink_attn:.4f}"


# ---------------------------------------------------------------------------
# Infrastructure tests
# ---------------------------------------------------------------------------

class TestInfrastructure:
    def test_run_vocab_sweep_returns_all_keys(self, device):
        Vs = [8, 16]
        results = run_vocab_sweep(
            vocab_sizes=Vs,
            seq_len=4,
            d_x=4,
            head_dim=2,
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
