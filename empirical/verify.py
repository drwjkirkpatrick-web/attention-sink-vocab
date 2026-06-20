"""
verify.py
=========

Empirical verification of Theorem 6: Vocabulary-Size Dependent
Attention Sink Strength.

We sweep vocabulary size V and measure the steady-state sink attention
after fixed training budget.

Reuses the AttentionWithAnchorBias model and constant-target dataset
from the parent attention-sink project.

Usage:
    python empirical/verify.py
    python -m pytest tests/ -v
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Section 0: Import from parent project
# ---------------------------------------------------------------------------

PARENT_EMP = Path(__file__).parent.parent.parent / "attention-sink" / "empirical"
assert PARENT_EMP.exists(), f"Parent project not found at {PARENT_EMP}"
sys.path.insert(0, str(PARENT_EMP))

from attention_sink import (
    AttentionWithAnchorBias,
    make_sink_dataset,
    run_sink_training,
    get_device,
    manual_seed,
    TheoremResult,
)

# ---------------------------------------------------------------------------
# Section 1: Vocabulary sweep experiment
# ---------------------------------------------------------------------------

@dataclass
class VocabSweepResult:
    vocab_size: int
    final_sink_attn: float
    final_bias: float
    final_loss: float
    b_history: List[float]
    attn_history: List[float]
    loss_history: List[float]


def run_vocab_sweep(
    vocab_sizes: List[int],
    seq_len: int = 16,
    d_x: int = 32,
    head_dim: int = 16,
    n_train_sequences: int = 1024,
    n_test_sequences: int = 256,
    batch_size: int = 32,
    n_steps: int = 1500,
    learning_rate: float = 0.05,
    anchor_token: int = 0,
    device: torch.device = None,
    seed_base: int = 42,
) -> Dict[int, VocabSweepResult]:
    """Train the model at each vocabulary size with fixed hyperparameters.

    Returns a dict mapping vocab_size -> VocabSweepResult.
    """
    if device is None:
        device = get_device()

    results: Dict[int, VocabSweepResult] = {}

    for V in vocab_sizes:
        print(f"\n{'='*60}")
        print(f"Training with vocab_size = {V}")
        print(f"{'='*60}")

        manual_seed(seed_base)
        traj = run_sink_training(
            vocab_size=V,
            seq_len=seq_len,
            d_x=d_x,
            head_dim=head_dim,
            n_train_sequences=n_train_sequences,
            n_test_sequences=n_test_sequences,
            batch_size=batch_size,
            n_steps=n_steps,
            learning_rate=learning_rate,
            anchor_token=anchor_token,
            device=device,
            seed=seed_base,
            log_every=300,
        )

        results[V] = VocabSweepResult(
            vocab_size=V,
            final_sink_attn=traj["final_mean_anchor_attn"],
            final_bias=traj["final_bias"],
            final_loss=traj["loss_history"][-1] if traj["loss_history"] else float("nan"),
            b_history=traj["b_history"],
            attn_history=traj["attn_history"],
            loss_history=traj["loss_history"],
        )

        print(f"  Final sink attention: {traj['final_mean_anchor_attn']:.4f}")
        print(f"  Final bias:           {traj['final_bias']:.4f}")
        print(f"  Final loss:           {traj['loss_history'][-1]:.4f}" if traj["loss_history"] else "  N/A")

    return results


# ---------------------------------------------------------------------------
# Section 2: Theorem checks
# ---------------------------------------------------------------------------

def check_theorem_6a_lower_bound(
    results: Dict[int, VocabSweepResult],
    seq_len: int,
) -> TheoremResult:
    """Verify Part (a): sink attention >= 1/T for all V."""
    min_margin = float("inf")
    worst_V = None
    for V, r in results.items():
        margin = r.final_sink_attn - 1.0 / seq_len
        if margin < min_margin:
            min_margin = margin
            worst_V = V

    passed = min_margin >= -1e-3  # tolerance for numerical noise
    detail = (f"Worst case: V={worst_V}, sink={results[worst_V].final_sink_attn:.4f}, "
              f"1/T={1.0/seq_len:.4f}, margin={min_margin:.4e}")
    return TheoremResult(
        name="Theorem 6(a): Lower Bound >= 1/T",
        passed=passed,
        metric=min_margin,
        detail=detail,
    )


def check_theorem_6b_monotonicity(
    results: Dict[int, VocabSweepResult],
    strict: bool = True,
) -> TheoremResult:
    """Verify Part (b): sink attention decreases monotonically with V.

    We check that for each consecutive pair in the sorted vocab sizes,
    sink(V_{i+1}) < sink(V_i) [strict] or <= [non-strict].
    """
    Vs_sorted = sorted(results.keys())
    violations = []
    for i in range(len(Vs_sorted) - 1):
        V1 = Vs_sorted[i]
        V2 = Vs_sorted[i + 1]
        a1 = results[V1].final_sink_attn
        a2 = results[V2].final_sink_attn
        if strict and a2 >= a1:
            violations.append((V1, V2, a1, a2))
        elif not strict and a2 > a1:
            violations.append((V1, V2, a1, a2))

    passed = len(violations) == 0
    if violations:
        detail = f"{len(violations)} violations: " + ", ".join(
            f"V={v1}->{v2} ({a1:.4f} -> {a2:.4f})" for v1, v2, a1, a2 in violations
        )
    else:
        ratios = [
            results[Vs_sorted[i]].final_sink_attn / results[Vs_sorted[i+1]].final_sink_attn
            for i in range(len(Vs_sorted) - 1)
        ]
        detail = (f"All {len(Vs_sorted)-1} consecutive pairs monotonic. "
                  f"Ratios a(V_i)/a(V_{{i+1}}): " + ", ".join(f"{r:.3f}" for r in ratios))

    return TheoremResult(
        name=f"Theorem 6(b): Monotonicity ({'strict' if strict else 'non-strict'})",
        passed=passed,
        metric=len(violations),
        detail=detail,
    )


def check_theorem_6c_gap_scaling(
    results: Dict[int, VocabSweepResult],
) -> Tuple[TheoremResult, float]:
    """Verify Part (c): logarithmic gap scaling.

    Fit log-ratio vs log(V-ratio) and check for positive slope.
    """
    Vs_sorted = sorted(results.keys())
    if len(Vs_sorted) < 3:
        return TheoremResult(
            name="Theorem 6(c): Log Gap Scaling",
            passed=False,
            metric=float("nan"),
            detail="Need at least 3 vocab sizes to fit log scaling.",
        ), float("nan")

    xs = []  # log(V_j / V_i)
    ys = []  # a(V_i) / a(V_j) - 1
    for i in range(len(Vs_sorted)):
        for j in range(i + 1, len(Vs_sorted)):
            vi, vj = Vs_sorted[i], Vs_sorted[j]
            ai = results[vi].final_sink_attn
            aj = results[vj].final_sink_attn
            xs.append(math.log(vj / vi))
            ys.append(ai / aj - 1.0)

    # Linear regression: y = c * x + intercept
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n

    if var_x < 1e-12:
        return TheoremResult(
            name="Theorem 6(c): Log Gap Scaling",
            passed=False,
            metric=float("nan"),
            detail="Zero variance in log(V) — insufficient spread.",
        ), float("nan")

    c_est = cov_xy / var_x
    intercept = mean_y - c_est * mean_x

    # R^2
    ss_res = sum((y - (c_est * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    passed = c_est > 0.01  # positive slope, not noise
    detail = (f"Estimated c = {c_est:.4f}, R^2 = {r2:.4f}, "
              f"intercept = {intercept:.4f} ({n} pairs)")

    return TheoremResult(
        name="Theorem 6(c): Log Gap Scaling",
        passed=passed,
        metric=c_est,
        detail=detail,
    ), r2


# ---------------------------------------------------------------------------
# Section 3: Summary printer
# ---------------------------------------------------------------------------

def print_summary_table(results: Dict[int, VocabSweepResult]) -> None:
    print("\n" + "=" * 60)
    print("VOCABULARY SWEEP RESULTS")
    print("=" * 60)
    print(f"{'V':>6} {'Sink Attn':>10} {'Bias':>10} {'Loss':>10}")
    print("-" * 40)
    for V in sorted(results.keys()):
        r = results[V]
        print(f"{V:>6} {r.final_sink_attn:>10.4f} {r.final_bias:>10.4f} "
              f"{r.final_loss:>10.4f}")
    print("-" * 40)


# ---------------------------------------------------------------------------
# Section 4: Main runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print(" Theorem 6: Vocabulary-Size Dependent Sink Strength")
    print(" Empirical Verification")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Sweep parameters (Jetson-friendly)
    VOCAB_SIZES = [8, 16, 32, 64, 128]
    SEQ_LEN = 16
    D_X = 32
    HEAD_DIM = 16
    N_TRAIN = 1024
    N_TEST = 256
    BATCH = 32
    N_STEPS = 1500
    LR = 0.05

    print("Configuration:")
    print(f"  Vocabulary sizes: {VOCAB_SIZES}")
    print(f"  Sequence length:  {SEQ_LEN}")
    print(f"  Embed dim d_x:    {D_X}")
    print(f"  Head dim:         {HEAD_DIM}")
    print(f"  Train sequences:  {N_TRAIN}")
    print(f"  Steps:            {N_STEPS}")
    print(f"  Learning rate:    {LR}")
    print()

    t0 = time.time()
    results = run_vocab_sweep(
        vocab_sizes=VOCAB_SIZES,
        seq_len=SEQ_LEN,
        d_x=D_X,
        head_dim=HEAD_DIM,
        n_train_sequences=N_TRAIN,
        n_test_sequences=N_TEST,
        batch_size=BATCH,
        n_steps=N_STEPS,
        learning_rate=LR,
        device=device,
    )
    elapsed = time.time() - t0

    print_summary_table(results)

    # Run theorem checks
    print("\n" + "=" * 60)
    print("THEOREM CHECKS")
    print("=" * 60)

    res_6a = check_theorem_6a_lower_bound(results, SEQ_LEN)
    print(f"\n{'PASS' if res_6a.passed else 'FAIL'} — {res_6a.name}")
    print(f"  Metric: {res_6a.metric:.4e}")
    print(f"  Detail: {res_6a.detail}")

    res_6b = check_theorem_6b_monotonicity(results, strict=True)
    print(f"\n{'PASS' if res_6b.passed else 'FAIL'} — {res_6b.name}")
    print(f"  Metric: {res_6b.metric}")
    print(f"  Detail: {res_6b.detail}")

    res_6c, r2 = check_theorem_6c_gap_scaling(results)
    print(f"\n{'PASS' if res_6c.passed else 'FAIL'} — {res_6c.name}")
    print(f"  Metric (c_est): {res_6c.metric:.4f}")
    print(f"  R^2: {r2:.4f}")
    print(f"  Detail: {res_6c.detail}")

    overall = all([res_6a.passed, res_6b.passed, res_6c.passed])
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
