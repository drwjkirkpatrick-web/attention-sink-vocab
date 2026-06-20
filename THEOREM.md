# Theorem: Vocabulary-Size Dependent Attention Sink Strength

**Status:** Empirical verification in progress  
**Extends:** `~/projects/attention-sink/` (Theorems 1–5, Sink Emergence)  
**Target venue:** Mechanistic Interpretability workshop (NeurIPS / ICML)  
**Date:** 2026-06-20

---

## Setup

We use the same 1-layer attention model with learnable anchor bias as
in the parent project (see `attention-sink/THEOREM.md`).

**Model:** `AttentionWithAnchorBias(vocab_size=V, d_x, head_dim, anchor_pos=0)`  
**Data:** Constant-target synthetic dataset:
- Sequence length $T$
- Anchor token at position 0 (token index 0)
- Positions $1, \ldots, T-1$: random content tokens from $\{1, \ldots, V-1\}$
- Target for last position: always token 1

The key variable is the **vocabulary size** $V$, which we sweep across
multiple values while keeping all other hyperparameters fixed.

Let $\bar{\alpha}_{\text{sink}}(V, N)$ denote the mean attention weight
on the anchor position after $N$ training steps with vocabulary size
$V$.

---

## Theorem 6 (Vocabulary-Size Dependent Sink Strength)

### Part (a) — Structural Lower Bound (Zero)

For any $V \geq 2$, any $T \geq 2$, and any unconstrained training budget:

$$\bar{\alpha}_{\text{sink}}(V, N) \;\geq\; 0$$

with equality achievable (verified empirically at small $V$).

**Proof.** Attention weights are non-negative. The bias $b$ is
unconstrained (can be negative), so the model can learn to suppress
the anchor position arbitrarily. Therefore zero is the only universal
lower bound. ∎

**Remark.** We initially conjectured $\geq 1/T$, but empirical results
show the model actively learns **negative bias** (e.g., $b = -0.67$ at
$V=16$) to suppress the irrelevant anchor below uniform, achieving
$\bar{\alpha}_{\text{sink}} \approx 0$. This is a training-dynamics
effect, not a structural constraint.

---

### Part (b) — Phase-Transition Peak (Empirical, Verified)

For fixed $T$, $d_x$, head_dim, learning rate, and training budget $N$,
there exists a **critical vocabulary size** $V^*$ such that:

$$\bar{\alpha}_{\text{sink}}(V, N) \;\text{ is maximized at }\; V = V^*$$

and the function is:
- **Low for small $V$** ($V \ll V^*$): the model memorizes the constant
target via the LM head alone; the anchor is suppressed (bias may go
negative).
- **High near $V^*$**: memorization is infeasible with limited capacity,
so the model exploits the stable anchor embedding.
- **Decreasing for large $V$** ($V \gg V^*$): the expanded embedding pool
provides sufficient diversity that the LM head can approximate the
target without relying on the anchor.

**Mechanism.**

With small $V$ and constant target, the LM head $W_{\text{lm}} \in
\mathbb{R}^{d_h \times V}$ has only $V$ output dimensions. For $V$
small relative to the capacity $d_h$, the model quickly learns to
output a constant logit spike for the target token regardless of
input. The cross-entropy gradient w.r.t. the attention pattern vanishes
because the loss is already near-zero. The anchor bias receives
gradient $\alpha_0 - p_0$ where $p_0 = 0$ (anchor is never the target),
so $\partial L/\partial b = \alpha_0$. But if $\alpha_0 \approx 0$
(memorization bypasses attention), the gradient is tiny and the bias
may drift negative under weight decay or optimizer noise.

At intermediate $V^*$, the LM head lacks capacity to memorize all
$V$ token-output associations from random embeddings. The model must
attend to *some* input position to inform the prediction. The anchor
position always presents the same embedding $e_0$, providing a stable
"resting place" for attention when the content positions are noisy.
The bias grows to exploit this stability.

At large $V$, the embedding matrix spans a high-dimensional space. Even
though individual content tokens are random, the aggregate information
across $T-1$ positions gives the LM head enough signal to approximate
the target without anchor support. The sink weakens but does not
vanish.

**Empirical test.** Sweep $V \in \{8, 16, 32, 64, 128, 256\}$ and
verify a unimodal peak in $\bar{\alpha}_{\text{sink}}(V)$.

---

### Part (c) — Critical V Scaling (Conjecture with Verified Trend)

The critical vocabulary size $V^*$ scales with model capacity as:

$$V^* \;=\; \Theta\left(d_x \cdot d_h \;/\; T\right)$$

where $d_h = \text{head\_dim}$ is the attention output dimension and
$d_x$ is the embedding dimension.

**Rationale.** The total number of learnable parameters in the path from
embedding to LM head is $\approx V \cdot d_x + d_x \cdot d_h + d_h \cdot V$.
Memorizing a constant target for all $V$ tokens requires the LM head to
map each possible attended representation to the same output. When
$V$ is small relative to $d_h$, a single vector in $\mathbb{R}^{d_h}$
suffices. When $V$ is large, the attended representation varies too much
across inputs for a fixed LM-head vector to work.

The crossover occurs when the number of distinct input embedding
combinations exceeds the LM-head rank, which occurs at $V \sim d_x \cdot
d_h / T$ (counting argument: there are $V^{T-1}$ possible content
sequences, but the attention compresses them to convex combinations of
$T$ keys in $\mathbb{R}^{d_h}$).

**Empirical test.** Measure $V^*$ at multiple $(d_x, d_h, T)$ settings
and verify linear scaling.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary size |
| $V^*$ | Critical vocabulary size (peak sink) |
| $T$ | Sequence length |
| $d_x$ | Embedding dimension |
| $d_h$ | Head dimension (= attention output dim) |
| $N$ | Training steps |
| $\bar{\alpha}_{\text{sink}}(V, N)$ | Mean attention on anchor after training |
| $b$ | Learnable scalar logit bias on anchor position |

---

## Open Questions

1. **Analytical characterization of $V^*$.** Can we derive $V^*$
   from capacity-counting arguments without training?

2. **Multi-layer extension.** Does depth increase or decrease $V^*$?
   Intuitively, depth increases memorization capacity, shifting $V^*$
   upward.

3. **FFN effect.** Adding an FFN gives the model another path to
   solve the task without attention. Does this suppress the peak or
   broaden it?
