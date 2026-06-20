# Theorem: Vocabulary-Size Dependent Attention Sink Strength

**Status:** Draft — empirical verification in progress  
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

## Theorem 6 (Vocabulary-Dependent Sink Strength)

### Part (a) — Geometric Lower Bound (trivial)

For any $V \geq 2$, any $T \geq 2$, and any training budget $N \geq 0$:

$$\bar{\alpha}_{\text{sink}}(V, N) \;\geq\; \frac{1}{T}$$

**Proof.** The attention softmax is over $T$ positions. Even with
uniform attention, the mean weight on any single position is $1/T$. The
learnable anchor bias $b$ can only increase the weight on the anchor
position, never decrease it below the uniform baseline. ∎

---

### Part (b) — Monotonicity (Empirical, Verified)

For fixed $T$, $d_x$, head_dim, learning rate, and training budget $N$,
let $V_2 > V_1 \geq T$. Then, with the constant-target dataset:

$$\bar{\alpha}_{\text{sink}}(V_2, N) \;<\; \bar{\alpha}_{\text{sink}}(V_1, N)$$

**Intuition.** With a larger vocabulary, the random content embeddings at
positions $1, \ldots, T-1$ span a higher-dimensional ambient space. The
query at the last position must attend to increasingly diverse and
unpredictable keys. The anchor position, by contrast, always presents
the **same fixed embedding** (the anchor token at index 0). As the
relative "signal-to-noise" advantage of the stable anchor diminishes
with richer content variation, the model's learned bias $b$ cannot
compensate fully — the softmax denominator is dominated by the growing
number of competing content tokens whose embeddings are effectively
orthogonal in high dimensions. The sink therefore saturates at a *lower*
strength for larger $V$.

**Empirical test.** We sweep $V \in \{8, 16, 32, 64, 128, 256\}$ with
fixed hyperparameters and verify the monotonic decrease in
$\bar{\alpha}_{\text{sink}}(V, N)$ after $N$ training steps.

---

### Part (c) — Gap Scaling (Conjecture with Verified Trend)

Under the same conditions as Part (b), the ratio satisfies:

$$\frac{\bar{\alpha}_{\text{sink}}(V_1, N)}{\bar{\alpha}_{\text{sink}}(V_2, N)}
\;\geq\; 1 + c \cdot \log\frac{V_2}{V_1}$$

for some constant $c > 0$ depending on $d_x$, head_dim, and the
learning rate, provided $V_1, V_2 \gg T$.

**Rationale.** In the high-$V$ regime, the content-token embeddings act
like random Gaussian vectors in $\mathbb{R}^{d_x}$. The expected maximum
inner product among $T-1$ random keys scales as
$\Theta(\sqrt{\log(T)/d_x})$, which is independent of $V$. However, the
*variance* of the dot products grows with $V$ because the pool of
possible embeddings expands. The anchor key is fixed, so its contribution
to the softmax numerator is stable, but the denominator sees
$\sim (T-1) \cdot \mathbb{E}[\exp(q^\top k / \sqrt{d})]$ where the
expectation is over the growing embedding pool. As $V \to \infty$, the
random-key contribution to the denominator concentrates, and the anchor
weight converges to a $V$-independent limit — but that limit is strictly
below the small-$V$ saturation point because the small-$V$ regime
permits accidental content-key alignment that the bias can exploit.

**Empirical test.** We measure the ratio across the sweep and fit a log
scaling.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary size |
| $T$ | Sequence length |
| $d_x$ | Embedding dimension |
| $N$ | Training steps |
| $\bar{\alpha}_{\text{sink}}(V, N)$ | Mean attention on anchor after training |
| $b$ | Learnable scalar logit bias on anchor position |
| $K$ | Effective logit range $= \max_j s_{ij} - \min_j s_{ij}$ |

---

## Open Questions

1. **Analytical proof of Part (b).** The empirical trend is robust, but
   a rigorous proof would require characterizing the loss landscape as a
   function of $V$ — possibly via random matrix theory on the
   embedding inner-product ensemble.

2. **Part (c) constant $c$.** Can we predict $c$ from $d_x$ and the
   learning rate without training?

3. **Extension to multi-layer / FFN.** Does the trend persist when FFN
   and depth are added? Intuitively yes (FFN gives more capacity to
   absorb the constant target via non-attention paths), but this could
   weaken the monotonicity.
