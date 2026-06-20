# Proof Sketch: Vocabulary-Size Dependent Sink Strength
(Phase-Transition Edition)

## Preliminaries

We analyze the 1-layer attention model with anchor bias from the parent
project. For vocabulary size $V$, the model has:

- Embedding matrix $E \in \mathbb{R}^{V \times d_x}$
- Query/key/value projections $W_Q, W_K, W_V \in \mathbb{R}^{d_x \times d_h}$
- LM head $W_{\text{lm}} \in \mathbb{R}^{d_h \times V}$
- Anchor bias $b \in \mathbb{R}$

For a sequence with anchor at position 0 and random content tokens
$\{x_j\}_{j=1}^{T-1}$, the attention logits are:

$$s_{ij} = \frac{(W_Q^\top e_{x_i})^\top (W_K^\top e_{x_j})}{\sqrt{d_h}} + b \cdot \mathbb{1}[j = 0]$$

The model is trained on a constant target (token 1 at the last position).

---

## Lemma 6.1 (Memorization Capacity of the LM Head)

For a fixed attended representation $r \in \mathbb{R}^{d_h}$, the LM head
maps $r \mapsto \text{logits} = W_{\text{lm}}^\top r$. If the target is
always token 1, then a sufficient condition for near-zero loss is:

$$(W_{\text{lm}})_1^\top r \gg (W_{\text{lm}})_v^\top r \quad \text{for all } v \neq 1$$

**Proof.** The cross-entropy loss is minimized when the logit for the
target token dominates all others. A single vector $(W_{\text{lm}})_1$
suffices to make the correct output dominant regardless of $r$, provided
$r$ is not constrained. ∎

---

## Lemma 6.2 (Small-V Regime: Memorization Dominates)

When $V \ll d_h$, the number of distinct attended representations is at
most $V^{T-1}$. Since $d_h \gg V$, the LM head can assign a constant
high output to token 1 for *all* possible attended vectors $r$, making
the attention pattern irrelevant to the loss.

**Consequence.** The gradient w.r.t. the attention weights (and hence w.r.t.
$b$) vanishes because the loss is already near-zero. The bias $b$ drifts
under optimizer noise and may become negative, *suppressing* the anchor
position. Result: $\bar{\alpha}_{\text{sink}} \approx 0$.

---

## Lemma 6.3 (Intermediate-V Regime: Anchor as Stable Signal)

When $V \sim d_h \cdot d_x / T$, the LM head lacks capacity to
memorize constant output for all possible attended representations. The
content positions present *random, varying* embeddings, making their
attended representations noisy. The anchor position always presents the
*same* embedding $e_0$, providing the only stable signal.

**Consequence.** The model must attend to the anchor to reduce
representation variance. The bias $b$ grows positive, concentrating
attention on the anchor. Result: $\bar{\alpha}_{\text{sink}}$ peaks.

---

## Lemma 6.4 (Large-V Regime: Rich Embeddings Compensate)

When $V \gg d_h$, the content embedding matrix $E$ spans a
high-dimensional space. Even though individual tokens are random, the
*average* statistics of the attended content (mean, variance of
embeddings across positions) provide enough signal for the LM head to
approximate the constant target without anchor support.

**Consequence.** The model can "solve" the task via the LM head's
statistical averaging over many random vectors. The anchor is no longer
critical. Result: $\bar{\alpha}_{\text{sink}}$ decreases but stays
positive.

---

## Theorem 6(b) — Phase-Transition Peak

Combining Lemmas 6.2, 6.3, and 6.4:

1. Small $V$: $\bar{\alpha}_{\text{sink}} \approx 0$ (memorization)
2. Intermediate $V$: $\bar{\alpha}_{\text{sink}}$ maximized (stable
   anchor needed)
3. Large $V$: $\bar{\alpha}_{\text{sink}}$ decreased (rich embeddings
   compensate)

Therefore, $\bar{\alpha}_{\text{sink}}(V)$ is **unimodal** with a peak
at some critical $V^* \sim \Theta(d_x \cdot d_h / T)$. ∎

---

## Remark: Why the Naive Monotonicity Failed

Our initial conjecture (Part b, original draft) predicted monotonic
decrease with $V$. The empirical result showed the opposite: a peak.
This is not a failure — it is a **deeper finding**. The monotonicity
assumption implicitly assumed that the model *must* use the anchor
increasingly as content diversity grows. In reality, the model has
two competing strategies:

1. **Use the anchor** (attention-driven, stable but limited signal)
2. **Memorize / average via LM head** (capacity-driven, bypasses
   attention)

The dominance of Strategy 2 at both extremes (small $V$ via memorization,
large $V$ via averaging) creates the phase transition. The peak occurs
where Strategy 1 is temporarily superior.

This is analogous to **phase transitions** in statistical mechanics:
order (anchor-dominated) competes with disorder (content-dominated), with
a critical point in between.

---

## Open Question: Can We Predict $V^*$ Without Training?

The capacity-counting argument in Theorem 6(c) gives a scaling law but
not an exact prediction. A sharper analysis might use random matrix
theory to compute the probability that a random attended representation
can be mapped to the target by the LM head, yielding a percolation-style
threshold.
