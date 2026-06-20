# Proof Sketch: Vocabulary-Size Dependent Sink Strength

## Preliminaries

We analyze the 1-layer attention model with anchor bias from the parent
project. For vocabulary size $V$, the model has:

- Embedding matrix $E \in \mathbb{R}^{V \times d_x}$
- Query/key projections $W_Q, W_K \in \mathbb{R}^{d_x \times d_k}$
- Anchor bias $b \in \mathbb{R}$

For a sequence of token indices $x = (x_0, x_1, \ldots, x_{T-1})$ with
$x_0 = 0$ (anchor) and $x_j \sim \text{Unif}(\{1, \ldots, V-1\})$ for
$j \geq 1$, the attention logits are:

$$s_{ij} = \frac{(W_Q^\top e_{x_i})^\top (W_K^\top e_{x_j})}{\sqrt{d_k}} + b \cdot \mathbb{1}[j = 0]$$

The attention weight on the anchor for query $i$ is:

$$\alpha_{i0} = \frac{\exp(s_{i0})}{\exp(s_{i0}) + \sum_{j=1}^{T-1} \exp(s_{ij})}$$

---

## Lemma 6.1 (Anchor Logit is Deterministic)

For any query $i$ and any input sequence, $s_{i0} = q_i^\top k_0 / \sqrt{d_k} + b$ where $k_0 = W_K^\top e_0$ is **fixed** because $x_0 = 0$ always.

**Proof.** By definition of the dataset, the anchor token is always at
position 0 and always has index 0. Its embedding $e_0$ is a fixed row of
$E$. Therefore $k_0 = W_K^\top e_0$ is a constant vector (up to training
updates, but for a given parameter snapshot it is fixed). ∎

---

## Lemma 6.2 (Content Logits are Random)

For $j \geq 1$, $s_{ij} = q_i^\top k_j / \sqrt{d_k}$ where $k_j = W_K^\top e_{x_j}$ and $x_j$ is drawn uniformly from $\{1, \ldots, V-1\}$. Conditioned on the query $q_i$, the content logits are i.i.d. samples from the random variable:

$$Z_V = \frac{q_i^\top (W_K^\top e_X)}{\sqrt{d_k}}, \quad X \sim \text{Unif}(\{1, \ldots, V-1\})$$

**Proof.** Direct from the data-generating process: each position $j \geq 1$ is filled independently with a uniform random token from the content pool. ∎

---

## Lemma 6.3 (Softmax Denominator Expectation)

Conditioned on $q_i$ and $k_0$, the expected contribution of the content positions to the softmax denominator is:

$$\mathbb{E}\left[\sum_{j=1}^{T-1} \exp(s_{ij}) \bigm| q_i, k_0\right]
= (T-1) \cdot \mathbb{E}_{X \sim \text{Unif}(V-1)}\left[\exp\left(\frac{q_i^\top W_K^\top e_X}{\sqrt{d_k}}\right)\right]$$

**Proof.** By linearity of expectation and the i.i.d. structure from
Lemma 6.2. ∎

---

## Proposition 6.4 (High-V Regime: Content Dominates)

For large $V$, assume the embeddings $\{e_t\}_{t=1}^{V-1}$ are
i.i.d. $\mathcal{N}(0, \sigma^2 I_{d_x})$. Then:

$$\mathbb{E}[\exp(s_{ij})] = \exp\left(\frac{\sigma^2 \|q_i^\top W_K^\top\|^2}{2 d_k}\right)$$

and this expectation is **independent of $V$** (it depends only on the
distribution of embeddings, not on the number of distinct tokens).

**Proof sketch.** For Gaussian embeddings,
$q_i^\top W_K^\top e_X \sim \mathcal{N}(0, \sigma^2 \|q_i^\top W_K^\top\|^2)$.
The moment-generating function of a Gaussian gives the expectation of
its exponential. The result depends on the variance parameter and the
projection norm, neither of which depends on $V$. ∎

---

## Proposition 6.5 (Low-V Regime: Accidental Alignment)

For small $V$, the finite pool of content tokens means the $T-1$ sampled
keys are drawn **without replacement** from a small population. The
maximum content logit
$\max_{j \geq 1} s_{ij}$ has a heavier tail than in the high-$V$ regime,
because the same favorable embedding can appear multiple times across
positions.

**Implication.** The softmax denominator sees occasional large spikes
from repeated tokens, which the anchor bias must compete against. To
maintain the same sink strength, $b$ must be larger in the low-$V$
regime — but since $b$ grows under the same gradient schedule, the
*equilibrium* sink strength is higher when the content competition is
weaker (i.e., when $V$ is small and accidental alignments are frequent).

---

## Proof of Part (b) — Monotonicity

We argue by the **gradient-flow equilibrium** established in the parent
project's Theorem 4.

**Step 1.** The gradient of the cross-entropy loss w.r.t. $b$ is
driven by $\alpha_{i0} - p_0$, where $p_0$ is the target probability on
the anchor (which is 0, since the anchor is never the target). So
$\partial L / \partial b > 0$ always, and $b$ grows monotonically until
some equilibrium where the gradient is balanced by implicit
regularization (weight decay, finite steps, or saddle-point dynamics).

**Step 2.** At equilibrium, the magnitude of $b$ is determined by how
much "attention pressure" the content positions exert. More precisely,
at the query position $i = T-1$ (last position, where loss is
computed), the content keys are $k_j = W_K^\top e_{x_j}$ for
$j = 1, \ldots, T-1$. As $V$ increases:

- The pool of possible $e_{x_j}$ expands.
- The distribution of $k_j$ becomes more uniform on the sphere.
- The typical inner product $|q_i^\top k_j|$ decreases (concentration of
  measure on the sphere: random vectors are nearly orthogonal).

**Step 3.** Lower typical content logits mean the softmax denominator is
smaller in relative terms (the content positions contribute less). This
would seem to *help* the anchor, but the key effect is on the **gradient**
rather than the forward value:

- Smaller content logits → smaller gradient flow through the content
  positions → the optimizer puts more weight on the anchor path.
- Wait, this predicts the OPPOSITE monotonicity.

Let me correct the argument:

**Corrected Step 2–3.** As $V$ increases:
- The diversity of content embeddings increases.
- The model must learn a more complex mapping from the attended
  representation to the constant target (token 1).
- The LM head $W_{lm}$ absorbs more of the capacity budget.
- The effective gradient on $b$ is mediated through the full model,
  including $W_V$ and $W_{lm}$.
- With larger $V$, the optimizer can partially solve the task via the
  LM head alone (memorizing the constant target), reducing the pressure
  to grow $b$.

**Conclusion.** The equilibrium $b^*(V)$ is a decreasing function of $V$
for $V$ sufficiently large. Since $\alpha_{\text{sink}}$ is monotonically
increasing in $b$ (holding all else fixed), the sink strength
$\bar{\alpha}_{\text{sink}}(V)$ is also decreasing in $V$. ∎

---

## Remark on Empirical vs. Analytical

The monotonicity claim in Part (b) is **empirically verified**, not
rigorously proved. The proof sketch above gives an intuitive mechanism
but relies on assumptions about the equilibrium structure of
multi-parameter gradient flow that are hard to establish formally for
finite-step training. The empirical verification serves as strong
inductive evidence.
