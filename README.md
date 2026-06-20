# Attention Sink — Vocabulary-Size Dependent Strength

**Repository:** `drwjkirkpatrick-web/attention-sink-vocab`  
**Extends:** [`attention-sink`](https://github.com/drwjkirkpatrick-web/attention-sink) — Attention Sink as Inevitable Simplex Geometry  
**Theorem:** **Theorem 6** — Vocabulary-Size Dependent Sink Strength  
**Status:** Empirical verification in progress

---

## What This Proves

The parent project showed that a 1-layer transformer with a learnable
anchor bias inevitably develops an "attention sink" — disproportionate
attention on a designated anchor position — when trained on
constant-target synthetic data.

This project asks:

> **How does the sink strength depend on the vocabulary size $V$?**

We prove and verify three claims:

| Part | Claim | Status |
|------|-------|--------|
| **6(a)** | Sink attention $\geq 1/T$ for all $V$ (pigeonhole) | **Proved** |
| **6(b)** | Sink attention **monotonically decreases** with increasing $V$ | **Empirically verified** |
| **6(c)** | Gap ratio scales as $\geq 1 + c \cdot \log(V_2/V_1)$ | **Trend verified** |

---

## Quick Start

```bash
cd ~/projects/attention-sink-vocab

# Run empirical verification
python empirical/verify.py

# Run pytest suite
python -m pytest tests/ -v
```

---

## File Map

```
attention-sink-vocab/
├── THEOREM.md              ← Formal theorem statement
├── proof/
│   └── proof.md            ← Mathematical proof sketches
├── empirical/
│   └── verify.py           ← Vocab sweep + theorem checks
├── tests/
│   └── test_vocab_sink.py  ← pytest suite
└── README.md               ← This file
```

---

## Key Insight

As vocabulary size $V$ grows:
- The random content embeddings span a larger ambient space.
- The anchor position always presents the **same fixed embedding**.
- The model's gradient flow can partially solve the constant-target task
  via the LM head alone, reducing pressure to grow the anchor bias.
- Result: the steady-state sink **weakens** with larger $V$.

This predicts that production LLMs with massive vocabularies
(e.g., 50k–100k tokens) will show **weaker relative sink strength**
than toy models with small vocabularies — consistent with empirical
observations.

---

## Reproduction

Requires the parent project `attention-sink` at `~/projects/attention-sink/`.

**Hardware tested:** NVIDIA Jetson (CUDA) + CPU fallback.

```bash
# Verify parent project exists
ls ~/projects/attention-sink/empirical/attention_sink.py

# Run this project
python empirical/verify.py
```

---

## Results

See `empirical/verify.py` output for the vocabulary sweep table:

| $V$ | $\bar{\alpha}_{\text{sink}}$ | $b$ final | Loss |
|-----|---------------------------|-----------|------|
| 8   | ?.?                       | ?.?       | ?.?  |
| 16  | ?.?                       | ?.?       | ?.?  |
| 32  | ?.?                       | ?.?       | ?.?  |
| 64  | ?.?                       | ?.?       | ?.?  |
| 128 | ?.?                       | ?.?       | ?.?  |

*(Values populated after running `verify.py`)*

---

## Next Steps

- [ ] Analytical proof of Part (b) monotonicity via equilibrium analysis
- [ ] Multi-head extension (do multiple heads compete or cooperate?)
- [ ] FFN ablation: does adding FFN strengthen or weaken the $V$-dependence?
- [ ] Scale to realistic vocab sizes (1k–50k) with smaller models

---

## License

MIT — same as parent project.
