# SCRIBE: Sequential Compositional Reasoning with Iterative Belief Encoding

**Author:** Anand Trivedi (Independent Researcher)  
**Paper:** [arXiv link coming soon]  
**Contact:** trivedi.anand029@gmail.com

---

## Overview

SCRIBE is a lightweight neural architecture for multi-hop reasoning that processes sentences **sequentially** and manages a persistent memory through **attention-based, confidence-weighted write operations**. Unlike standard attention models that see all context at once (enabling reasoning shortcuts), SCRIBE enforces a causal constraint: each sentence can only access information stored by preceding sentences.

### Key Results

| Model | Params | Overall | bAbI | ProofWriter | Easy | Med | Hard |
|-------|--------|---------|------|-------------|------|-----|------|
| MLP Baseline | 0.2M | 70.2% | 48.5% | 91.8% | 72.4% | 56.6% | 73.4% |
| LSTM Baseline | 0.8M | 75.5% | 59.1% | 91.8% | 75.9% | 66.0% | 77.9% |
| Flat Attention | 0.7M | 79.0% | 64.4% | 93.6% | 79.0% | 73.1% | 80.5% |
| **SCRIBE** | **1.1M** | **85.8%** | **78.0%** | **93.7%** | **82.1%** | **85.2%** | **86.5%** |

### Key Finding: Inverse Depth Scaling

Standard models degrade on harder examples. SCRIBE shows the **opposite** — harder problems are solved *more* accurately:

- **Easy (1-3 sentences):** 82.1%
- **Medium (4-6 sentences):** 85.2%  
- **Hard (7+ sentences):** 86.5%

This suggests genuine compositional reasoning rather than shallow pattern matching.

---



**Core components:**
- **Sequential processing** — causal constraint prevents reasoning shortcuts
- **Confidence-weighted writes** — uncertain writes are soft and overwritable (key mechanism)
- **Sentinel slot** — learned null slot absorbs irrelevant information
- **Multi-hop reader** — 2-hop gated attention over final memory state

**What we tried and removed (via ablation):**
- ~~Memory revision gate~~ — negligible impact (-0.2%)
- ~~Periodic consolidation~~ — actually hurts performance (-5.5%)

---

## Quick Start

### Requirements

```bash
pip install torch pandas sentence-transformers tqdm matplotlib
```

### 1. Prepare Dataset

The training uses a unified parquet file combining bAbI and ProofWriter. Place `unified_reasoning_dataset.parquet` in your working directory.

### 2. Train SCRIBE

Run the complete training script (single file, works in Colab or locally):

```bash
python scribe_train.py
```

Or copy-paste into a Google Colab cell. Trains in ~12 minutes on a T4 GPU.

### 3. Run All Experiments (Baselines + Ablations)

```bash
python scribe_experiments.py
```

Runs 8 experiments (~2 hours total on T4):
- 3 baselines (MLP, LSTM, Flat Attention)
- SCRIBE (Full)
- 4 ablations (no revision, no consolidation, no sentinel, no confidence)

Prints a final comparison table.

---

## Repository Structure

```
SCRIBE/
├── README.md                  # This file
├── SCRIBE_Paper_v2.pdf        # Paper (PDF)
├── SCRIBE_Paper_v2.tex        # Paper (LaTeX source)
├── scribe_train.py            # Complete training script
├── scribe_experiments.py      # Baselines + ablations script
├── preencode_dataset.py       # Optional: pre-encode large datasets
└── experiments/
    └── scribe_best.pt         # Best model checkpoint
```

---

## Ablation Results

| Configuration | Params | Overall | bAbI | Hard |
|--------------|--------|---------|------|------|
| SCRIBE (all components) | 1.4M | 80.3% | 70.5% | 80.6% |
| − No Revision | 1.3M | 80.1% | 69.8% | 80.5% |
| − No Sentinel | 1.4M | 80.0% | 70.2% | 80.2% |
| − No Confidence | 1.4M | 77.9% | 65.6% | 78.9% |
| **− No Consolidation** | **1.1M** | **85.8%** | **78.0%** | **86.5%** |

**Takeaway:** Confidence weighting is essential. Consolidation hurts. The simplest architecture wins.

---

## Citation

```bibtex
@article{trivedi2026scribe,
  title={SCRIBE: Sequential Compositional Reasoning with Iterative Belief Encoding},
  author={Trivedi, Anand},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## License

MIT License

## Acknowledgments

Built with PyTorch and Sentence-Transformers. Trained on bAbI (Weston et al., 2016) and ProofWriter (Tafjord et al., 2021).
