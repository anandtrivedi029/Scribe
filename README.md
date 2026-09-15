# SCRIBE: Sequential Compositional Reasoning with Iterative Belief Encoding

**Author:** Anand Trivedi
**Affiliation:** Independent Researcher
**Repository:** https://github.com/anandtrivedi029/Scribe

SCRIBE is an experimental lightweight neural architecture for multi-hop reasoning. It processes context sequentially and maintains a persistent memory through attention-based, confidence-weighted write operations.

The central design constraint is causal: when a sentence is processed, it can use only information already written to memory by preceding sentences. The aim is to study whether this restriction can reduce shortcut-style reasoning that may arise when a model can attend to an entire context simultaneously.

> **Research status:** This repository accompanies an independent research manuscript and contains experimental code/notebooks, benchmark data, and reported evaluation results. The work should be treated as research in progress and has not been independently replicated.

---

## Main Result

The final SCRIBE configuration used in the reported comparison has approximately **1.1 million trainable parameters**.

| Model              |   Params |   Overall |      bAbI | ProofWriter |      Easy |    Medium |      Hard |
| ------------------ | -------: | --------: | --------: | ----------: | --------: | --------: | --------: |
| MLP Baseline       |     0.2M |     70.2% |     48.5% |       91.8% |     72.4% |     56.6% |     73.4% |
| LSTM Baseline      |     0.8M |     75.5% |     59.1% |       91.8% |     75.9% |     66.0% |     77.9% |
| Flat Attention     |     0.7M |     79.0% |     64.4% |       93.6% |     79.0% |     73.1% |     80.5% |
| **SCRIBE (final)** | **1.1M** | **85.8%** | **78.0%** |   **93.7%** | **82.1%** | **85.2%** | **86.5%** |

These are the results reported by the experiments accompanying this repository and manuscript.

---

## Observed Performance by Reasoning Depth

For the reported evaluation, examples were grouped by context/reasoning depth:

* **Easy (1–3 sentences):** 82.1%
* **Medium (4–6 sentences):** 85.2%
* **Hard (7+ sentences):** 86.5%

Within this benchmark setup, SCRIBE did not show the usual decline in accuracy as input depth increased. Instead, the reported accuracy increased across these depth groups.

This pattern is **consistent with improved use of sequentially accumulated information**, but it should not by itself be interpreted as proof of general compositional reasoning. Further evaluation on additional datasets, controlled counterfactual tests, and independent replication would be needed to establish that claim more broadly.

---

## Architecture

SCRIBE is built around four main ideas:

1. **Sequential context processing**
   Sentences are processed one at a time rather than exposing the complete context to the reasoning module at once.

2. **Persistent memory**
   Information extracted from earlier sentences is retained in a memory state that later sentences can access.

3. **Confidence-weighted writes**
   Memory updates are weighted by confidence, allowing uncertain information to have a weaker effect than high-confidence information.

4. **Multi-hop reading**
   The final prediction is produced using repeated attention over the resulting memory state.

The architecture also uses a learned **sentinel/null memory slot** intended to absorb information that does not need to be stored as a task-relevant belief.

---

## Architecture Evolution and Ablations

The earliest tested SCRIBE configuration included a periodic memory-consolidation mechanism and contained approximately **1.4M parameters**.

Ablation experiments showed that removing consolidation improved performance substantially. The no-consolidation configuration therefore became the **final SCRIBE architecture** reported in the main comparison.

| Configuration                          |   Params |   Overall |      bAbI |      Hard |
| -------------------------------------- | -------: | --------: | --------: | --------: |
| Initial SCRIBE (all tested components) |     1.4M |     80.3% |     70.5% |     80.6% |
| − Revision                             |     1.3M |     80.1% |     69.8% |     80.5% |
| − Sentinel                             |     1.4M |     80.0% |     70.2% |     80.2% |
| − Confidence weighting                 |     1.4M |     77.9% |     65.6% |     78.9% |
| **− Consolidation (final SCRIBE)**     | **1.1M** | **85.8%** | **78.0%** | **86.5%** |

The ablation results suggest two useful observations within the tested setup:

* confidence-weighted memory updates contribute materially to performance;
* periodic consolidation was not beneficial in the tested configuration.

The final reported model therefore uses the simpler no-consolidation design.

---

## Repository Contents

The repository currently contains:

```text
Scribe/
├── README.md
├── LICENSE
├── SCRIBE_Paper_v2.pdf
├── ScribeNetColab3.ipynb
├── Scribe_benchmarking.ipynb
└── unified_reasoning_dataset.parquet
```

### Files

**`SCRIBE_Paper_v2.pdf`**
Research manuscript describing the motivation, architecture, experiments, and results.

**`ScribeNetColab3.ipynb`**
Development/training notebook for the SCRIBE model and associated experiments.

**`Scribe_benchmarking.ipynb`**
Notebook used for benchmark comparisons and evaluation.

**`unified_reasoning_dataset.parquet`**
Prepared dataset used by the experiments, combining examples derived from the reasoning benchmarks used in this work.

---

## Benchmarks

The experiments use two established reasoning benchmarks:

### bAbI

The bAbI tasks were introduced by Weston et al. to evaluate forms of synthetic text understanding and multi-step reasoning.

Reference:

> Weston, J., Bordes, A., Chopra, S., Rush, A. M., van Merriënboer, B., Joulin, A., & Mikolov, T.
> *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks.*
> arXiv:1502.05698.

### ProofWriter

ProofWriter evaluates logical reasoning over natural-language rule sets and facts.

Reference:

> Tafjord, O., Dalvi, B., & Clark, P.
> *ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language.*
> Findings of ACL-IJCNLP, 2021.

---

## Running the Experiments

The repository is currently notebook-based.

### 1. Clone the repository

```bash
git clone https://github.com/anandtrivedi029/Scribe.git
cd Scribe
```

### 2. Create a Python environment

For example:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the main dependencies

The notebooks use the Python/PyTorch scientific-computing stack. A minimal starting environment is:

```bash
pip install torch pandas sentence-transformers tqdm matplotlib jupyter pyarrow
```

Depending on the notebook environment, additional packages may be required by individual experimental cells.

### 4. Start Jupyter

```bash
jupyter notebook
```

Then open:

* `ScribeNetColab3.ipynb` for SCRIBE development/training experiments;
* `Scribe_benchmarking.ipynb` for benchmark evaluation.

The included `unified_reasoning_dataset.parquet` should remain in the repository root unless the notebook path is changed accordingly.

---

## Reproducibility Notes

The repository contains the notebooks and dataset used for the reported experimental workflow.

For rigorous reproduction, researchers should record and report:

* Python version;
* PyTorch version;
* GPU/CPU environment;
* random seed;
* train/validation/test split;
* number of training epochs;
* optimization parameters;
* model-selection criterion.

Where a result depends on stochastic training, replication across multiple random seeds is recommended.

The current reported values should therefore be interpreted as the results of the experiments described in the accompanying manuscript and notebooks, rather than as independently replicated benchmark estimates.

---

## Scope and Limitations

SCRIBE is a compact experimental architecture designed to study sequential memory and multi-hop reasoning. The current evidence is limited to the benchmark setup reported here.

Important limitations include:

* evaluation on a limited number of reasoning benchmarks;
* synthetic or controlled reasoning tasks may not reflect open-domain reasoning;
* increasing accuracy with depth does not by itself establish causal or human-like reasoning;
* benchmark-specific artifacts may contribute to measured performance;
* broader comparison with modern transformer and state-space baselines is still desirable;
* independent replication has not yet been reported.

Future work should evaluate SCRIBE on additional compositional-reasoning datasets, adversarial and counterfactual tests, out-of-distribution reasoning tasks, and stronger parameter-matched baselines.

---

## Research Motivation

Many reasoning models can access all contextual statements simultaneously. This is powerful, but it can also make it difficult to determine whether a model is actually constructing intermediate beliefs or exploiting correlations available in the complete input.

SCRIBE explores a different constraint:

> **What happens if information must enter the reasoning state sequentially and later reasoning can depend only on what has already been encoded?**

The project is intended as an experimental investigation of that question rather than a claim that sequential memory is universally superior to attention-based architectures.

---

## Manuscript

The current manuscript is included in this repository:

```text
SCRIBE_Paper_v2.pdf
```

No arXiv identifier or DOI is listed here until one has been formally assigned.

---

## Citation

Until a DOI or archival preprint identifier is available, this repository can be cited as:

```bibtex
@misc{trivedi2026scribe,
  title        = {SCRIBE: Sequential Compositional Reasoning with Iterative Belief Encoding},
  author       = {Trivedi, Anand},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/anandtrivedi029/Scribe}
}
```

Once an archival DOI or preprint identifier is assigned, this citation should be updated to reference that record.

---

## License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgments

This work uses PyTorch and Sentence-Transformers and evaluates on datasets derived from the bAbI and ProofWriter reasoning benchmarks.

---

## Contact

**Anand Trivedi**
Independent Researcher
GitHub: https://github.com/anandtrivedi029
