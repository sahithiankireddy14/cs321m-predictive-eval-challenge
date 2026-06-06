# Predictive AI Evaluation: Lookup Tables + k-NN Item Difficulty

## Submission Overview

This submission combines a condition-aware lookup table with k-nearest-neighbor item difficulty estimation. The approach achieves a negative log-loss of **-0.59** and AUROC of **0.71** on the private test set.

### Three-Stage Pipeline

**Stage 1: Subject Ability (Lookup)**
- For known subjects, retrieve historical benchmark accuracy from a lookup table
- Fallback chain: subject+benchmark+condition → subject+benchmark → subject → family → benchmark+condition → global mean
- Apply Platt scaling calibration trained on adaptive labels

**Stage 2a: Item Difficulty (k-NN)**
- Encode test item with all-MiniLM-L6-v2 sentence embeddings (384-d, L2-normalized)
- Retrieve 10 most similar training items by cosine similarity
- Compute weighted average of their observed pass rate residuals (deviation from benchmark mean)
- Adjust base prediction: `p = sigmoid(logit(p_lookup) + 0.3 * item_residual)`

**Stage 3: Calibration (Platt Scaling)**
- Per-benchmark Platt scaling (a, b parameters) fitted on revealed labels from adaptive labeling
- Falls back to global calibration if insufficient per-benchmark labels

## Files

### Code
- **model.py** — Main submission entry point; implements predict(input, labeled)
- **labeling.py** — Acquisition function for adaptive labeling; selects high-uncertainty items for label revelation

### Data
- **means_exp10.json** — Lookup table with subject means, benchmark means, condition deltas, family fallbacks, and per-benchmark Platt parameters
- **item_knn_data.npz** — Compressed numpy archive containing:
  - `embeddings` (float16, shape 44283 × 384) — pre-encoded training items with all-MiniLM-L6-v2
  - `residuals` (float32) — observed pass rate residuals per item
  - `bench_ids` (str) — benchmark ID for each item (used for same-benchmark boosting)
- **item_knn_meta.json** — Benchmark-level statistics (per-benchmark mean pass rates)

### Dependencies
- **models.txt** — Declares `sentence-transformers/all-MiniLM-L6-v2` for platform pre-fetching

## Running Locally

### Setup
```bash
pip install sentence-transformers numpy
```

### Acquisition Function
```python
from labeling import acquisition_function

# Score an item for adaptive labeling
score = acquisition_function(test_input)
print(f"Acquisition score: {score:.4f}")
```

## Model Loading

Models are loaded at module scope (top level of model.py), not inside predict(). This ensures:
- Sentence transformer loads once when the container starts
- All subsequent predict() calls reuse the already-loaded encoder
- Lookup tables and k-NN data are loaded once on startup

## Data Preprocessing

The lookup table and k-NN data were constructed offline:
- **exp1_subject_mean.py** — Computes per-subject, per-benchmark, and per-condition pass rates
- **exp17_item_knn.py** — Encodes 44k training items and computes residuals

Both experiments read from the public HuggingFace dataset (`aims-foundations/measurement-db`) and preprocess locally before submission.

## Key Design Choices

1. **Lookup over Neural Models** — Simple empirical lookup outperforms all neural approaches (NCF, IRT, LLM-as-judge) because subjects are fully observed in training.

2. **k-NN over Learned Prediction** — Direct k-NN matching of item embeddings succeeds where neural regression fails, likely because observed item difficulties have stronger signal than learned predictions on sparse per-item observations.

3. **Per-Benchmark Calibration** — Platt scaling is fit per benchmark using adaptive labels, with regularization proportional to the number of labels received. This provides significant improvement (+0.07 from base lookup).

4. **Condition-Aware Lookup** — Treating zero-shot and chain-of-thought as distinct conditions adds +0.05 improvement, confirming that prompting strategy matters.

5. **Adaptive Labeling** — The acquisition function prioritizes items with higher fallback uncertainty (items grounded only in global mean) for label revelation, focusing limited labeling budget on highest-leverage examples.

## Reproduction

To reproduce the scores reported in the technical report/competition submission:
1. Download the public training data from HuggingFace (`aims-foundations/measurement-db`)
2. Run exp1_subject_mean.py to generate means_exp10.json
3. Run exp17_item_knn.py to generate item_knn_data.npz and item_knn_meta.json
4. ZIP model.py, labeling.py, models.txt, and the three data files
5. Submit to Codabench



