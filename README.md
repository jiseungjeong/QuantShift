# QuantShift

**How weight quantization reshapes the per-layer acceptance landscape of self-speculative decoding.**

QuantShift measures whether LayerSkip self-speculative decoding (self-SD) keeps its
*acceptance* — the draft hit-rate that sets its speed — after 4/8-bit weight quantization,
and where in the network quantization changes predictions. The headline findings:

1. **Acceptance is preserved.** The per-layer acceptance erosion is `Δα(E) ≈ 0` at every exit
   layer on both tasks (`max|Δα| ≤ 0.012`), clearing criteria fixed *before* measuring and
   reproduced on a second model.
2. **Most tokens do not move.** A per-token breakdown shows quantization leaves ~97% of
   predictions unchanged; the few that change cancel (loss ≈ gain).
3. **Speed is conditional.** Self-SD beats autoregressive (AR) decoding only when the AR
   baseline is slow. On NF4 the AR baseline is already fast, so no exit depth lets self-SD win;
   the bottleneck is the dequantization kernel, not the exit policy.

## Setup

This project drives the [LayerSkip](https://github.com/facebookresearch/LayerSkip) codebase
(vendored under `LayerSkip/`, with a few patched files tracked in this repo) on
`facebook/layerskip-llama2-7B`.

```bash
# 1. Environment (Python 3.10+, CUDA GPU for the measurement scripts)
conda create -n NLP python=3.10 && conda activate NLP
pip install torch transformers bitsandbytes datasets numpy matplotlib

# 2. Download the model
python download_layer_skip.py            # saves to ./models/layerskip-llama2-7B

# 3. Point the scripts at the model (optional; defaults to ~/models/layerskip-llama2-7B)
export QUANTSKIP_MODEL=$PWD/models/layerskip-llama2-7B
export QUANTSKIP_MODEL_1B=$PWD/models/layerskip-llama3.2-1B   # only for gen_1b.py
```

Scripts that load the model import from the `LayerSkip/` tree, so run them with that on the
path, e.g.:

```bash
cd LayerSkip && PYTHONPATH=$PWD python ../shadow_landscape.py --precision fp16 --num 30
```

CPU-only analysis and figure scripts read the saved artifacts and can run from the repo root.

## Pipeline

The scripts form a pipeline: **measurement** produces raw artifacts under `logs/`, **analysis**
turns them into metrics, and **figure** scripts render the report PDFs. The central artifact is
`logs/shadow/preds_{fp16,int8,nf4}.pt` from `shadow_landscape.py`, which most analyses consume.

### Measurement (GPU; writes `logs/`)

| Script | What it produces |
| --- | --- |
| `shadow_landscape.py` | Per-layer shadow-token predictions `preds_*.pt` + cross-validation gate |
| `baseline_eval.py` | AR vs self-SD throughput / quality / VRAM per precision (`logs/baselines/`) |
| `gen_1b.py` | Same shadow collection on Llama-3.2-1B (reproduction check) |
| `batch_roofline.py` | One-forward time vs batch size (memory-bound vs compute-bound) |
| `block_budget.py` | Per-block draft+verify budget vs AR |
| `dequant_cost_compare.py` | Per-step dequant cost, NF4 vs INT8 |
| `int8_kernel_probe.py` | INT8 decode slowdown vs outlier-split threshold |
| `nf4_E_sweep.py` | Throughput sweep over the fixed exit layer for NF4 |
| `rE_diagnostic.py` | `r(E) = c_draft(E)/c_verify` feasibility probe |
| `pilot_sample_size.py` | Sample-size pilot for the acceptance CI |
| `aed_probe.py`, `aed_measured.py`, `mag_probe.py` | Depth/gamma-adaptive drafting probes |

### Analysis (CPU; reads `preds_*.pt`, writes `logs/report/`)

| Script | What it computes |
| --- | --- |
| `analyze_landscape.py` | `Δα(E)`, `D(E)`, pre-registration gate |
| `analyze_tokencat.py` | `D(E)` by token category + margin↔`D(E)` correlation |
| `cancel_mechanism.py` | Why `Δα ≈ 0`: kept-accept rate and flip-to-match gain |
| `transition_matrix.py`, `transition_by_layer.py` | 2×2 acceptance-event transition (loss ≈ gain) |
| `content_change_by_layer.py` | Per-layer share of changed predicted tokens |
| `coflip_test.py`, `corr_drift.py` | Shared-trajectory diagnostics |
| `aggregate_baselines.py`, `breakeven.py` | Throughput table and break-even analysis |

### Figures (CPU; writes `figures/*.pdf`)

| Script | Figures |
| --- | --- |
| `figures_for_report.py` | Gate, acceptance curves, `Δα`, divergence |
| `make_heatmap_pdf.py` | Token-category × exit-layer `D(E)` heatmap |
| `make_roofline_pdf.py` | Batch-roofline figure |
| `report_figs.py`, `report_extra.py` | Precision-trend, break-even, extra artifacts |

## Notes

- `logs/`, model weights, and dataset caches are gitignored; regenerate them with the scripts above.
- The cross-precision wall-clock comparison is kernel-dependent (bitsandbytes INT8 decode is
  unusually slow); the per-precision and acceptance results are kernel-independent. See the
  Limitations section of the report.
