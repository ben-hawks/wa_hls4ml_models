# Retraining on post-synthesis resource labels

The paper's surrogate models ([arXiv:2511.05615](https://arxiv.org/abs/2511.05615)) were trained on the HLS resource estimates (`hls_resource_report`). Here both models are retrained with FF, LUT, BRAM and DSP taken from the post-logic-synthesis `resource_report` instead. Cycles and II still come from the HLS `latency_report`, because the dataset has no post-synthesis latency.

## Data

Built from the HuggingFace [wa-hls4ml](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml) `train/`, `val/` and `test/` splits with:

```bash
cd dataset
python Dataset_to_csvs6_with_ii.py --hf-root ../wa-hls4ml \
    --output-dir output/hf_split_resource_report --resource-key resource_report
```

| Split | Models | Array shape |
|---|---|---|
| Train | 433,676 | (433676, 51, 18) |
| Validation | 92,992 | (92992, 51, 18) |
| Test | 92,933 | (92933, 51, 18) |

620,488 of the 684,061 records have post-synthesis numbers; models without them are skipped. Coverage is lower for convolutional models (about 29% of conv2d and 42% of conv1d records), so the data leans heavily toward fully connected models. BRAM keeps half blocks (BRAM18 = 0.5).

## Models

| | Transformer | GNN |
|---|---|---|
| Code | `transformer/run.py` | `GNN/training_scripts/y_03_GAT_vanilla_bigboi.py` |
| Architecture | 2 encoder blocks, 8 heads, 512-dim | `FPGA_GNN_GATv2`: 5 GATv2 layers × 5 heads, hidden 512 |
| Labels | log, then z-score | log, then z-score |
| Optimizer | Adam, lr 1e-5 | AdamW, lr 3e-3, ReduceLROnPlateau |
| Batch size | 512 | 1024 |
| Epochs | 200 (best: epoch 197) | up to 200, early stopping; stopped at 105 (best: epoch 65) |
| Weights | `transformer/model.pt` | release `gnn-weights-resource-report-v1` (see below) |

The paper trained the transformer for 250 epochs at batch size 1024. That wasn't repeated here: validation loss was flat over the last 50 epochs while training loss kept falling, and a larger batch at the same learning rate would make fewer updates.

### GNN weights

The GNN checkpoint (`final_gatv2_model.pth`, 218 MB) is over GitHub's file size limit, so it's attached to the GitHub release `gnn-weights-resource-report-v1` (MD5 `e09b10ec8fb3f8a016ee081711d170e3`). It must be used with `gnn/normalization_stats_log.npy` from this folder:

```bash
cd GNN
python load_pretrained.py final_gatv2_model.pth \
    --data-dir ../dataset/output/hf_split_resource_report \
    --stats ../resource_report_results/gnn/normalization_stats_log.npy
```

The transformer recomputes its normalization statistics from the training split, so it needs no stats file.

## Results

Test set, both models evaluated with the same metric code. Predictions are capped at the largest training label (see below). "Rounded" SMAPE rounds BRAM to half blocks and DSP to whole DSPs. The paper's numbers are for the HLS-estimate labels, so they're a reference point, not a like-for-like target.

| R² | BRAM | DSP | FF | LUT | Cycles | II |
|---|---|---|---|---|---|---|
| GNN, paper | 0.51 | 0.89 | 0.74 | 0.73 | 0.89 | 0.91 |
| GNN, post-synthesis | 0.64 | 0.56 | 0.93 | 0.90 | 0.81 | 0.83 |
| Transformer, paper | 0.39 | 0.29 | 0.72 | 0.67 | 0.95 | 0.95 |
| Transformer, post-synthesis | 0.35 | 0.83 | 0.93 | 0.90 | 0.93 | 0.92 |

| SMAPE % (rounded) | BRAM | DSP | FF | LUT | Cycles | II |
|---|---|---|---|---|---|---|
| GNN, paper | 19.5 | 15.1 | 11.6 | 11.4 | 15.7 | 13.4 |
| GNN, post-synthesis | 24.8 | 15.8 | 14.1 | 14.0 | 17.9 | 14.7 |
| Transformer, paper | 14.1 | 10.8 | 2.9 | 2.9 | 10.1 | 14.1 |
| Transformer, post-synthesis | 21.0 | 8.7 | 3.6 | 4.5 | 11.3 | 15.0 |

[`comparison_to_paper.md`](comparison_to_paper.md) has the Dense, Conv1D and Conv2D subsets, raw and rounded SMAPE, and results restricted to the xcu250 models.

## Findings

- **FF and LUT** are predicted much better from post-synthesis labels than in the paper (R² 0.90–0.93 against 0.67–0.74).
- **Prediction cap.** Without it, the transformer predicts up to 1.98 million DSPs for a few models, which is impossible on the target FPGA, and DSP R² falls to −117.6. Capping at the largest training label touches 124 of about 558,000 transformer predictions (9 for the GNN) and gives DSP R² 0.83. The cap is now part of `transformer/run.py`, the GNN training script and `GNN/load_pretrained.py`.
- **GNN DSP** stays well below the paper (R² 0.56 against 0.89). The worst errors are large designs (about 12,000 DSPs) predicted at 1,000–2,000. After synthesis, about a third of conv models use no DSPs, which the HLS estimates never show.
- **BRAM and the `2_20` subset.** The `2_20` models were synthesized for three different FPGAs (xcu200, xc7z020, xczu9eg) with Vivado 2019.1, and the target part isn't an input feature. They're 1.5% of the test set but cause 49% (GNN) and 77% (transformer) of the BRAM squared error. On the xcu250 models alone, transformer BRAM R² is 0.71. Adding the target part as a feature, or restricting to xcu250, would address it.
- **Half-block BRAM** isn't predictable from the features: the models get the .5 right about half the time. Even perfect half-block prediction would lower BRAM SMAPE by only about 3 points.
- **Zero labels.** DSP is 0 for 38% of models. With the log transform's epsilon of 1e-6 these sit at log(1e-6) ≈ −13.8, far from all non-zero values. SMAPE counts any tiny non-zero prediction as 200% error, which is why DSP SMAPE is reported rounded.

## Contents

- `transformer/`: best weights (`model.pt`), training log, evaluation log with the cap, plots, test predictions.
- `gnn/`: metrics, training log, normalization stats, plots, test predictions.
- `eval/`: `predict_transformer.py` and `predict_gnn.py` write `test_predictions.npz`; `compare_to_paper.py` prints the comparison tables. They expect the converted arrays in `dataset/output/hf_split_resource_report/` and the HuggingFace download in `wa-hls4ml/`.
