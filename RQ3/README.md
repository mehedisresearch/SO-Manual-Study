# RQ3 — Which edits should we review?

## Goal

Rank suggested edits by predicted rejection probability and evaluate the trade-off between **review rate** (fraction sent to humans) and **rejected-edit recall** across thresholds τ.

Compared approaches:

1. **Feature Model** — XGBoost on the 35 RQ1 features (best overall trade-off in the paper)
2. **Embedding Model** — Qwen embeddings + MLP
3. **EditEx** — prior edit-quality baseline / RF features
4. **Zero-shot LLMs** — GPT-4o, Qwen2.5-32B, Llama-3.3-70B on a balanced 500-edit subset

## Layout

| Path | Contents |
|------|----------|
| `data/` | Train/val/test parquet (local/symlink), EditEx feature parquets, LLM sample |
| `analysis/` | Training / evaluation entrypoints (`main*.py`, `moderation_llm_analysis.py`, `src/`) |
| `results/feature_model/` | XGBoost 35-feature model, scaler, metrics, importance |
| `results/editex/` | EditEx / RF probabilities and metrics |
| `results/embedding_model/` | Wide-MLP metrics / figures |
| `results/llm_baseline_500/` | Balanced-500 LLM vs Feature Model comparison |

## Reproduce (Feature Model)

```bash
# requires RQ3/data/train_bert.parquet and test.parquet
cd analysis
python "main5(xgb-metadata).py"
```

Precomputed metrics live in `results/feature_model/xgb_metadata_35feat_metrics.json`.

## Threshold policy

For a model probability \(p(\text{reject})\), send an edit to human review if \(p \ge \tau\). Paper highlights:

| τ | Review rate | Rejected recall (Feature Model) |
|---|-------------|-------------------------------|
| 0.40 | 52.3% | 80.8% |
| 0.80 | 14.6% | 41.7% |

See `results/llm_baseline_500/xgb_35feat_threshold_sweep.json` for the balanced-500 calibration curves.
