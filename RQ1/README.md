# RQ1 — Are accepted and rejected suggested edits different?

## Goal

Compare accepted vs rejected suggested edits across 35 numeric features using the Mann–Whitney U test (Bonferroni-corrected) and Cliff’s δ effect size.

## Layout

| Path | Contents |
|------|----------|
| `data/` | Enriched corpus parquet (local / symlink; not in Git) |
| `analysis/rq1_feature_comparison.py` | Main statistical comparison script |
| `analysis/selection_scaling.py` | Feature column list used by models |
| `results/accepted_stats_rq1.xlsx` | `describe()` stats for accepted edits |
| `results/rejected_stats_rq1.xlsx` | `describe()` stats for rejected edits |
| `results/rq1_feature_comparison_cliffs_delta.csv` | Per-feature U, p, Bonferroni p, Cliff’s δ |

## Reproduce

1. Place / link `enriched_dataset4(only body edited data).parquet` under `data/`.
2. Update `DATA_PATH` / `OUT_DIR` in `analysis/rq1_feature_comparison.py` if needed (defaults point at the study paths).
3. Run:

```bash
python analysis/rq1_feature_comparison.py
```

## Notes on `NumTags`

`NumTags` is the count of tags on the **pre-edit** post (`PreEditTags`), not the number of tags in the proposed `EditedTags` field. Values typically range from 0–5 (Stack Overflow tag limit), with a handful of rare 6-tag rows.
