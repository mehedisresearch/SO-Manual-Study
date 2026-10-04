# Understanding Suggested Edits in Stack Overflow

This package is organized by research question. Each RQ folder contains its own `data/`, `analysis/`, and `results/` materials.

```
replication_package/
├── README.md                 # this file
├── requirements.txt
├── RQ1/                      # accepted vs rejected feature differences
│   ├── data/
│   ├── analysis/
│   └── results/
├── RQ2/                      # rejection-reason taxonomy (manual study)
│   ├── data/
│   ├── analysis/
│   └── results/
└── RQ3/                      # review prioritization models
    ├── data/
    ├── analysis/
    └── results/
```

## Research questions

| RQ | Question | What is in this package |
|----|----------|-------------------------|
| **RQ1** | Are accepted and rejected suggested edits different? | Feature comparison script (Mann–Whitney U + Cliff’s δ), accepted/rejected describe stats, comparison CSV |
| **RQ2** | What are the reasons for rejecting edits? | 17-reason taxonomy with counts, example screenshots for each reason |
| **RQ3** | Which edits should we review? | Feature / EditEx / Embedding / LLM baseline scripts and result artifacts (metrics, models, threshold sweeps) |

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install openpyxl scipy pyarrow   # needed for RQ1 Excel I/O and stats
```

Python 3.10+ is recommended.

## Large datasets (not shipped in Git)

GitHub cannot host the full multi-GB study corpus. Place the following files locally before re-running analyses:

| File | Approx. size | Used by | Local path in this package |
|------|-------------|---------|----------------------------|
| Enriched suggested-edit corpus | ~4 GB | RQ1, RQ3 training | `RQ1/data/enriched_dataset4(only body edited data).parquet` |
| Train / val / test splits | ~500 MB | RQ3 | `RQ3/data/{train_bert,val,test}.parquet` |

On the authors’ workstation these paths may already be present as symlinks to the project `data/` directory. For external replication, download the Stack Overflow May 2023 dump and SEDE tables (`SuggestedEdits`, `ReviewTasks`, `ReviewTaskResults`, `ReviewRejectionReasons`) and regenerate features with the preprocessing scripts under `RQ3/analysis/src/Preprocessing/`.

## Quick start per RQ

```bash
# RQ1 – regenerate feature comparison (requires the enriched parquet)
cd RQ1/analysis
python rq1_feature_comparison.py

# RQ2 – browse taxonomy + examples (no model training)
# see RQ2/README.md and RQ2/results/

# RQ3 – Feature Model (XGBoost, 35 features)
cd RQ3/analysis
python "main5(xgb-metadata).py"
```

Detailed instructions live in each RQ’s `README.md`.

## Citation

If you use this package, please cite the paper (TOSEM 2026). Study materials corresponding to the manuscript artifact note are maintained in this repository.

## License

Research code and derived analysis artifacts in this package are provided for academic replication. Stack Overflow content remains subject to Stack Overflow’s content license (CC BY-SA).
