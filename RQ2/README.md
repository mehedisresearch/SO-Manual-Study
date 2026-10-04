# RQ2 — What are the reasons for rejecting edits?

## Goal

Manual coding of **690** randomly sampled rejected edits (99% confidence, 5% MoE) yielded **17** fine-grained rejection reasons (Cohen’s κ = 0.80 between two independent coders).

## Layout

| Path | Contents |
|------|----------|
| `data/` | Placeholder for the labeled coding sheet (see below) |
| `analysis/` | Coding protocol notes |
| `results/taxonomy.csv` | The 17-reason taxonomy with counts from the paper |
| `results/reason_examples/` | Screenshot examples for each rejection reason |

## Taxonomy (paper Table 3)

See `results/taxonomy.csv`. High-level findings:

- **Minor Format Only Change** is the most frequent reason (**36%**).
- ~**19.9%** misuse the edit queue to communicate (comment / question / alternate suggestion / status update).
- ~**25.5%** introduce incorrect or harmful changes.

## Reproduce the qualitative study

1. Sample 690 rejected edits from the corpus (exclude suggested-edit conflicts).
2. Open-code ~100 edits to build the guideline.
3. Two annotators independently label all 690; resolve disagreements (third annotator if needed).
4. Report frequencies and κ.

Mapped SO→taxonomy labels are in `data/mapped_rejection_reasons.*`. The anonymized per-edit 690-row coding sheet (if released separately) should also live under `data/`.
