#!/usr/bin/env python3
"""RQ1: accepted vs rejected numeric-feature comparison.

Uses Mann-Whitney U (non-parametric) with Cliff's delta (non-parametric
effect size). Bonferroni-corrects p-values for the number of features tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Defaults resolve relative to this package layout:
#   RQ1/analysis/rq1_feature_comparison.py
#   RQ1/data/<enriched parquet>
#   RQ1/results/
_RQ1_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = _RQ1_ROOT / "data" / "enriched_dataset4(only body edited data).parquet"
OUT_DIR = _RQ1_ROOT / "results"


def cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    """Cliff's delta from Mann-Whitney U of group 1 vs group 2.

    δ = 2U / (n1 n2) - 1
    Positive δ means group 1 (accepted) tends to have larger values.
    """
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def cliffs_delta_label(d: float) -> str:
    """Romano et al. (2006) magnitude labels, mapped from Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    if abs_d < 0.330:
        return "small"
    if abs_d < 0.474:
        return "medium"
    return "large"


def compare_features(accepted: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = accepted.select_dtypes(include=[np.number]).columns
    results = []

    for col in numeric_cols:
        accepted_vals = accepted[col].dropna().to_numpy()
        rejected_vals = rejected[col].dropna().to_numpy()
        if len(accepted_vals) == 0 or len(rejected_vals) == 0:
            continue

        u_stat, p_val = mannwhitneyu(
            accepted_vals, rejected_vals, alternative="two-sided", method="asymptotic"
        )
        n1, n2 = len(accepted_vals), len(rejected_vals)
        cliffs_d = cliffs_delta_from_u(float(u_stat), n1, n2)

        results.append(
            {
                "feature": col,
                "n_accepted": n1,
                "n_rejected": n2,
                "mannwhitneyu_U": float(u_stat),
                "mannwhitneyu_p": float(p_val),
                "cliffs_delta": cliffs_d,
                "accepted_mean": float(np.mean(accepted_vals)),
                "rejected_mean": float(np.mean(rejected_vals)),
                "accepted_median": float(np.median(accepted_vals)),
                "rejected_median": float(np.median(rejected_vals)),
            }
        )

    results_df = pd.DataFrame(results)
    n_tests = len(results_df)
    results_df["n_tests"] = n_tests
    results_df["mannwhitneyu_p_bonferroni"] = np.minimum(
        results_df["mannwhitneyu_p"] * n_tests, 1.0
    )
    results_df["significant_raw_0.05"] = results_df["mannwhitneyu_p"] < 0.05
    results_df["significant_bonferroni_0.05"] = (
        results_df["mannwhitneyu_p_bonferroni"] < 0.05
    )
    results_df["cliffs_delta_magnitude"] = results_df["cliffs_delta"].apply(
        cliffs_delta_label
    )
    results_df["cliffs_delta_sig"] = results_df.apply(
        lambda r: f"{r['cliffs_delta']:.3f} ({r['cliffs_delta_magnitude'][0].upper()})",
        axis=1,
    )
    results_df.sort_values(
        ["mannwhitneyu_p_bonferroni", "feature"], inplace=True
    )
    results_df.reset_index(drop=True, inplace=True)
    return results_df


def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    accepted = df[df["Label"] == "Accepted"]
    rejected = df[df["Label"] == "Rejected"]
    print(f"Accepted: {len(accepted):,}  Rejected: {len(rejected):,}")

    results_df = compare_features(accepted, rejected)
    n_tests = int(results_df["n_tests"].iloc[0]) if len(results_df) else 0
    n_raw = int(results_df["significant_raw_0.05"].sum())
    n_bonf = int(results_df["significant_bonferroni_0.05"].sum())
    print(f"Features tested: {n_tests}")
    print(f"Significant at α=0.05 (raw): {n_raw}")
    print(f"Significant at α=0.05 (Bonferroni): {n_bonf}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "rq1_feature_comparison_cliffs_delta.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    display_cols = [
        "feature",
        "mannwhitneyu_p",
        "mannwhitneyu_p_bonferroni",
        "cliffs_delta_sig",
        "accepted_mean",
        "rejected_mean",
        "significant_bonferroni_0.05",
    ]
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
