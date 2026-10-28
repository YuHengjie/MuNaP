from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average per-scheme rank percentiles across model runs"
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def assign_tiers(rank: pd.Series) -> pd.Series:
    third = len(rank) // 3
    group = pd.Series("Medium", index=rank.index, dtype=object)
    group.loc[rank <= third] = "High"
    group.loc[rank > 2 * third] = "Low"
    return group


def main():
    args = parse_args()
    runs = [Path(value).resolve() for value in args.runs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    rank_columns = []
    for run_dir in runs:
        frame = pd.read_csv(
            run_dir / "results/scheme_scores.csv"
        ).set_index("sample")
        frames.append(frame)
        rank_columns.append(
            frame["mean_prob"].rename(run_dir.name).rank(
                pct=True, method="average"
            )
        )

    reference = frames[0]
    for frame in frames[1:]:
        if set(frame.index) != set(reference.index):
            raise ValueError("All runs must contain the same 36 schemes")
        if not frame.loc[reference.index, "raw_rank"].equals(
            reference["raw_rank"]
        ):
            raise ValueError("Experimental ranks differ between runs")

    rank_frame = pd.concat(rank_columns, axis=1)
    result = rank_frame.copy()
    result["ensemble_rank_score"] = rank_frame.mean(axis=1)
    result["ensemble_rank"] = result["ensemble_rank_score"].rank(
        ascending=False, method="first"
    )
    result["experimental_coverage"] = reference["experimental_coverage"]
    result["raw_rank"] = reference["raw_rank"]
    result["raw_group"] = reference["raw_group"]
    result["pred_group"] = assign_tiers(result["ensemble_rank"])
    result = result.sort_values("ensemble_rank")

    rho, p_value = spearmanr(result["raw_rank"], result["ensemble_rank"])
    metrics = {
        "aggregation": "mean of per-run scheme rank percentiles",
        "num_models": len(runs),
        "tier_agreement": int(
            (result["raw_group"] == result["pred_group"]).sum()
        ),
        "experimental_high_correct": int(
            (
                (result["raw_group"] == "High")
                & (result["pred_group"] == "High")
            ).sum()
        ),
        "opposite_extreme_errors": int(
            (
                ((result["raw_group"] == "High") & (result["pred_group"] == "Low"))
                | ((result["raw_group"] == "Low") & (result["pred_group"] == "High"))
            ).sum()
        ),
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
    }

    result.reset_index().to_csv(output_dir / "scheme_scores.csv", index=False)
    result.reset_index().to_excel(
        output_dir / "rank_compare.xlsx", index=False
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "runs": [str(run_dir) for run_dir in runs],
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
