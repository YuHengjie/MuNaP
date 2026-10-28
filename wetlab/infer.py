from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from common import (
    DATA_DIR,
    SchemeProteinDataset,
    build_model,
    load_embeddings,
    load_qc_dataset,
    load_raw_dataset,
    parse_targets,
    predict_loader,
    scheme_coverage_map,
    set_seed,
    target_key,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer and evaluate a scheme-ranking fine-tuned model"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--candidate-dataset", type=Path)
    parser.add_argument("--dataset-qc", type=Path)
    parser.add_argument(
        "--targets",
        default="NP10:0.6,NP2:0.6,NP4:0.6,NP8:1.0",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--best-epoch", type=int)
    parser.add_argument("--expected-test-schemes", type=int, default=36)
    return parser.parse_args()


def tier_groups(rank: pd.Series) -> pd.Series:
    count = len(rank)
    third = count // 3
    groups = pd.Series("Medium", index=rank.index, dtype=object)
    groups.loc[rank <= third] = "High"
    groups.loc[rank > 2 * third] = "Low"
    return groups


def evaluate_rank_frame(frame: pd.DataFrame, pred_column: str):
    pred_rank = frame[pred_column].rank(ascending=False, method="first")
    pred_group = tier_groups(pred_rank)
    raw_group = frame["raw_group"]
    agreement = int((raw_group == pred_group).sum())
    high_correct = int(
        ((raw_group == "High") & (pred_group == "High")).sum()
    )
    extreme_errors = int(
        (
            ((raw_group == "High") & (pred_group == "Low"))
            | ((raw_group == "Low") & (pred_group == "High"))
        ).sum()
    )
    rho, p_value = spearmanr(frame["raw_rank"], pred_rank)
    return {
        "score": pred_column,
        "tier_agreement": agreement,
        "experimental_high_correct": high_correct,
        "opposite_extreme_errors": extreme_errors,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "pred_rank": pred_rank,
        "pred_group": pred_group,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data_dir = args.data_dir.resolve()
    candidate_dataset_path = (
        args.candidate_dataset
        or input_data_dir / "candidate_raw.csv"
    ).resolve()
    dataset_qc_path = (
        args.dataset_qc or input_data_dir / "dataset_qc.csv"
    ).resolve()
    run_dir = args.run_dir.resolve()
    model_dir = run_dir / "model"
    results_dir = run_dir / "results"
    data_dir = run_dir / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint or (model_dir / "saved_model_ft.pt")
    best_epoch = json.loads(
        (model_dir / "best_epoch.json").read_text(encoding="utf-8")
    )
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(best_epoch["best_threshold"])
    )
    selected_epoch = (
        args.best_epoch
        if args.best_epoch is not None
        else int(best_epoch["best_epoch"])
    )
    targets = parse_targets(args.targets)
    target_keys = {target_key(material, dose) for material, dose in targets}

    candidate_df = load_raw_dataset(candidate_dataset_path)
    qc_df = load_qc_dataset(dataset_qc_path)
    test_df = candidate_df
    test_samples = sorted(test_df["sample"].unique())
    if len(test_samples) != args.expected_test_schemes:
        raise ValueError(
            f"Expected {args.expected_test_schemes} test schemes, "
            f"found {len(test_samples)}"
        )
    overlap = sorted(set(test_samples) & target_keys)
    if overlap:
        raise ValueError(
            "candidate_raw.csv must not contain calibration schemes: "
            f"{overlap}"
        )
    qc_samples = sorted(qc_df["sample"].unique())
    if set(qc_samples) != set(test_samples):
        extra = sorted(set(qc_samples) - set(test_samples))
        missing = sorted(set(test_samples) - set(qc_samples))
        raise ValueError(
            "dataset_qc.csv must contain exactly the candidate schemes; "
            f"extra={extra}, missing={missing}"
        )
    coverage = pd.Series(scheme_coverage_map(qc_df, test_df))

    text_embeddings, sequence_embeddings, structure_embeddings = load_embeddings(
        input_data_dir
    )
    test_dataset = SchemeProteinDataset(
        test_df,
        text_embeddings,
        sequence_embeddings,
        structure_embeddings,
        scheme_to_index={sample: idx for idx, sample in enumerate(test_samples)},
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(
        text_embeddings,
        sequence_embeddings,
        structure_embeddings,
        checkpoint,
        device,
    )
    probabilities, _, _, _ = predict_loader(model, test_loader, device)
    predictions = (probabilities >= threshold).astype(int)

    result_df = test_df[
        ["ID", "dose", "Accession", "RPA", "x_index", "sample"]
    ].rename(columns={"RPA": "raw_RPA"})
    result_df["pred_prob"] = probabilities
    result_df["pred_label"] = predictions
    qc_eval = qc_df[
        ["sample", "Accession", "RPA", "label"]
    ].rename(columns={"RPA": "qc_RPA", "label": "qc_label"})
    result_df = result_df.merge(
        qc_eval,
        on=["sample", "Accession"],
        how="left",
        validate="one_to_one",
    )
    if result_df[["qc_RPA", "qc_label"]].isna().any().any():
        raise ValueError(
            "dataset_qc.csv does not cover every candidate scheme-protein pair"
        )
    qc_labels = result_df["qc_label"].to_numpy(dtype=np.int32)
    result_df.to_csv(results_dir / "infer_test.csv", index=False)

    pair_metrics = {
        "test_AUROC": float(roc_auc_score(qc_labels, probabilities)),
        "test_AUPRC": float(average_precision_score(qc_labels, probabilities)),
        "test_Accuracy": float(accuracy_score(qc_labels, predictions)),
        "test_Precision": float(
            precision_score(qc_labels, predictions, zero_division=0)
        ),
        "test_Recall": float(
            recall_score(qc_labels, predictions, zero_division=0)
        ),
        "test_F1": float(f1_score(qc_labels, predictions, zero_division=0)),
        "test_MCC": float(matthews_corrcoef(qc_labels, predictions)),
        "threshold": threshold,
        "best_epoch": selected_epoch,
        "best_validation_selection_score": float(
            best_epoch["best_selection_score"]
        ),
    }

    scheme_df = result_df.groupby("sample", as_index=False).agg(
        mean_prob=("pred_prob", "mean"),
        median_prob=("pred_prob", "median"),
        fraction_above_threshold=("pred_label", "mean"),
        detected_proteins=("pred_label", "sum"),
        total_proteins=("pred_label", "size"),
        observed_coverage=("qc_label", "mean"),
    )
    for fixed_threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        column = f"fraction_above_{fixed_threshold:.1f}"
        fixed_labels = result_df["pred_prob"] >= fixed_threshold
        scheme_df[column] = (
            fixed_labels.groupby(result_df["sample"]).mean().reindex(
                scheme_df["sample"]
            ).to_numpy()
        )
    scheme_df["experimental_coverage"] = scheme_df["sample"].map(coverage)
    scheme_df["raw_rank"] = scheme_df["experimental_coverage"].rank(
        ascending=False, method="first"
    )
    scheme_df["raw_group"] = tier_groups(scheme_df["raw_rank"])
    scheme_df.to_csv(results_dir / "scheme_scores.csv", index=False)

    score_rows = []
    score_columns = [
        "mean_prob",
        "median_prob",
        "fraction_above_threshold",
        "fraction_above_0.3",
        "fraction_above_0.4",
        "fraction_above_0.5",
        "fraction_above_0.6",
        "fraction_above_0.7",
    ]
    for score_column in score_columns:
        metrics = evaluate_rank_frame(scheme_df, score_column)
        rank_frame = scheme_df[
            [
                "sample",
                "experimental_coverage",
                "raw_rank",
                "raw_group",
                "mean_prob",
                "median_prob",
                "fraction_above_threshold",
                "fraction_above_0.3",
                "fraction_above_0.4",
                "fraction_above_0.5",
                "fraction_above_0.6",
                "fraction_above_0.7",
            ]
        ].copy()
        rank_frame["pred_rank"] = metrics.pop("pred_rank")
        rank_frame["pred_group"] = metrics.pop("pred_group")
        rank_frame = rank_frame.sort_values("pred_rank")
        rank_frame.to_excel(
            results_dir / f"rank_compare_{score_column}.xlsx", index=False
        )
        score_rows.append(
            {
                **pair_metrics,
                "scheme_score": score_column,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(score_rows)
    metrics_df.to_csv(results_dir / "scheme_rank_metrics.csv", index=False)
    metrics_df.to_csv(results_dir / "comparison.csv", index=False)
    (results_dir / "metrics.json").write_text(
        json.dumps(
            {
                "pair_metrics": pair_metrics,
                "scheme_metrics": metrics_df.to_dict(orient="records"),
                "test_schemes": test_samples,
                "candidate_dataset": str(candidate_dataset_path),
                "dataset_qc": str(dataset_qc_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
