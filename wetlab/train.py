from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from common import (
    BalancedSchemeBatchSampler,
    DATA_DIR,
    SchemeProteinDataset,
    build_model,
    load_embeddings,
    load_raw_dataset,
    parse_targets,
    prepare_head_training,
    predict_loader,
    raw_scheme_coverage,
    set_seed,
    target_key,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-only fine-tuning with scheme coverage and ranking losses"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--calibration-dataset", type=Path)
    parser.add_argument(
        "--targets",
        default="NP10:0.6,NP2:0.6,NP4:0.6,NP8:1.0",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["bce", "coverage", "coverage_rank"],
        default="coverage_rank",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Seed for data splitting; defaults to --seed",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Use stratified k-fold splitting when greater than one",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Validation fold index for k-fold splitting",
    )
    parser.add_argument("--batch-per-scheme", type=int, default=512)
    parser.add_argument("--batches-per-epoch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--coverage-weight", type=float, default=0.3)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--rank-temperature", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=60)
    parser.add_argument(
        "--fixed-epochs",
        action="store_true",
        help="Disable best-checkpoint selection and use the final epoch",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="Pretrained MuNaP checkpoint; defaults to data/pretrained_model.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scheme_coverage_loss(probabilities, schemes, targets, num_schemes):
    sums = torch.zeros(num_schemes, device=probabilities.device)
    counts = torch.zeros(num_schemes, device=probabilities.device)
    sums.index_add_(0, schemes, probabilities)
    counts.index_add_(
        0, schemes, torch.ones_like(probabilities, dtype=probabilities.dtype)
    )
    present = counts > 0
    predicted = sums[present] / counts[present]
    expected = targets.to(probabilities.device)[present]
    loss = F.smooth_l1_loss(predicted, expected, reduction="mean")
    return loss, predicted, expected, present


def pairwise_rank_loss(scores, expected, temperature):
    losses = []
    weights = []
    count = len(scores)
    for i in range(count):
        for j in range(i + 1, count):
            gap = expected[j] - expected[i]
            if abs(float(gap)) < 1e-8:
                continue
            if gap > 0:
                high = scores[j]
                low = scores[i]
            else:
                high = scores[i]
                low = scores[j]
            losses.append(F.softplus(-(high - low) / temperature))
            weights.append(abs(gap))
    if not losses:
        return scores.new_tensor(0.0)
    losses_t = torch.stack(losses)
    weights_t = torch.tensor(weights, device=scores.device)
    return torch.sum(losses_t * weights_t) / weights_t.sum()


def rank_accuracy(scores, expected):
    correct = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if abs(float(expected[j] - expected[i])) < 1e-8:
                continue
            total += 1
            if (scores[j] > scores[i]) == (expected[j] > expected[i]):
                correct += 1
    return correct / total if total else math.nan


def spearman_safe(left, right):
    if len(left) < 2:
        return math.nan
    left_rank = pd.Series(left).rank().to_numpy()
    right_rank = pd.Series(right).rank().to_numpy()
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def evaluate_scheme_outputs(
    probabilities, labels, weights, schemes, targets, num_schemes, threshold
):
    probs_t = torch.as_tensor(probabilities, dtype=torch.float32)
    schemes_t = torch.as_tensor(schemes, dtype=torch.long)
    coverage_loss, predicted_cov, expected_cov, _ = scheme_coverage_loss(
        probs_t, schemes_t, targets, num_schemes
    )
    coverage_mae = float(torch.mean(torch.abs(predicted_cov - expected_cov)).item())
    scores = torch.logit(predicted_cov.clamp(1e-5, 1 - 1e-5))
    rank_loss = pairwise_rank_loss(
        scores, expected_cov, temperature=0.1
    )
    expected_np = expected_cov.detach().cpu().numpy()
    predicted_np = predicted_cov.detach().cpu().numpy()
    rho = spearman_safe(expected_np, predicted_np)
    pair_acc = rank_accuracy(list(predicted_np), list(expected_np))
    unique_labels = np.unique(labels.astype(int))
    auroc = (
        roc_auc_score(labels.astype(int), probabilities)
        if len(unique_labels) > 1
        else math.nan
    )
    auprc = (
        average_precision_score(labels.astype(int), probabilities)
        if len(unique_labels) > 1
        else math.nan
    )
    y_pred = (probabilities >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (labels == 1))
    fp = np.sum((y_pred == 1) & (labels == 0))
    fn = np.sum((y_pred == 0) & (labels == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    selection_score = (
        0.6 * ((rho + 1.0) / 2.0 if math.isfinite(rho) else 0.0)
        + 0.3 * (1.0 - min(coverage_mae / 0.2, 1.0))
        + 0.1 * (auprc if math.isfinite(auprc) else 0.0)
    )
    return {
        "val_scheme_auroc": float(auroc),
        "val_scheme_auprc": float(auprc),
        "val_threshold_f1": float(f1),
        "val_coverage_loss": float(coverage_loss.item()),
        "val_coverage_mae": coverage_mae,
        "val_rank_loss": float(rank_loss.item()),
        "val_spearman": float(rho),
        "val_pairwise_accuracy": float(pair_acc),
        "val_selection_score": float(selection_score),
    }


def best_threshold(labels, probabilities):
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = probabilities >= threshold
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum((~pred) & (labels == 1))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(threshold)
    return best_t, best_f1


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data_dir = args.data_dir.resolve()
    calibration_dataset_path = (
        args.calibration_dataset
        or input_data_dir / "calibration_raw.csv"
    ).resolve()
    pretrained_checkpoint = (
        args.pretrained or input_data_dir / "pretrained_model.pt"
    ).resolve()
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "model"
    data_dir = output_dir / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.targets)
    target_keys = [target_key(material, dose) for material, dose in targets]
    if len(set(target_keys)) != len(target_keys):
        raise ValueError("Duplicate target schemes")

    fit_df = load_raw_dataset(calibration_dataset_path)
    observed = sorted(fit_df["sample"].unique())
    if set(observed) != set(target_keys):
        raise ValueError(
            "calibration_raw.csv must contain exactly the target schemes "
            f"{target_keys}; observed {observed}"
        )
    coverage_by_scheme = raw_scheme_coverage(fit_df)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.cv_folds > 1:
        if not 0 <= args.fold_index < args.cv_folds:
            raise ValueError("fold-index must be in [0, cv-folds)")
        splitter = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=split_seed,
        )
        folds = list(splitter.split(fit_df, fit_df["sample"]))
        train_indices, val_indices = folds[args.fold_index]
        train_df = fit_df.iloc[train_indices]
        val_df = fit_df.iloc[val_indices]
    else:
        train_df, val_df = train_test_split(
            fit_df,
            test_size=args.val_fraction,
            random_state=split_seed,
            shuffle=True,
            stratify=fit_df["sample"],
        )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    scheme_to_index = {sample: idx for idx, sample in enumerate(target_keys)}
    scheme_target_values = [coverage_by_scheme[sample] for sample in target_keys]
    scheme_targets = torch.tensor(scheme_target_values, dtype=torch.float32)

    text_embeddings, sequence_embeddings, structure_embeddings = load_embeddings(
        input_data_dir
    )
    train_dataset = SchemeProteinDataset(
        train_df,
        text_embeddings,
        sequence_embeddings,
        structure_embeddings,
        scheme_to_index,
    )
    val_dataset = SchemeProteinDataset(
        val_df,
        text_embeddings,
        sequence_embeddings,
        structure_embeddings,
        scheme_to_index,
    )
    batch_sampler = BalancedSchemeBatchSampler(
        train_df,
        target_keys,
        num_per_scheme=args.batch_per_scheme,
        num_batches=args.batches_per_epoch,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8192,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if not pretrained_checkpoint.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrained_checkpoint}"
        )
    model = build_model(
        text_embeddings,
        sequence_embeddings,
        structure_embeddings,
        pretrained_checkpoint,
        device,
    )
    prepare_head_training(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=0.3 * args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=5,
        min_lr=args.lr * 0.01,
        cooldown=1,
    )

    label_counts = train_df["label"].value_counts()
    num_negative = int(label_counts.get(0, 0))
    num_positive = int(label_counts.get(1, 0))
    pos_weight_value = (
        2.0 * num_negative / num_positive if num_positive else 1.0
    )
    pos_weight = torch.tensor(
        [pos_weight_value], dtype=torch.float32, device=device
    )

    config = {
        "targets": args.targets,
        "target_keys": target_keys,
        "input_data_dir": str(input_data_dir),
        "calibration_dataset": str(calibration_dataset_path),
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "scheme_coverage": {
            sample: coverage_by_scheme[sample] for sample in target_keys
        },
        "loss_mode": args.loss_mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "split_seed": split_seed,
        "cv_folds": args.cv_folds,
        "fold_index": args.fold_index if args.cv_folds > 1 else None,
        "batch_per_scheme": args.batch_per_scheme,
        "batches_per_epoch": args.batches_per_epoch,
        "coverage_weight": args.coverage_weight,
        "rank_weight": args.rank_weight,
        "rank_temperature": args.rank_temperature,
        "pos_weight": pos_weight_value,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "device": str(device),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    train_df[["ID", "dose", "Accession", "sample", "label"]].to_csv(
        data_dir / "train.csv", index=False
    )
    val_df[["ID", "dose", "Accession", "sample", "label"]].to_csv(
        data_dir / "val.csv", index=False
    )

    history = []
    best_score = -math.inf
    best_epoch = 0
    best_threshold_value = 0.5
    stale_epochs = 0

    print(json.dumps(config, indent=2))
    for epoch in range(1, args.epochs + 1):
        batch_sampler.set_epoch(epoch)
        model.eval()
        model.classifier.train()
        totals = {
            "loss": 0.0,
            "bce": 0.0,
            "coverage": 0.0,
            "ranking": 0.0,
        }

        for (
            text,
            sequence,
            structure,
            labels,
            weights,
            schemes,
        ) in train_loader:
            text = text.to(device, non_blocking=True)
            sequence = sequence.to(device, non_blocking=True)
            structure = structure.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            schemes = schemes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(text, sequence, structure)
            probabilities = torch.sigmoid(logits)
            bce = F.binary_cross_entropy_with_logits(
                logits, labels, weight=weights, pos_weight=pos_weight
            )
            coverage = torch.zeros((), device=device)
            ranking = torch.zeros((), device=device)
            if args.loss_mode in {"coverage", "coverage_rank"}:
                coverage, predicted_cov, expected_cov, _ = scheme_coverage_loss(
                    probabilities,
                    schemes,
                    scheme_targets,
                    len(target_keys),
                )
                if args.loss_mode == "coverage_rank":
                    scores = torch.logit(
                        predicted_cov.clamp(1e-5, 1 - 1e-5)
                    )
                    ranking = pairwise_rank_loss(
                        scores, expected_cov, args.rank_temperature
                    )
            loss = bce
            if args.loss_mode in {"coverage", "coverage_rank"}:
                loss = loss + args.coverage_weight * coverage
            if args.loss_mode == "coverage_rank":
                loss = loss + args.rank_weight * ranking

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["bce"] += float(bce.item())
            totals["coverage"] += float(coverage.item())
            totals["ranking"] += float(ranking.item())

        for key in totals:
            totals[key] /= max(len(train_loader), 1)

        probabilities, labels, weights, schemes = predict_loader(
            model, val_loader, device
        )
        validation_threshold, validation_f1 = best_threshold(labels, probabilities)
        metrics = evaluate_scheme_outputs(
            probabilities,
            labels,
            weights,
            schemes,
            scheme_targets,
            len(target_keys),
            validation_threshold,
        )
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": totals["loss"],
                "train_bce": totals["bce"],
                "train_coverage_loss": totals["coverage"],
                "train_rank_loss": totals["ranking"],
                "val_best_threshold": validation_threshold,
                "val_best_f1": validation_f1,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        history.append(metrics)
        scheduler.step(metrics["val_selection_score"])

        if epoch == 1 or epoch % 10 == 0:
            print(
                "Epoch {epoch}/{epochs} "
                "loss={loss:.5f} bce={bce:.5f} cov={cov:.5f} rank={rank:.5f} "
                "val_auprc={auprc:.4f} val_mae={mae:.4f} "
                "val_spearman={rho:.4f} select={select:.4f}".format(
                    epoch=epoch,
                    epochs=args.epochs,
                    loss=totals["loss"],
                    bce=totals["bce"],
                    cov=totals["coverage"],
                    rank=totals["ranking"],
                    auprc=metrics["val_scheme_auprc"],
                    mae=metrics["val_coverage_mae"],
                    rho=metrics["val_spearman"],
                    select=metrics["val_selection_score"],
                ),
                flush=True,
            )

        if args.fixed_epochs:
            best_epoch = epoch
            best_threshold_value = validation_threshold
            best_score = metrics["val_selection_score"]
        elif metrics["val_selection_score"] > best_score:
            best_score = metrics["val_selection_score"]
            best_epoch = epoch
            best_threshold_value = validation_threshold
            torch.save(model.state_dict(), model_dir / "saved_model_ft.pt")
            stale_epochs = 0
        else:
            stale_epochs += 1

        torch.save(model.state_dict(), model_dir / "last_model_ft.pt")
        pd.DataFrame(history).to_csv(model_dir / "history_ft.csv", index=False)
        if not args.fixed_epochs and stale_epochs >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}",
                flush=True,
            )
            break

    if args.fixed_epochs:
        torch.save(model.state_dict(), model_dir / "saved_model_ft.pt")
    pd.DataFrame(history).to_csv(model_dir / "history_ft.csv", index=False)
    (model_dir / "best_epoch.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "best_threshold": best_threshold_value,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Training complete: best_epoch={best_epoch}, "
        f"best_selection={best_score:.6f}, threshold={best_threshold_value:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
