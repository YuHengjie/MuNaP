from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from model import CrossAttentionClassifierGated, finetune_head_only


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dose_key(value: float) -> str:
    return f"{float(value):g}"


def parse_targets(spec: str) -> list[tuple[str, float]]:
    targets: list[tuple[str, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        material, dose = item.split(":", 1)
        targets.append((material.strip(), float(dose)))
    return targets


def target_key(material: str, dose: float) -> str:
    return f"{material}_{dose_key(dose)}"


def load_full_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dose"] = pd.to_numeric(df["dose"], errors="raise")
    df["RPA"] = pd.to_numeric(df["RPA"], errors="raise")
    df["x_index"] = pd.to_numeric(df["x_index"], errors="coerce").astype(int)
    df["label"] = (df["RPA"] > 1e-6).astype(np.float32)

    # Some source rows repeat an accession after trait expansion. Collapse them
    # so a scheme bag is one observation per protein, matching coverage.
    df = (
        df.groupby(["ID", "dose", "Accession"], as_index=False)
        .agg(RPA=("RPA", "mean"), x_index=("x_index", "first"))
        .reset_index(drop=True)
    )
    df["label"] = (df["RPA"] > 1e-6).astype(np.float32)
    df["sample"] = (
        df["ID"].astype(str) + "_" + df["dose"].map(dose_key)
    )
    return df


def scheme_coverage_map(qc_path: Path, df: pd.DataFrame) -> dict[str, float]:
    # Keep per-protein labels from the prepared training dataset, while using
    # the independently prepared QC dataset for scheme-level coverage targets.
    qc = pd.read_csv(qc_path)
    qc["dose"] = pd.to_numeric(qc["dose"], errors="raise")
    qc["RPA"] = pd.to_numeric(qc["RPA"], errors="raise")
    qc["sample"] = (
        qc["ID"].astype(str) + "_" + qc["dose"].map(dose_key)
    )
    qc = qc.drop_duplicates(["sample", "Accession"])
    coverage = qc.assign(label=(qc["RPA"] > 1e-6).astype(float)).groupby(
        "sample"
    )["label"].mean()
    missing = sorted(set(df["sample"].unique()) - set(coverage.index))
    if missing:
        raise ValueError(f"Coverage missing for schemes: {missing}")
    return coverage.astype(float).to_dict()


def load_embeddings(data_dir: Path = DATA_DIR):
    with open(data_dir / "protein_embeddings.pkl", "rb") as handle:
        sequence_embeddings = pickle.load(handle)
    with open(data_dir / "esmfold_protein_embeddings.pkl", "rb") as handle:
        structure_embeddings = pickle.load(handle)
    text_embeddings = np.load(data_dir / "text_embeddings.npy")
    return text_embeddings, sequence_embeddings, structure_embeddings


class SchemeProteinDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_embeddings: np.ndarray,
        sequence_embeddings: dict,
        structure_embeddings: dict,
        scheme_to_index: dict[str, int] | None = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.text_embeddings = text_embeddings
        self.sequence_embeddings = sequence_embeddings
        self.structure_embeddings = structure_embeddings
        self.scheme_to_index = scheme_to_index or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = self.text_embeddings[int(row["x_index"])]
        sequence = self.sequence_embeddings[row["Accession"]]
        structure = self.structure_embeddings[row["Accession"]]
        label = float(row["label"])
        weight = float(row.get("Overall data quality", 1.0))
        scheme_index = int(self.scheme_to_index.get(row["sample"], -1))
        return (
            torch.as_tensor(text, dtype=torch.float32),
            torch.as_tensor(sequence, dtype=torch.float32),
            torch.as_tensor(structure, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(scheme_index, dtype=torch.long),
        )


class BalancedSchemeBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        df: pd.DataFrame,
        samples: Iterable[str],
        num_per_scheme: int,
        num_batches: int,
        seed: int,
    ):
        self.num_per_scheme = num_per_scheme
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0
        self.scheme_indices = {}
        for sample in samples:
            indices = np.flatnonzero(df["sample"].to_numpy() == sample)
            if len(indices) == 0:
                raise ValueError(f"No training proteins found for {sample}")
            self.scheme_indices[sample] = indices
        self.samples = list(samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for sample in self.samples:
                source = self.scheme_indices[sample]
                replace = len(source) < self.num_per_scheme
                chosen = rng.choice(
                    source, size=self.num_per_scheme, replace=replace
                )
                batch.extend(int(value) for value in chosen)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def build_model(
    text_embeddings: np.ndarray,
    sequence_embeddings: dict,
    structure_embeddings: dict,
    checkpoint: Path | None,
    device: torch.device,
):
    model = CrossAttentionClassifierGated(
        text_embeddings.shape[1],
        next(iter(sequence_embeddings.values())).shape[0],
        next(iter(structure_embeddings.values())).shape[0],
    ).to(device)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {checkpoint}")
    return model


def prepare_head_training(model: torch.nn.Module) -> None:
    finetune_head_only(model)


@torch.no_grad()
def predict_loader(model, loader, device: torch.device):
    model.eval()
    probabilities = []
    labels = []
    weights = []
    schemes = []
    for text, sequence, structure, label, weight, scheme in loader:
        logits, _ = model(
            text.to(device, non_blocking=True),
            sequence.to(device, non_blocking=True),
            structure.to(device, non_blocking=True),
        )
        probabilities.append(torch.sigmoid(logits).detach().cpu())
        labels.append(label.detach().cpu())
        weights.append(weight.detach().cpu())
        schemes.append(scheme.detach().cpu())
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(weights).numpy(),
        torch.cat(schemes).numpy(),
    )
