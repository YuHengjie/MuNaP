# Laboratory-specific MuNaP adaptation and screening

This directory contains only the post-preprocessing workflow:

- five-fold head-only fine-tuning;
- inference on held-out candidate schemes;
- rank-percentile aggregation for final scheme prioritization.

Dataset construction, feature engineering, text embedding generation, and
protein embedding generation are intentionally not included. The input files
described in [`data/README.md`](data/README.md) must already be prepared.

## Files

- `model.py`: MuNaP fusion model and head-only fine-tuning utility.
- `common.py`: prepared-data loaders, balanced scheme sampler, and inference
  helpers.
- `train.py`: one fold of head-only fine-tuning with protein-level BCE,
  scheme-level coverage loss, and pairwise scheme-ranking loss.
- `infer.py`: inference of one fold model on all held-out candidate schemes.
- `aggregate.py`: equal-weight aggregation of scheme rank percentiles from
  multiple fold models.
- `run_fivefold.sh`: launches five fold models, runs inference, and generates
  the final rank ensemble.

## Prepared inputs

Place the following files in `wetlab/data/`:

```text
calibration_raw.csv
candidate_raw.csv
dataset_qc.csv
protein_embeddings.pkl
esmfold_protein_embeddings.pkl
text_embeddings.npy
pretrained_model.pt
```

See [`data/README.md`](data/README.md) for the required columns and file
formats.

## Train one fold

```bash
python wetlab/train.py \
  --data-dir wetlab/data \
  --calibration-dataset wetlab/data/calibration_raw.csv \
  --targets NP10:0.6,NP2:0.6,NP4:0.6,NP8:1.0 \
  --loss-mode coverage_rank \
  --cv-folds 5 \
  --fold-index 0 \
  --seed 42 \
  --split-seed 42 \
  --output-dir wetlab/runs/fold0
```

## Infer one fold

```bash
python wetlab/infer.py \
  --data-dir wetlab/data \
  --candidate-dataset wetlab/data/candidate_raw.csv \
  --dataset-qc wetlab/data/dataset_qc.csv \
  --targets NP10:0.6,NP2:0.6,NP4:0.6,NP8:1.0 \
  --run-dir wetlab/runs/fold0
```

## Run all five folds

```bash
GPU_LIST="0 1 2 3 4" \
TARGETS="NP10:0.6,NP2:0.6,NP4:0.6,NP8:1.0" \
bash wetlab/run_fivefold.sh
```

The output directory contains one folder per fold and
`final_rank_ensemble/`.

The four-scheme raw file is used only for fine-tuning. The 36-scheme raw file
defines the candidate scheme-protein pairs to score. Experimental labels and
scheme coverage for the 36 candidates are read only from `dataset_qc.csv`.
The QC file must contain exactly those 36 candidate schemes; extra or missing
schemes are rejected.

## Final aggregation

For each fold model, the scheme score is the mean predicted protein
interaction probability across all proteins in that scheme. Each model ranks
the 36 held-out schemes and converts the ranks to within-cohort percentiles.
The final priority score is the unweighted mean of the five percentile-rank
vectors. Schemes are sorted by the final priority score, and the top, middle,
and bottom 12 schemes are assigned to High, Medium, and Low tiers.
