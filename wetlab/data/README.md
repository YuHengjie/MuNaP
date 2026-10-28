# Prepared input files

The `wetlab` code starts after dataset construction and all text, sequence, and
structure embedding generation have been completed. Dataset-generation and
embedding scripts are intentionally not included.

The default protocol expects 40 schemes in total: four calibration schemes
used for fine-tuning and 36 held-out candidate schemes used for screening.
The raw abundance data are split into two files:

| File | Schemes | Role |
| --- | ---: | --- |
| `calibration_raw.csv` | 4 | raw protein abundance used for all fine-tuning losses |
| `candidate_raw.csv` | 36 | raw protein abundance used to define screening pairs |

`dataset_qc.csv` contains quality-controlled measurements and is used only for
experimental comparison after prediction. It is not read during fine-tuning.
All input files must use the same scheme and protein identifiers.

## Raw abundance files

`calibration_raw.csv` and `candidate_raw.csv` use the same schema. Required
columns:

```text
ID
dose
Accession
RPA
x_index
```

- `ID` is the nanomaterial identifier, for example `NP10`.
- `dose` and `RPA` are numeric. A protein is labelled enriched when
  `RPA > 1e-6`.
- The scheme key is formed as `{ID}_{dose:g}`, for example `NP10_0.6`.
- `x_index` is a zero-based row index into `text_embeddings.npy`.
- Repeated rows for the same `ID`, `dose`, and `Accession` are collapsed by
  averaging `RPA`; the first `x_index` is retained. Duplicate rows belonging
  to the same scheme must therefore use the same text index.
- Additional columns are ignored by the loader.

### `calibration_raw.csv`

- Must contain exactly the four schemes listed in `--targets`.
- Its protein-level `RPA` values define the BCE labels.
- Its scheme-level enrichment fractions define the Coverage and Ranking loss
  targets. QC measurements are not used to fit these targets.

The reference file contains 40,248 raw rows, 17,888 unique scheme-protein
observations, four schemes, and 4,472 protein accessions.

### `candidate_raw.csv`

- Must contain exactly the 36 held-out candidate schemes.
- It defines which scheme-protein pairs are scored and supplies their
  `x_index` values.
- Its `RPA` column is retained as raw metadata but is not used for training or
  experimental evaluation.

The reference file contains 447,200 raw rows, 160,992 unique scheme-protein
observations, 36 schemes, and 4,472 protein accessions.

## `dataset_qc.csv`

Quality-controlled observations used only to evaluate the 36 predictions.
Required columns:

```text
ID
dose
Accession
RPA
```

- There must be one retained row per scheme-protein pair. Duplicate pairs are
  removed by keeping the first row.
- Predictions and experimental labels are joined by `sample` and `Accession`.
- Protein-level AUROC, AUPRC, accuracy, precision, recall, F1, and MCC use the
  QC-derived label `RPA > 1e-6`.
- Scheme coverage is the fraction of candidate proteins with a QC-derived
  label of one.
- Every candidate scheme-protein pair must be present.
- Extra columns such as `RPA_raw_mean`, `weight`, and `x_index` are currently
  ignored by the training and inference code.

The reference QC file contains 178,880 rows covering all 40 schemes. The 36
candidate schemes contribute 160,992 comparison rows; QC rows belonging to
the four calibration schemes are not used for inference evaluation.

## Embeddings

- `protein_embeddings.pkl`: a Python dictionary mapping each protein
  `Accession` to a one-dimensional ESM2 vector. The reference vectors have
  2,560 dimensions.
- `esmfold_protein_embeddings.pkl`: a Python dictionary mapping each protein
  `Accession` to a one-dimensional ESMFold vector. The reference vectors have
  384 dimensions.
- `text_embeddings.npy`: a two-dimensional numeric array of precomputed
  experimental-context vectors. The reference matrix has shape `40 x 4096`,
  with one row per scheme.

Every accession in either raw abundance file must exist in both protein
embedding dictionaries. Every value used by `x_index` must be a valid row in
`text_embeddings.npy`.

## `pretrained_model.pt`

A PyTorch `state_dict` directly compatible with
`CrossAttentionClassifierGated` in `wetlab/model.py`. It is loaded with
`strict=True`; the final MuNaP checkpoint expects text, sequence, and
structure dimensions of 4,096, 2,560, and 384, respectively.

The prepared data and model files are not included in this repository.
