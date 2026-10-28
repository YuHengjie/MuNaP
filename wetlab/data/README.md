# Prepared input files

The `wetlab` code starts after dataset construction and all text, sequence, and
structure embedding generation have been completed. Dataset-generation and
embedding scripts are intentionally not included.

The default protocol expects 40 schemes in total: four calibration schemes
used for fine-tuning and 36 held-out candidate schemes used for screening.
All input files below must use the same scheme and protein identifiers.

## `full_dataset.csv`

Protein-level observations used for fine-tuning and evaluation. Required
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

The reference prepared file contains 487,448 raw rows, 178,880 unique
scheme-protein observations, 40 schemes, and 4,472 protein accessions.

## `dataset_qc.csv`

Quality-controlled observations used only to define the experimental
coverage target. Required columns:

```text
ID
dose
Accession
RPA
```

- There must be one retained row per scheme-protein pair. Duplicate pairs are
  removed by keeping the first row.
- For each scheme, coverage is the fraction of retained proteins with
  `RPA > 1e-6`.
- Every scheme present in `full_dataset.csv` must also be present here.
- Extra columns such as `RPA_raw_mean`, `weight`, and `x_index` are currently
  ignored by the training and inference code.

The reference QC file contains 178,880 rows covering the same 40 schemes and
4,472 proteins as `full_dataset.csv`.

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

Every accession in `full_dataset.csv` must exist in both protein embedding
dictionaries. The reference QC file uses the same accession set. Every value
used by `x_index` must be a valid row in `text_embeddings.npy`.

## `pretrained_model.pt`

A PyTorch `state_dict` directly compatible with
`CrossAttentionClassifierGated` in `wetlab/model.py`. It is loaded with
`strict=True`; the final MuNaP checkpoint expects text, sequence, and
structure dimensions of 4,096, 2,560, and 384, respectively.

The prepared data and model files are not included in this repository.
