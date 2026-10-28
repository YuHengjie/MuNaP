# Prepared input files

The `adaption_screen` code starts after all dataset and embedding
preprocessing has been completed. Do not place dataset-generation scripts
here.

Required files:

## `full_dataset.csv`

One row per protein-scheme measurement. Required columns:

```text
ID
dose
Accession
RPA
x_index
```

`x_index` must index the corresponding row in `text_embeddings.npy`.
Replicate rows for the same `ID`, `dose`, and `Accession` are averaged before
training.

## `dataset_qc.csv`

Quality-controlled experimental data used to calculate scheme coverage.
Required columns:

```text
ID
dose
Accession
RPA
```

## Embeddings

- `protein_embeddings.pkl`: mapping from protein accession to the ESM2 vector.
- `esmfold_protein_embeddings.pkl`: mapping from protein accession to the
  ESMFold vector.
- `text_embeddings.npy`: matrix of precomputed experimental-context vectors.

## Pretrained model

- `pretrained_model.pt`: a PyTorch state dictionary compatible with
  `CrossAttentionClassifierGated` in `model.py`.

The data files are not included in this repository and must be obtained or
prepared separately.
