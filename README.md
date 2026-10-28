# MuNaP

**MuNaP** (**Mu**ltimodal foundation model for **Na**nomaterial-**P**rotein interaction) is a multimodal foundation model integrating protein sequence and structural representations with text-encoded experimental conditions for generalizable prediction of nanomaterial-protein interactions (NPIs) and automated selection of nanomaterials for plasma protein enrichment. It supports virtual screening and prioritization of plasma protein nano-enrichment schemes, with generalization to unseen proteins, nanomaterials and experimental conditions.

MuNaP learns from 1.97 million literature-curated interaction records by integrating:

- protein sequence representations derived from ESM2,
- protein structural representations derived from ESMFold,
- a text-based representation of the experimental context,
- sample-level quality weighting during model optimization.

To improve generalization to previously unseen materials and proteins, MuNaP uses interaction data from heterogeneous biofluids to broaden the nanomaterial-protein interaction space encountered during training. These data are incorporated through a biologically ordered curriculum that progressively introduces increasingly shifted source domains before refinement on human plasma data.

This repository provides the supporting code for the associated research manuscript and is released for transparency, reproducibility, and research use.

## Model

MuNaP is a **13.5B-parameter multimodal foundation model** that integrates protein sequence and structure representations with text-encoded experimental conditions to enable generalizable prediction of nanomaterial-protein interactions (NPIs) and automated selection of nanomaterials for plasma protein enrichment. The exact parameter count is **13,489,546,853** (approximately 13.5B).

### Parameter breakdown

| Module | Base model | Output dimension | Parameters |
| --- | --- | ---: | ---: |
| Text encoder | Linq-Embed-Mistral (Mistral-7B) | 4096 | **7,110,660,096** |
| Protein sequence encoder | ESM2 `esm2_t36_3B_UR50D` | 2560 | **2,839,000,481** |
| Protein structure encoder | ESMFold `esmfold_v1` | 384 | **3,525,038,915** |
| Multimodal fusion and classifier | `CrossAttentionClassifierGated` | 1024 | **14,847,361** |
| **Total** | | | **13,489,546,853** |

ESMFold internally uses `esm2_3B` as its language-model backbone. The ESM2 sequence encoder and ESMFold are two independently loaded, frozen checkpoints, so their parameters are counted separately.

#### Fusion-head breakdown

| Component | Structure | Parameters |
| --- | --- | ---: |
| Text projection | 4096 -> 1024 + BatchNorm | 4,197,376 |
| Protein sequence projection | 2560 -> 1024 + BatchNorm | 2,624,512 |
| Protein structure projection | 384 -> 1024 + BatchNorm | 396,288 |
| Gated fusion | 2048 -> 1024 -> 1024 | 3,147,776 |
| Post-fusion BatchNorm | 1024 dimensions | 2,048 |
| Cross-attention | 1024 dimensions, 8 heads | 4,198,400 |
| Classifier | 1024 -> 256 -> 64 -> 1 | 280,961 |
| **Total** | | **14,847,361** |

The value `14,855,557`, sometimes obtained by summing all tensors in the fusion-head `state_dict`, includes **8,196 BatchNorm buffers** (running mean, running variance, and `num_batches_tracked`). It is not the number of model parameters.

### Modular encoding and inference

Running the full 13.5B-parameter model end-to-end for every candidate combination is computationally expensive and impractical on a personal computer. MuNaP therefore uses a modular pipeline:

1. **Offline modality encoding.** Protein sequences, protein structures, and structured experimental text are encoded independently by their respective modules and cached:
   - text: `text_embeddings.npy`, 4096 dimensions;
   - protein sequence: `protein_embeddings.pkl`, 2560 dimensions;
   - protein structure: `esmfold_protein_embeddings.pkl`, 384 dimensions.
2. **Lightweight multimodal inference.** The cached embeddings are combined and passed to `CrossAttentionClassifierGated` for prediction, adaptation, and virtual screening.

This design avoids repeatedly re-encoding the same structured text and protein modalities for every pairwise combination. Only the 14,847,361-parameter fusion module is instantiated for downstream inference; the three base encoders are not loaded. As a result, inference and laboratory-specific adaptation can run on a personal computer, and the overall workflow is substantially faster.

During laboratory-specific adaptation in `adaption_screen/train.py`, only the **280,961-parameter classifier head** is trainable; the remaining **14,566,400** fusion parameters are frozen.

---

## Repository structure

```
MuNaP/
├── data/
├── text_embedding/
├── protein_embedding/
├── train_basic_10/
├── train_basic_30/
├── train_basic_50/
├── train_basic_75/
├── train_basic_100/
├── train_basic_only_protein/
├── train_basic_only_text/
├── train_date_10/
├── train_date_30/
├── train_date_50/
├── train_date_75/
├── train_date_100/
├── train_nano/
├── train_protein/
├── text_embedding_4_explain/
├── model_explain/
├── finetuning/
├── adaption_screen/
│   ├── data/
│   ├── model.py
│   ├── common.py
│   ├── train.py
│   ├── infer.py
│   ├── aggregate.py
│   └── run_fivefold.sh
├── run_test.sh
└── run_roc_pr.sh
```

---

## Directory description

### `data/`

Stores the input dataset and the data-split artifacts generated by the scripts.

- The main curated dataset, `all_curated_nonfill.csv`, should be downloaded and placed here.
- Data-split scripts write files such as `data_split_indices.json`, `basic_*.csv`, `date_external_plasma_human.csv`, `nano_external_plasma_human.csv`, and `protein_external_plasma_human.csv`, together with the corresponding train/validation/test subsets.

### `text_embedding/`

Encodes experimental-context information into vector representations. Curated material, biological, and workflow descriptors are converted into structured textual descriptions and then encoded with a pretrained text model (SentenceTransformer / Linq-Embed-Mistral). The resulting embeddings are used as the experimental-context modality.

### `protein_embedding/`

Computes and stores protein representations used by the multimodal model:

- sequence embeddings from ESM2,
- structural embeddings from ESMFold,
- combined sequence-structure representations,
- embeddings for unseen proteins.

Precomputing these representations avoids repeated inference during model training.

### `train_basic_10/`, `train_basic_30/`, `train_basic_50/`, `train_basic_75/`, `train_basic_100/`

Original/standard data-split experiments with curriculum learning. These directories construct the biologically ordered curriculum from non-blood biofluids, non-human blood-derived biofluids, human serum and low-quality human plasma, and human plasma, and train the MuNaP model.

The numeric suffix indicates the fraction of the high-quality human-plasma training pool used in the final curriculum stage (Stage E): 10%, 30%, 50%, 75%, or 100%. This set of experiments tests how much typical human-plasma data is needed at the final refinement stage.

### `train_basic_only_protein/`

Ablation experiment using only the protein modality (sequence plus structure), without the experimental-context/text modality.

### `train_basic_only_text/`

Ablation experiment using only the experimental-context/text modality, without the protein modality.

### `train_date_10/`, `train_date_30/`, `train_date_50/`, `train_date_75/`, `train_date_100/`

Temporal generalization experiments based on a time/publication-based split. Five recent independent human-plasma proteomics studies are held out as an external test set and excluded from model development; the remaining data are used for the same curriculum-learning workflow.

As with `train_basic_*`, the numeric suffix indicates the fraction of high-quality human-plasma training data used in the final curriculum stage.

### `train_nano/`

Nanomaterial-held-out generalization experiment. Nanomaterials are defined by core-surface modification combinations; 20% of qualified combinations are withheld together with all associated samples. The model is trained on the remaining data and evaluated on the held-out nanomaterial combinations.

### `train_protein/`

Protein-held-out generalization experiment. 25% of moderately represented protein accessions are withheld, and all interactions involving those proteins are excluded from training. The model is evaluated on the corresponding held-out protein set.

### `text_embedding_4_explain/`

Generates experimental-context embeddings specifically for feature-ablation explanation experiments, for example by re-encoding the textual descriptions after removing or modifying selected features or feature subsets.

### `model_explain/`

Contains model-explanation modules. These scripts quantify how model performance (for example F1 or loss) changes under feature or modality ablation, and produce feature-importance tables and plots used to interpret multimodal contributions.

### `finetuning/`

Contains adaptation experiments for unseen materials and proteins.

- `train_nano/`: all gold-nanoparticle measurements are excluded during pretraining and introduced only during adaptation.
- `train_protein/`: a set of protein accessions is completely withheld from pretraining and introduced only during adaptation.

For each target domain, the scripts split independent test and validation sets, train models with increasing fractions of the remaining target-domain data, and compare training from scratch with MuNaP adaptation, in which the pretrained representation and fusion modules are retained and only the prediction head is optimized.

### `adaption_screen/`

Contains the post-preprocessing workflow for laboratory-specific MuNaP adaptation and virtual screening of plasma nano-enrichment schemes.

- `data/`: prepared protein embeddings, experimental-context embeddings, pretrained checkpoint, full protein-scheme dataset, and QC dataset. Dataset preparation and embedding-generation code are intentionally not included.
- `model.py`: MuNaP fusion model and classifier-head-only adaptation utility.
- `common.py`: prepared-data loaders, scheme-balanced sampling, and inference helpers.
- `train.py`: five-fold-compatible fine-tuning with protein-level BCE, scheme coverage loss, and pairwise scheme-ranking loss.
- `infer.py`: fold-model inference and scheme-level ranking on held-out candidates.
- `aggregate.py`: equal-weight aggregation of within-cohort scheme rank percentiles across fold models.
- `run_fivefold.sh`: end-to-end five-fold training, inference, and rank aggregation.

See `adaption_screen/README.md` and `adaption_screen/data/README.md` for the required prepared inputs and commands.

### `run_test.sh` and `run_roc_pr.sh`

Top-level batch scripts that iterate over selected experiment folders and run the evaluation scripts:

- `run_test.sh` runs `t2_model_binary_test.py`,
- `run_roc_pr.sh` runs `t3_model_binary_test_ROC.py` for ROC/PR evaluation.

---

## Notes

- This repository was previously released under the names COMPASS and TRACE. The code and experimental setup are unchanged by the rename.
- This repository is intended for research and academic use.
- Commercial use of the software is strictly prohibited.
- The implementation reflects the experimental setup described in the manuscript and may be further refined or reorganized after peer review.

---

## Citation

If you use this work, please cite the arXiv preprint:

[https://arxiv.org/abs/2507.14245](https://arxiv.org/abs/2507.14245)

Formal citation information will be updated after publication.
