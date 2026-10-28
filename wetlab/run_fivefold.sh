#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
DATASET="${DATASET:-${DATA_DIR}/full_dataset.csv}"
DATASET_QC="${DATASET_QC:-${DATA_DIR}/dataset_qc.csv}"
PRETRAINED="${PRETRAINED:-${DATA_DIR}/pretrained_model.pt}"
TARGETS="${TARGETS:-NP10:0.6,NP2:0.6,NP4:0.6,NP8:1.0}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
GPU_LIST="${GPU_LIST:-4 5 6 7 1}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/seed${SEED}_5fold_rankonly_${STAMP}}"
LOG_DIR="${SCRIPT_DIR}/runs/logs"
LOG_FILE="${LOG_DIR}/seed${SEED}_5fold_rankonly_${STAMP}.log"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

run_fold() {
    local gpu="$1"
    local fold="$2"
    local run_dir="${RUN_ROOT}/fold${fold}"

    echo "===== fold ${fold} on GPU ${gpu} | $(date '+%F %T') ====="
    mkdir -p "${run_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT_DIR}/train.py" \
        --data-dir "${DATA_DIR}" \
        --dataset "${DATASET}" \
        --dataset-qc "${DATASET_QC}" \
        --pretrained "${PRETRAINED}" \
        --targets "${TARGETS}" \
        --loss-mode coverage_rank \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --split-seed "${SPLIT_SEED}" \
        --cv-folds 5 \
        --fold-index "${fold}" \
        --coverage-weight 0.3 \
        --rank-weight 0.2 \
        --rank-temperature 0.1 \
        --output-dir "${run_dir}"

    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT_DIR}/infer.py" \
        --data-dir "${DATA_DIR}" \
        --dataset "${DATASET}" \
        --dataset-qc "${DATASET_QC}" \
        --targets "${TARGETS}" \
        --run-dir "${run_dir}" \
        --seed "${SEED}"
}

echo "Five-fold rank-only run started at $(date '+%F %T')"
echo "Data directory: ${DATA_DIR}"
echo "Prepared dataset: ${DATASET}"
echo "Prepared QC dataset: ${DATASET_QC}"
echo "Pretrained checkpoint: ${PRETRAINED}"
echo "Run root: ${RUN_ROOT}"
echo "Model seed: ${SEED}; split seed: ${SPLIT_SEED}"
echo "Targets: ${TARGETS}"
echo "Epochs: ${EPOCHS}"
echo "GPUs: ${GPU_LIST}"
echo "Log: ${LOG_FILE}"

read -r GPU0 GPU1 GPU2 GPU3 GPU4 <<< "${GPU_LIST}"
if [[ -z "${GPU0:-}" || -z "${GPU1:-}" || -z "${GPU2:-}" || -z "${GPU3:-}" || -z "${GPU4:-}" ]]; then
    echo "GPU_LIST must contain exactly five GPU indices."
    exit 1
fi

run_fold "${GPU0}" 0 &
PID0=$!
run_fold "${GPU1}" 1 &
PID1=$!
run_fold "${GPU2}" 2 &
PID2=$!
run_fold "${GPU3}" 3 &
PID3=$!
run_fold "${GPU4}" 4 &
PID4=$!

status=0
wait "${PID0}" || status=1
wait "${PID1}" || status=1
wait "${PID2}" || status=1
wait "${PID3}" || status=1
wait "${PID4}" || status=1

if [[ "${status}" -ne 0 ]]; then
    echo "At least one fold failed."
    exit "${status}"
fi

"${PY}" "${SCRIPT_DIR}/aggregate.py" \
    --runs "${RUN_ROOT}/fold0" "${RUN_ROOT}/fold1" "${RUN_ROOT}/fold2" \
           "${RUN_ROOT}/fold3" "${RUN_ROOT}/fold4" \
    --output-dir "${RUN_ROOT}/final_rank_ensemble"

echo "Five-fold rank-only run completed at $(date '+%F %T')"
