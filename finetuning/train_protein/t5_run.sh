#!/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vllm-env

cd /path/to/your/munap/finetuning/train_protein

export CUDA_VISIBLE_DEVICES="6"

# ==============================================================================
# 脚本说明：
# 只执行一个 Python 文件，循环 train_ratio 从 10 到 90，间隔 10。
# 其它超参数固定不变。
# ==============================================================================

# --- 1. 需要执行的 Python 文件 ---
PYTHON_FILE="t5_unseen_train.py"   # <-- 改成你的新文件名

# --- 2. 固定超参数配置 ---
EPOCHS=100
LR=3e-4
BATCH_SIZE=4096
NUM_WORKERS=8
BASE_SAVE_DIR="./output/from_scratch"

echo "Starting train_ratio sweep for ${PYTHON_FILE}"
echo "Shared params: epochs=${EPOCHS}, lr=${LR}, batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}"
echo "--------------------------------------------------------"

# --- 3. 循环 train_ratio: 10,20,...,100 ---
for TRAIN_RATIO in {10..100..10}; do

    # 建议每个 ratio 单独保存目录，避免覆盖
    SAVE_DIR="${BASE_SAVE_DIR}/${TRAIN_RATIO}"

    FULL_COMMAND="python ${PYTHON_FILE} \
        --train_ratio ${TRAIN_RATIO} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --save_dir ${SAVE_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS}"

    echo ""
    echo "=========================================================="
    echo "Running train_ratio=${TRAIN_RATIO}"
    echo "Command: ${FULL_COMMAND}"
    echo "=========================================================="

    eval "${FULL_COMMAND}"

    if [ $? -ne 0 ]; then
        echo "ERROR: train_ratio=${TRAIN_RATIO} failed. Aborting."
        exit 1
    fi

    echo "train_ratio=${TRAIN_RATIO} finished successfully."
done

echo ""
echo "--------------------------------------------------------"
echo "All train_ratio jobs completed successfully."
