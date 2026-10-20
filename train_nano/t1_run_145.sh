#!/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vllm-env

cd /path/to/your/trace/train_nano

export CUDA_VISIBLE_DEVICES="3"

# ==============================================================================
# 脚本说明：
# 集中配置五个不同的训练任务，将通用参数定义为 BASE_CONFIG 并附加。
# WARNING: 所有任务使用同一个 --save_dir，请确保 Python 脚本内部分文件保存。
# ==============================================================================




# --- 1. 定义基础通用配置字符串 (共享参数) ---
# 包含：--save_dir, --batch_size, --num_workers
BASE_CONFIG="--save_dir ./output/stage_145 --batch_size 4096 --num_workers 8"

# --- 2. Python 文件名列表 ---
declare -a PYTHON_FILES=(
    "t1_model_binary_stage_1.py" # 对应任务 1
    "t1_model_binary_stage_4.py" # 对应任务 2
    "t1_model_binary_stage_5.py" # 对应任务 5
)

# --- 3. 对应的参数列表 (仅包含 --epochs 和 --lr) ---
# 注意：我们移除了 BASE_CONFIG 中的共享参数
declare -a CONFIGS=(
    "--epochs 100 --lr 3e-4"
    "--epochs 50 --lr 1e-4 --load_stage 1"
    "--epochs 25 --lr 5e-5 --load_stage 4"
)

# 确保两个列表的长度一致
if [ ${#PYTHON_FILES[@]} -ne ${#CONFIGS[@]} ]; then
    echo "ERROR: PYTHON_FILES 列表和 CONFIGS 列表的元素数量不一致！"
    exit 1
fi

echo "Starting execution of ${#PYTHON_FILES[@]} training jobs."
echo "Shared Config: ${BASE_CONFIG}"
echo "--------------------------------------------------------"

# --- 4. 循环执行任务 ---
for i in "${!PYTHON_FILES[@]}"; do
    FILE="${PYTHON_FILES[$i]}"
    PARAMS="${CONFIGS[$i]}"
    JOB_NUMBER=$((i + 1))
    
    # 构造完整的参数字符串：特定参数 + 通用参数
    FULL_COMMAND="python ${FILE} ${PARAMS} ${BASE_CONFIG}"

    echo ""
    echo "=========================================================="
    echo "JOB ${JOB_NUMBER}: Executing ${FILE}"
    echo "Command: ${FULL_COMMAND}"
    echo "=========================================================="

    # 使用 eval 执行命令，将参数字符串展开
    eval "${FULL_COMMAND}"

    # 检查上一步是否成功
    if [ $? -ne 0 ]; then
        echo "ERROR: Job ${JOB_NUMBER} (${FILE}) failed. Aborting."
        exit 1
    fi
    echo "Job ${JOB_NUMBER} finished successfully."
done

echo ""
echo "--------------------------------------------------------"
echo "All training jobs completed successfully."