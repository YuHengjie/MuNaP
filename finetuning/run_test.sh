#!/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vllm-env

# 假设您要进入的基础目录
BASE_DIR="/path/to/your/munap/finetuning"


SUBDIRS=("train_nano" "train_protein" ) 

# 循环遍历子文件夹并执行命令
for DIR in "${SUBDIRS[@]}"; do
    TARGET_DIR="$BASE_DIR/$DIR"
    
    echo "--- Entering Directory: $TARGET_DIR ---"
    
    # 1. 进入目标文件夹
    cd "$TARGET_DIR" || { echo "Error: Failed to change directory to $TARGET_DIR. Aborting."; exit 1; }
    
    # 2. 执行 Python 脚本
    echo "Executing python t7_model_binary_test.py ..."
    python t7_model_binary_test.py
    
    echo "Executing python t2_model_binary_test.py ..."
    python t2_model_binary_test.py

    # 3. 返回基础目录，以便下一次循环进入新的子目录
    # 注意：如果不需要在每个循环后返回，可以省略这行，但为了安全和清晰，建议保留
    cd "$BASE_DIR" 
    
    echo "--- Finished $DIR ---"
done

echo "All tasks completed."