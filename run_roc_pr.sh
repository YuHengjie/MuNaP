#!/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vllm-env

# 基础目录
BASE_DIR="/path/to/your/trace"
# 依次执行的子文件夹列表
#SUBDIRS=("train_basic" "train_basic_with_fill" "train_basic_only_protein" "train_basic_only_text" "train_date" "train_date_with_fill" "train_nano" "train_nano_with_fill" "train_protein" "train_protein_with_fill") 
#SUBDIRS=("train_basic_10" "train_basic_30" "train_basic_50" "train_basic_75" "train_basic_100") 
#SUBDIRS=("train_date_10" "train_date_30" "train_date_50" "train_date_75" "train_date_100") 
#SUBDIRS=("train_nano" "train_protein" "train_basic_only_protein" "train_basic_only_text") 
#SUBDIRS=("train_date_10" "train_date_50" "train_date_75" "train_date_100" "train_nano" "train_protein") 
#SUBDIRS=("train_basic_10" "train_basic_30" "train_basic_50" "train_basic_75" "train_basic_100" "train_date_10" "train_date_30" "train_date_50" "train_date_75" "train_date_100") 
#SUBDIRS=("train_basic_10" "train_basic_30" "train_basic_50" "train_basic_75" "train_basic_100" "train_basic_only_protein" "train_basic_only_text" "train_date_10" "train_date_30" "train_date_50" "train_date_75" "train_date_100" "train_nano" "train_protein") 
#SUBDIRS=("train_basic_only_protein" "train_basic_only_text" "train_nano" "train_protein") 
SUBDIRS=("train_basic_only_protein" "train_basic_only_text" "train_basic_10") 

# 
# 循环遍历子文件夹并执行命令
for DIR in "${SUBDIRS[@]}"; do
    TARGET_DIR="$BASE_DIR/$DIR"
    
    echo "--- Entering Directory: $TARGET_DIR ---"
    
    # 1. 进入目标文件夹
    cd "$TARGET_DIR" || { echo "Error: Failed to change directory to $TARGET_DIR. Aborting."; exit 1; }
    
    # 2. 执行 Python 脚本
    echo "Executing python t3_model_binary_test_ROC.py ..."
    python t3_model_binary_test_ROC.py
    
    # 3. 返回基础目录，以便下一次循环进入新的子目录
    # 注意：如果不需要在每个循环后返回，可以省略这行，但为了安全和清晰，建议保留
    cd "$BASE_DIR" 
    
    echo "--- Finished $DIR ---"
done

echo "All tasks completed."