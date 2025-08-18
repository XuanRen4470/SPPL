#!/bin/bash -l
#SBATCH --job-name=plan_bench_reuse
#SBATCH -p a100
#SBATCH -n 8 # SBATCH -N 2
#SBATCH -A strategic
#SBATCH --time=2-00:00:00 # time allocation
#SBATCH --gres=gpu:1,tmpfs:100G # generic resource required
#SBATCH --mem=80GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xuan4470@gmail.com
#SBATCH --signal B:USR2

conda init
conda activate acl_2025

# Variables
train_task_name=plan_bench_reuse
file_suffix=dec_8
n_train=100
sft_epoch="40"
sft_lr=2e-4
num_of_sft_checkpoints=20
disable_final_eval=false
debug_mode=false
seed_num=0
variation_suffix=none
model_type=llama_3_instruct
train_method=claude

python /gpfs/users/a1796450/ACL_2024/Minimum_Change/main.py --file_suffix $file_suffix --train_task_name $train_task_name --n_train $n_train --n_eval 1000 --n_validation 300 --seed_num $seed_num --sft_epoch $sft_epoch --sft_lr $sft_lr --num_of_sft_checkpoints $num_of_sft_checkpoints --disable_final_eval $disable_final_eval --train_method $train_method --model_type $model_type --debug_mode $debug_mode --variation_suffix $variation_suffix