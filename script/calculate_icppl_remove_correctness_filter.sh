#!/bin/bash
#SBATCH --job-name=4_icpll
#SBATCH -p a100
#SBATCH -A strategic
#SBATCH -n 8
#SBATCH --time=2-00:00:00 # time allocation, format (D-HH:MM)
#SBATCH --gres=gpu:1,tmpfs:100G # requires 1 GPU, temp space
#SBATCH --mem=80GB # memory required per node
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xuan4470@gmail.com
#SBATCH --signal B:USR2

# Initialize and activate conda environment
conda init
conda activate acl_2025

# python /gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/icppl_calculation_remove_correctness_filter.py --num_of_incontext_examples 2 --model_name mistral

# python /gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/icppl_calculation_remove_correctness_filter.py --num_of_incontext_examples 2 --model_name llama_3_instruct

python /gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/icppl_calculation_remove_correctness_filter.py --num_of_incontext_examples 2 --model_name qwen