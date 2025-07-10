#!/bin/bash
#SBATCH --job-name=calc_metrics
#SBATCH -p a100
#SBATCH -A strategic
#SBATCH -n 8
#SBATCH --time=1-00:00:00 # time allocation, format (D-HH:MM)
#SBATCH --gres=gpu:1,tmpfs:100G # requires 1 GPU, temp space
#SBATCH --mem=40GB # memory required per node
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xuan4470@gmail.com
#SBATCH --signal B:USR2

conda init
conda activate acl_2025

# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/initial_prediction.py




 
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/other_metrics_calculation.py --model_name mistral --suffix unprocessed
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/other_metrics_calculation.py --model_name qwen --suffix unprocessed
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/other_metrics_calculation.py --model_name llama_3_instruct --suffix unprocessed



# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/other_metrics_calculation.py --model_name mistral 
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/other_metrics_calculation.py --model_name qwen 
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/other_metrics_calculation.py --model_name llama_3_instruct 






# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name mistral --num_of_incontext_examples 2 --suffix unprocessed
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name qwen --num_of_incontext_examples 2 --suffix unprocessed
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name llama_3_instruct --num_of_incontext_examples 2 --suffix unprocessed



# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name mistral --num_of_incontext_examples 2
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name qwen --num_of_incontext_examples 2
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name llama_3_instruct --num_of_incontext_examples 2



# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name mistral --num_of_incontext_examples 3 --suffix unprocessed
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name qwen --num_of_incontext_examples 3 --suffix unprocessed
# python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name llama_3_instruct --num_of_incontext_examples 3 --suffix unprocessed


python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name mistral --num_of_incontext_examples 3
python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name qwen --num_of_incontext_examples 3
python /gpfs/users/a1796450/ACL_2024/SPPL/Mix_Score_Ranking_Calculation/icppl_calculation.py --model_name llama_3_instruct --num_of_incontext_examples 3



# git config --global user.name  "XuanRen4470"
# git config --global user.email "你的邮箱"