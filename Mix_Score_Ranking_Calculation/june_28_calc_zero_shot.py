import sys
import os

# 将上一级目录加入 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import json
from utils.data_loader import add_gold_label, load_gold_label_and_question_list
from evaluation.eval import Check_Correctness

# task_name = 'gsm8k'
# task_name = 'math_algebra'
# task_name = 'math_geometry'
# task_name = 'arc_challenge'
# task_name = 'piqa'
# task_name = 'ecqa'
# task_name = 'agieval'
task_name = 'squad'


if task_name == 'gsm8k':
    data_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/GSM8K/gpt4.json'
    predict_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_gsm8k_qwen_initial_prediction_1000.json'
elif task_name == 'math_algebra':
    data_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH_ALGEBRA/gpt4.json'
    predict_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_math_algebra_qwen_initial_prediction_1000.json'
elif task_name == 'arc_challenge':
    data_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/ARC_CHALLENGE/gpt4.json'
    predict_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_arc_challenge_qwen_initial_prediction_1000.json'
elif task_name == 'piqa':
    data_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PIQA/gpt4.json'
    predict_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_piqa_qwen_initial_prediction_1000.json'
elif task_name == 'ecqa':
    data_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/ECQA/gpt4.json'
    predict_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_ecqa_qwen_initial_prediction_1000.json'
elif task_name == 'agieval':
    data_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/AGIEVAL/gpt4.json'
    predict_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_agieval_qwen_initial_prediction_1000.json'
elif task_name == 'squad':
    data_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/SQUAD/gpt4.json'
    predict_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_squad_qwen_initial_prediction_1000.json'




with open(data_path, 'r') as f:
    data_list = json.load(f)
with open(predict_path, 'r') as f:
    prediction_list = json.load(f)
    

total_correct = 0
n_data_creation = 1000
    

data_list = data_list[:n_data_creation]

gold_label_list, groundtruth_list, question_list = load_gold_label_and_question_list(task_name, n_data_creation)    
original_question_list = question_list.copy()
original_gold_label_list = gold_label_list.copy()
original_groundtruth_list = groundtruth_list.copy()

for index, item in enumerate(data_list):
    pred_temp = []
    data_temp = []
    pred_temp.append(prediction_list['initial_prediction'][index])
    item = add_gold_label(task_name, item, gold_label_list[index])
    
    data_temp.append(item)
    accuracy, cover_ratio = Check_Correctness(pred_temp, data_temp, task_name, '/gpfs/users/a1796450/ACL_2024/Minimum_Change/useless', task_name = 'error_correction', extract_gold_label_as_gt = True, simple_evaluation = True)
    
    if accuracy == 1:
        total_correct += 1

print(total_correct/len(data_list))
a = 1