import numpy as np
import torch
import json
import sys
import os
import re

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import LLAMA_FACTORY_DIRECTORY, HOME_DIRECTORY, LLAMA_FACTORY_TEMP_DIRECTORY
from utils.in_context_data_loader import load_original_data
import argparse
import shutil
import time



os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name_list',
    type=str,
    nargs='+',   # 接收多个值
    required=False,
    default=['qwen', 'mistral', 'llama_3_instruct'],
    help='model name list'
)

args = parser.parse_args()
model_name_list = args.model_name_list
print(model_name_list)



train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification', 'plan_bench_replaning']

n_train = 1000#00

from evaluation.eval import do_predict_llama_factory_unify
from config.modify_config_on_current_job import set_config
output_folder_name_temp = f'kkk'
llama_factory_name_suffix = '_'.join(model_name_list)

LLAMA_FACTORY_DIRECTORY_new = f"{LLAMA_FACTORY_TEMP_DIRECTORY}/perplexity_calculation_in_context_{llama_factory_name_suffix}"
device_num = 1
seed_num = 0
# Check if the destination directory exists, and if so, remove it
if os.path.exists(LLAMA_FACTORY_DIRECTORY_new):
    shutil.rmtree(LLAMA_FACTORY_DIRECTORY_new)
    print(f"Existing directory {LLAMA_FACTORY_DIRECTORY_new} removed")
    time.sleep(10)
# Copy the directory
try:
    shutil.copytree(LLAMA_FACTORY_DIRECTORY, LLAMA_FACTORY_DIRECTORY_new)
except:
    time.sleep(10)
    shutil.copytree(LLAMA_FACTORY_DIRECTORY, LLAMA_FACTORY_DIRECTORY_new)
print(f"Directory copied successfully to {LLAMA_FACTORY_DIRECTORY_new}")
time.sleep(2)


for train_task_name in train_task_list:
    for model_name in model_name_list:
        if 'mistral' in model_name:
            model_name = 'mistral'
        elif 'llama_3_instruct' in model_name:
            model_name = 'llama_3_instruct'
        elif 'phi_4' in model_name:
            model_name = 'phi_4'
        elif 'qwen' in model_name:
            model_name = 'qwen'

        # Check if GPU is available and set the device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




        dataset_list = load_original_data(train_task_name)
        if 'math' in train_task_name or 'gsm8k' in train_task_name:
            end_template = "please inference first then place the final neumerical answer after the word 'Final Answer: ' at the end."
        elif 'arc_challenge' in train_task_name:
            end_template = "please inference first then place the final answer(A/B/C/D) after the word 'Final Answer: ' at the end."
        elif 'piqa' in train_task_name:
            end_template = "please inference first then place the final answer(1/2) after the word 'Final Answer: ' at the end."
        elif 'ecqa' in train_task_name:
            end_template = "please inference first then place the final answer(1/2/3/4/5) after the word 'Final Answer: ' at the end."
        elif 'agieval' in train_task_name:
            end_template = "please inference first then place the final answer(A/B/C/D) after the word 'Final Answer: ' at the end."
        elif 'squad' in train_task_name:
            end_template = "please inference first then place the final answer(a text span) after the word 'Final Answer: ' at the end."


        elif 'boolq' in train_task_name:
            end_template = "Please inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'"
        elif 'mmlu_pro' in train_task_name:
            end_template = "Please inference first, then provide the final answer (A/B/C/D/E/F/G/H/I/J) at the end, after 'Final Answer:'"
        elif 'mmlu' in train_task_name:
            end_template = "Please inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'"
        

        elif 'winogrande' in train_task_name:
            end_template = "Please inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'"
        elif 'drop' in train_task_name:
            end_template = "Please inference first, then provide the final answer at the end, after 'Final Answer:'"
        elif 'agieval' in train_task_name:
            end_template = "Please inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'"
        elif 'api_bank' in train_task_name:
            end_template = "\nPlease inference first then provide the API-Request at the end after the word 'Final Answer:'"

        elif 'mbpp' in train_task_name:
            end_template = ""

        elif 'hellaswag':
            end_template = "\nPlease inference first, then provide the final answer (1/2/3/4) at the end, after 'Final Answer:'"

        elif 'plan_bench_verification' in test_task_name.lower():
            end_template = "\n\nPlease inference first then check if the plan is valid follow by an explaination at the end after the word 'Final Answer:'"
        elif 'plan_bench_execution' in test_task_name.lower():
            end_template = "\n\nplease inference first then put the resulting state at the end after 'Final Answer:'"
        elif 'plan_bench' in test_task_name.lower():
            end_template = "\n\nPlease infer first, then place the plan at the end, after 'Final Answer:'. The plan you place after 'Final Answer:' should be written in triplet format, which contains (action, object_1, object_2). For example, (unstack red blue) means that you unstack the red object from the blue object."


        


        for i_, iitem in enumerate(dataset_list):
            q = iitem['question'] + '\n' + end_template
            dataset_list[i_]['question'] = q

        dataset_list = dataset_list[:n_train]
        test_task_name = train_task_name
        
        initial_prediction_dict = {}

        data_list = dataset_list      
        origianl_data_list = data_list

        print(f'----------------------------------------------------------initial prediction creation----------------------------------------------------------')
        file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/zero_shot_prediction/zero_shot_{train_task_name}_{model_name}_initial_prediction_{n_train}.json'

        train_config, test_config = set_config(test_task_name.lower(), device_num, seed_num, model_name = model_name)
        data_name_temp = test_task_name.lower() + '_full_zeroshot'
        
        predict_list = do_predict_llama_factory_unify(origianl_data_list, output_folder_name_temp, test_config, 'xxx', check_point_folder_name = '', merged_base_model_dir = '', data_name = data_name_temp, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)
        initial_prediction_dict['initial_prediction'] = predict_list
        with open(file_path_temp, 'w') as json_file:
            json.dump(initial_prediction_dict, json_file, indent=4)

