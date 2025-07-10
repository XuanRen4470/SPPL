import numpy as np
import torch
import json
import sys
import os
import re

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.data_loader import load_gold_label_and_question_list
from evaluation.eval import Find_correct_initial_prediction
from config.config import LLAMA_FACTORY_DIRECTORY, HOME_DIRECTORY
from utils.in_context_data_loader import load_original_data
import argparse
import shutil
import time



os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

parser = argparse.ArgumentParser(description='train and evaluate')
parser.add_argument('--train_task_name', type=str, required=True, help='model name')

args = parser.parse_args()

train_task_name = args.train_task_name




n_train = 1000#00

from evaluation.eval import do_predict_llama_factory_unify
from config.modify_config_on_current_job import set_config
output_folder_name_temp = f'kkk'
LLAMA_FACTORY_DIRECTORY_new = f"/gpfs/users/a1796450/llama_factory_temp/delete_later/perplexity_calculation_in_context____"
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

for model_name in ['qwen']:
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
    if 'arc_challenge' in train_task_name:
        end_template = "please inference first then place the final answer(A/B/C/D) after the word 'Final Answer: ' at the end."
    if 'piqa' in train_task_name:
        end_template = "please inference first then place the final answer(1/2) after the word 'Final Answer: ' at the end."
    
    if 'ecqa' in train_task_name:
        end_template = "please inference first then place the final answer(1/2/3/4/5) after the word 'Final Answer: ' at the end."

    if 'agieval' in train_task_name:
        end_template = "please inference first then place the final answer(A/B/C/D) after the word 'Final Answer: ' at the end."

    if 'squad' in train_task_name:
        end_template = "please inference first then place the final answer(a text span) after the word 'Final Answer: ' at the end."


    for i_, iitem in enumerate(dataset_list):
        q = iitem['question'] + end_template
        dataset_list[i_]['question'] = q

    dataset_list = dataset_list[:n_train]
    test_task_name = train_task_name
    
    initial_prediction_dict = {}

    data_list = dataset_list      
    origianl_data_list = data_list

    print(f'----------------------------------------------------------initial prediction creation----------------------------------------------------------')

    file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/zero_shot_{train_task_name}_{model_name}_initial_prediction_{n_train}.json'

    train_config, test_config = set_config(test_task_name.lower(), device_num, seed_num, model_name = model_name)
    data_name_temp = test_task_name.lower() + '_full_zeroshot'
    
    predict_list = do_predict_llama_factory_unify(origianl_data_list, output_folder_name_temp, test_config, 'xxx', check_point_folder_name = '', merged_base_model_dir = '', data_name = data_name_temp, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)
    initial_prediction_dict['initial_prediction'] = predict_list
    with open(file_path_temp, 'w') as json_file:
        json.dump(initial_prediction_dict, json_file, indent=4)

