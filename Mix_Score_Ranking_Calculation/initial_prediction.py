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


train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification', 'plan_bench_replaning']


train_task_list = ['plan_bench_replaning', 'plan_bench_verification', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution']


train_task_list = ['api_bank']


n_train = 500#00


test_idx = -1




end_template = """
Now please solve the following question using the same inference style and format as the examples above. 
Question: """


from evaluation.eval import do_predict_llama_factory_unify
from config.modify_config_on_current_job import set_config
output_folder_name_temp = f'kkk'
LLAMA_FACTORY_DIRECTORY_new = f"/gpfs/users/a1796450/llama_factory_temp/delete_later/perplexity_calculation_in_context__"
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

# for model_name in [model_name]:
# for model_name in ['mistral', 'llama_3_instruct', 'qwen']:
# for model_name in ['mistral', 'llama_3_instruct']:
# for model_name in ['llama_3_instruct']:
# for model_name in ['mistral']:
# for model_name in ['qwen', 'mistral', 'llama_3_instruct']:
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


    for train_task_name in train_task_list:
        dataset_list = load_original_data(train_task_name)
        if 'plan_bench_execution' in train_task_name:
            for i_, iitem in enumerate(dataset_list):
                q = iitem['question'] 
                # print(q)
                # The target string to replace
                old_text = "\n\n[STATEMENT]"
                ccc = q.count('[STATEMENT]')

                # The replacement string
                new_text = "\n\nThe statement, action sequence and the resulting state above only help you to understand the task. now please solve the following problem. Please notice that the problem below is the problem you need to solve. Please solve the problem below.\n\n[STATEMENT]"

                # Replace only the second occurrence
                count = 0  # To track occurrences
                def replace_second_occurrence(match):
                    global count
                    count += 1
                    return new_text if count == ccc else match.group(0)

                # Use regex to replace the second occurrence
                q = re.sub(re.escape(old_text), replace_second_occurrence, q)

                q = q.replace('[STATEMENT]', '[STATEMENT]')
                q = q.replace('[PLAN END]', '[PLAN END]\n')
                
                q += "\n\nplease inference first then place the resulting state at the end"
                dataset_list[i_]['question'] = q
        elif 'plan_bench_verification' in train_task_name:
            for i_, iitem in enumerate(dataset_list):
                q = iitem['question'] 
                # The target string to replace
                old_text = "[STATEMENT]"

                ccc = q.count('[STATEMENT]')

                # The replacement string
                new_text = "\n\nThe statement, verification and the plan above only help you to understand the task. now please solve the following problem. Please notice that the problem below is the problem you need to solve. Please solve the problem below.\n\n[STATEMENT]"

                # Replace only the second occurrence
                count = 0  # To track occurrences
                def replace_second_occurrence(match):
                    global count
                    count += 1
                    return new_text if count == ccc else match.group(0)

                # Use regex to replace the second occurrence
                q = re.sub(re.escape(old_text), replace_second_occurrence, q)

                q = q.replace('[STATEMENT]', '[STATEMENT]')
                q = q.replace('[PLAN END]', '[PLAN END]\n')

                # print(q)

                q += "\n\nplease inference first then place the verification result at the end"
                
                dataset_list[i_]['question'] = q
        
        elif 'plan_bench' in train_task_name:
            for i_, iitem in enumerate(dataset_list):
                q = iitem['question'] 
                # print(q)
                # The target string to replace
                old_text = "\n\n[STATEMENT]"
                ccc = q.count('[STATEMENT]')

                # The replacement string
                new_text = "\n\nThe statement and the plan above only help you to understand the task. now please solve the following problem. Please notice that the problem below is the problem you need to solve. Please solve the problem below.\n\n[STATEMENT]"

                # Replace only the second occurrence
                count = 0  # To track occurrences
                def replace_second_occurrence(match):
                    global count
                    count += 1
                    return new_text if count == ccc else match.group(0)

                # Use regex to replace the second occurrence
                q = re.sub(re.escape(old_text), replace_second_occurrence, q)

                # q = q.replace('[STATEMENT]', '[STATEMENT]')
                # q = q.replace('[PLAN END]', '[PLAN END]\n')
                # print(q)

                q += "\n\nplease inference first then place the plan at the end"

                dataset_list[i_]['question'] = q

        
        elif 'api_bank' in train_task_name:# and 'qwen' in model_name:
            for i_, iitem in enumerate(dataset_list):
                q = iitem['question'] 
                q = q.replace('Generate API Request:', '')
                q = q.replace('\nGenerate an API request ', 'Generate an API request ')

                q += 'What should be the next API Request?'
                q = 'given the available information, please answer the question, then ' + q
                


                # when asking qwen question, it tends to directly output labels rather than giving any explaination or providing any inference. We believe that might because the in-context example in api_bank datasets leads qwen to answer in that way. Therefore, we use this prompt to encourage qwen thinking before providing the api call.
                # if 'qwen' in model_name:
                #     q += "\n\nplease inference first then generate the API-Request at the end"

                dataset_list[i_]['question'] = q

        dataset_list = dataset_list[:n_train]
        test_task_name = train_task_name
        
        initial_prediction_dict = {}

        data_list = dataset_list      
        origianl_data_list = data_list

        print(f'----------------------------------------------------------initial prediction creation----------------------------------------------------------')

        file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/{train_task_name}_{model_name}_initial_prediction_{n_train}.json'

        if 'plan_bench' in train_task_name:
            file_path_temp = file_path_temp.replace('.json', '_plan_bench.json')

        if 'api_bank' in train_task_name:
            file_path_temp = file_path_temp.replace('.json', '_api_bank.json')
        
        
        train_config, test_config = set_config(test_task_name.lower(), device_num, seed_num, model_name = model_name)
        data_name_temp = test_task_name.lower() + '_full_zeroshot'


        predict_list = do_predict_llama_factory_unify(origianl_data_list, output_folder_name_temp, test_config, 'xxx', check_point_folder_name = '', merged_base_model_dir = '', data_name = data_name_temp, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)
        initial_prediction_dict['initial_prediction'] = predict_list
        with open(file_path_temp, 'w') as json_file:
            json.dump(initial_prediction_dict, json_file, indent=4)

        with open(file_path_temp, 'r') as json_file:
            initial_prediction_list = json.load(json_file)
        initial_prediction_list['initial_prediction'] = initial_prediction_list['initial_prediction'][:n_train]
        train_config, test_config = set_config(test_task_name.lower(), device_num, seed_num, model_name = 'qwen')
        gold_label_list, groundtruth_list, question_list = load_gold_label_and_question_list(train_task_name, n_train)
        initial_prediction_correct_example_dict = Find_correct_initial_prediction(test_task_name, initial_prediction_list, dataset_list, gold_label_list, output_folder_name_temp, test_config,  LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)

        if 'plan_bench' in train_task_name:
            file_path_correct_examples = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_correct_examples/{train_task_name}_{model_name}_{n_train}_plan_bench.json'
        elif 'api_bank' in train_task_name:
            file_path_correct_examples = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_correct_examples/{train_task_name}_{model_name}_{n_train}_api_bank.json'
        else:
            file_path_correct_examples = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_correct_examples/{train_task_name}_{model_name}_{n_train}.json'
        with open(file_path_correct_examples, 'w') as json_file:
            json.dump(initial_prediction_correct_example_dict, json_file, indent=4)
            


