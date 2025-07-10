import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
import copy
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import HOME_DIRECTORY, MODEL_DIRECTORY
from utils.in_context_perplexity_measurement_function import calculate_perplexity
from utils.in_context_data_loader import perplexity_calculation_in_context_data_loader
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import argparse

parser = argparse.ArgumentParser(description='train and evaluate')
parser.add_argument('--num_of_incontext_examples', type=int, required=True, help='Training method')
parser.add_argument('--use_plan_prompt', type=lambda x: (str(x).lower() == 'true'), default=False, required=False, help='Training method')
parser.add_argument('--model_name', type=str, required=True, help='model name')

args = parser.parse_args()

use_correct_initial_prediction = True
load_original_question = True
not_cap_perplexity = True


# train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification', 'plan_bench_replaning']

train_task_list = ['plan_bench_replaning', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification', 'gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']

train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']

train_task_list = ['plan_bench_replaning', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification']



num_of_incontext_examples = args.num_of_incontext_examples
use_plan_prompt = args.use_plan_prompt
model_name = args.model_name
if use_plan_prompt:
    suffix_ = f'_{num_of_incontext_examples}_examples_use_plan_prompt'
    train_task_list = ['plan_bench_replaning', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification']
else:
    suffix_ = f'_{num_of_incontext_examples}_examples'


test_idx = -1

index_range_list = [(0, 50), (50, 100), (100, 150), (0, 10), (10, 20), (20, 30), (0,30), (30, 60), (60, 90), (0, 100), (0, 200), (0, 300), (100, 200), (200, 300)]
n_train = 300

end_template = """
Now please solve the following question using the same inference style and format as the examples above. 
Question: """


def load_model(model_name):
    model_base = None
    if 'mistral' in model_name:
        model_path = f"{MODEL_DIRECTORY}/Mistral-7b-Instruct-v2"
    elif 'llama_3_instruct' in model_name:
        model_path = f"{MODEL_DIRECTORY}/Meta-Llama-3-8B-Instruct"
    elif 'phi_4' in model_name or 'phi-4' in model_name:
        model_path = f"{MODEL_DIRECTORY}/Phi-4"
    elif 'qwen' in model_name:
        model_path = f'{MODEL_DIRECTORY}/Qwen2.5-7B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for reduced memory usage
        device_map="auto"  # Automatic device mapping for optimal performance
    )
    model.to(device)
    return model, tokenizer, model_base

# for model_name in [model_name]:
for model_name in ['mistral', 'llama_3_instruct', 'qwen']:
# for model_name in ['mistral']:
# for model_name in ['llama_3_instruct']:
# for model_name in ['qwen']:
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

    model, tokenizer, model_base = load_model(model_name)
    
    for train_task_name in train_task_list:
        if 'plan' in train_task_name:
            file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/{train_task_name}_{model_name}_initial_prediction_{500}_plan_bench.json'
        else:
            file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/{train_task_name}_{model_name}_initial_prediction_{500}.json'
        
        with open(file_path_temp, 'r') as json_file:
            initial_prediction_list_total = json.load(json_file)
        
        print()
        print()
        print(f'----------------------------------------------------------{train_task_name}----------------------------------------------------------')

        dataset_list, _, _, _, _ = perplexity_calculation_in_context_data_loader(train_task_name, n_train, False, test_idx, end_template, correct_index_list = [])

        
        max_cccc = len(initial_prediction_list_total['initial_prediction']) - 1
        num_of_incontext_examples_temp = copy.deepcopy(num_of_incontext_examples)
        if num_of_incontext_examples > max_cccc:
            num_of_incontext_examples_temp = max_cccc

        for iii in range(len(dataset_list)):
            index_1 = 0
            index_2 = 1
            index_3 = 2
            index_4 = 3
            index_5 = 4
            for kkkk in range(len(dataset_list[iii][1])):
                index_1 += 1
                index_2 += 1
                index_3 += 1
                index_4 += 1
                index_5 += 1
                if index_1 == max_cccc or max_cccc < 2:
                    index_1 = 0
                if index_2 == max_cccc or max_cccc < 2:
                    index_2 = 0
                if index_3 == max_cccc or max_cccc < 2:
                    index_3 = 0
                if index_4 == max_cccc or max_cccc < 2:
                    index_4 = 0
                if index_5 == max_cccc or max_cccc < 2:
                    index_5 = 0
                original_question = dataset_list[iii][1][kkkk]['question']

                if num_of_incontext_examples_temp > 0:
                    initial_prediction_of_another_question_1 = initial_prediction_list_total['initial_prediction'][index_1]
                if num_of_incontext_examples_temp > 1:
                    initial_prediction_of_another_question_2 = initial_prediction_list_total['initial_prediction'][index_2]
                if num_of_incontext_examples_temp > 2:
                    initial_prediction_of_another_question_3 = initial_prediction_list_total['initial_prediction'][index_3]
                if num_of_incontext_examples_temp > 3:
                    initial_prediction_of_another_question_4 = initial_prediction_list_total['initial_prediction'][index_4]
                if num_of_incontext_examples_temp > 4:
                    initial_prediction_of_another_question_5 = initial_prediction_list_total['initial_prediction'][index_5]

                
                if num_of_incontext_examples_temp == 3:
                    in_context_question = \
f"""Question: {original_question}

We have inference examples below to show you how to solve the problem. please follow the inference style and solve the problem

inference example: {initial_prediction_of_another_question_1}

inference example: {initial_prediction_of_another_question_2}

inference example: {initial_prediction_of_another_question_3}

now, according to the inference examples, please solve the problem. 


"""
                elif num_of_incontext_examples_temp == 2:
                    in_context_question = \
f"""Question: {original_question}

We have inference examples below to show you how to solve the problem. please follow the inference style and solve the problem

inference example: {initial_prediction_of_another_question_1}

inference example: {initial_prediction_of_another_question_2}

now, according to the inference examples, please solve the problem. 


"""
                elif num_of_incontext_examples_temp == 1:
                    in_context_question = \
f"""Question: {original_question}

We have inference examples below to show you how to solve the problem. please follow the inference style and solve the problem

inference example: {initial_prediction_of_another_question_1}

now, according to the inference examples, please solve the problem. 


"""
                elif num_of_incontext_examples_temp == 4:
                    in_context_question = \
f"""Question: {original_question}

We have inference examples below to show you how to solve the problem. please follow the inference style and solve the problem

inference example: {initial_prediction_of_another_question_1}

inference example: {initial_prediction_of_another_question_2}

inference example: {initial_prediction_of_another_question_3}

inference example: {initial_prediction_of_another_question_4}

now, according to the inference examples, please solve the problem. 


"""
                elif num_of_incontext_examples_temp == 5:
                    in_context_question = \
f"""Question: {original_question}

We have inference examples below to show you how to solve the problem. please follow the inference style and solve the problem

inference example: {initial_prediction_of_another_question_1}

inference example: {initial_prediction_of_another_question_2}

inference example: {initial_prediction_of_another_question_3}

inference example: {initial_prediction_of_another_question_4}

inference example: {initial_prediction_of_another_question_5}

now, according to the inference examples, please solve the problem. 


"""
                dataset_list[iii][1][kkkk]['question'] = in_context_question
                answer = dataset_list[iii][1][kkkk]['answer']
                dataset_list[iii][1][kkkk]['answer'] = answer

        ppl_dict = {}
        data_name_list = []
        ii = 0
        for data_name, data_list, original_file_path, origianl_data_list, suffix in dataset_list:
            ii += 1
            print()
            print(f'----------------------------------------------------------{data_name}----------------------------------------------------------')

            perplexity_list = calculate_perplexity(data_list, model, tokenizer, model_name, device='cuda')
            ppl_dict[data_name] = perplexity_list
            data_name_list.append(data_name)

        # AFTER CHECK, I FOUND CASE 1 AND CASE 2 LOAD THE SAME ppl_dict for the first 50 ppl record
        icppl_dict_path = f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_icppl_{model_name}_{train_task_name}_icppl_dict{suffix_}_remove_correctness_filter.json"
        with open(icppl_dict_path, 'w') as f:
            json.dump(ppl_dict, f, indent=4) 

        with open(icppl_dict_path, 'r') as json_file:
            ppl_dict = json.load(json_file)
        
        for initial_index, last_index in index_range_list:
            record_book = {}
            for data_name in data_name_list:
                perplexity_list_temp = copy.deepcopy(ppl_dict[data_name])
                perplexity_list_temp = perplexity_list_temp[initial_index:last_index]

                average_perplexity = sum(perplexity_list_temp) / len(perplexity_list_temp) if perplexity_list_temp else float('inf')

                key = f'{train_task_name}_{data_name}'
                print(f'self aligned perplexity   {train_task_name} {data_name}: {average_perplexity}')
                record_book[key] = average_perplexity

                with open(f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_icppl_{model_name}_{train_task_name}_{data_name}{suffix_}_remove_correctness_filter.json", 'w') as f:
                    json.dump(record_book, f, indent=4)

            if 'plan_bench' in train_task_name:
                with open(f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_icppl_{model_name}_{train_task_name}_{initial_index}_{last_index}_plan_bench{suffix_}_remove_correctness_filter.json", 'w') as f:
                    json.dump(record_book, f, indent=4)
            else:
                with open(f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_icppl_{model_name}_{train_task_name}_{initial_index}_{last_index}_main{suffix_}_remove_correctness_filter.json", 'w') as f:
                    json.dump(record_book, f, indent=4)            

    import gc
    del model
    del tokenizer
    del model_base

    # Trigger garbage collection
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()

