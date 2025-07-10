import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import json
import sys
import os
import re

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.data_loader import load_gold_label_and_question_list, add_gold_label
from evaluation.eval import Check_Correctness, Find_correct_initial_prediction
from config.config import LLAMA_FACTORY_DIRECTORY, HOME_DIRECTORY, MODEL_DIRECTORY
from utils.in_context_perplexity_measurement_function import compute_similarity_scores, compute_diversity_pca_scores, compute_diversity_tsne_scores, compute_complexity_scores, perplexity_calculation, original_perplexity_calculation, in_context_perplexity_calculation, assumulated_perplexity_calculation, kl_calculation, find_similar_examples, compute_similarity_scores_question_attached, customized_perplexity_calculation
from utils.in_context_data_loader import perplexity_calculation_in_context_data_loader, perplexity_calculation_in_context_total_data_loader, load_original_data, initial_prediction_in_context_data_loader
import argparse
import shutil
import time


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


parser = argparse.ArgumentParser(description='train and evaluate')
parser.add_argument('--model_type', type=str, required=True, help='Training method')
parser.add_argument('--debug_mode', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--use_in_context_learning', type=lambda x: (str(x).lower() == 'true'), default=True, help='')
parser.add_argument('--similarity_compare_to_initial_prediction_with_in_context_example', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--similarity_compare_to_irrelevant_prediction', type=lambda x: (str(x).lower() == 'true'), default=False, help='use mc as answers and keep in context examples unchange')
parser.add_argument('--create_initial_prediction', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--hidden_states_layer_num', type=int, required=False, default=-1, help='')
parser.add_argument('--use_total', type=lambda x: (str(x).lower() == 'true'), default=True, help='')
parser.add_argument('--enable_skywork_reward', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--use_correct_initial_prediction', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--load_original_question', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--Find_similar_examples', type=lambda x: (str(x).lower() == 'true'), default=False, help='', required=False)
parser.add_argument('--record_token_length', type=lambda x: (str(x).lower() == 'true'), default=False, help='', required=False)

args = parser.parse_args()
model_name = args.model_type
debug_mode = args.debug_mode
use_total = args.use_total
use_in_context_learning = args.use_in_context_learning
similarity_compare_to_initial_prediction_with_in_context_example = args.similarity_compare_to_initial_prediction_with_in_context_example
similarity_compare_to_irrelevant_prediction = args.similarity_compare_to_irrelevant_prediction
create_initial_prediction = args.create_initial_prediction
hidden_states_layer_num = args.hidden_states_layer_num
enable_skywork_reward = args.enable_skywork_reward
use_correct_initial_prediction = args.use_correct_initial_prediction
load_original_question = args.load_original_question
Find_similar_examples = args.Find_similar_examples
record_token_length = args.record_token_length

not_cap_perplexity = True
calc_IDF = True
CAR_beta = 3
n_similar_self_generated_examples = 2

# train_task_list = ['piqa', 'mmlu', 'winogrande', 'agieval', 'squad', 'gsm8k', 'math_algebra', 'ecqa', 'boolq', 'api_bank', 'mmlu_pro', 'hellaswag', 'arc_challenge', 'drop']#, 'esnli', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'math_geometry', 'math_intermediate_algebra']
# train_task_list = ['gsm8k', 'math_algebra']
# train_task_list = ['drop']


# train_task_list = ['piqa', 'mmlu', 'winogrande', 'agieval', 'squad', 'gsm8k', 'math_algebra', 'ecqa', 'boolq', 'api_bank', 'mmlu_pro', 'hellaswag', 'arc_challenge', 'drop', 'api_bank']#, 'plan_bench_generalization']




# train_task_list = ['arc_challenge', 'drop', 'piqa', 'mmlu', 'winogrande', 'agieval', 'squad', 'ecqa', 'esnli', 'boolq', 'mmlu_pro', 'hellaswag']#, 'gsm8k', 'math_algebra']

# train_task_list = ['math_algebra']
# train_task_list = ['api_bank']
# train_task_list = ['mmlu_pro']

# train_task_list = ['esnli', 'mmlu_pro_law', 'math_intermediate_algebra']

# train_task_list = ['mmlu_pro', 'hellaswag', 'arc_challenge', 'math_geometry', 'math_intermediate_algebra']
# train_task_list = ['plan_bench_generation']
# train_task_list = ['math_geometry', 'math_intermediate_algebra']

# train_task_list = ['mbpp']
# train_task_list = ['api_bank', 'gsm8k', 'math_algebra']
train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']#, 'plan_bench_generalization']#, 'esnli']    
# train_task_list = ['math_geometry', 'api_bank']#, 'plan_bench_generalization']#, 'esnli']    



# train_task_list = ['api_bank', 'plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality']
# train_task_list = ['plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality']#, 'api_bank']
# train_task_list = ['math_algebra']
# train_task_list = ['api_bank']
# train_task_list = ['mbpp']
# train_task_list = ['ecqa']
# train_task_list = ['plan_bench_execution', 'plan_bench_generalization']
train_task_list = ['plan_bench_optimality', 'plan_bench_generation']

train_task_list = ['plan_bench_reuse', 'plan_bench_replaning', 'plan_bench_generalization', 'plan_bench_verification']
# train_task_list = ['mmlu_moral_scenarios']
# train_task_list = ['math_geometry']

train_task_list = ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_verification']
train_task_list = ['plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_verification']

train_task_list = ['plan_bench_replaning']


n_train = 500#00
index_range = [(0, 4)]
test_idx = -1


end_template = """
Now please solve the following question using the same inference style and format as the examples above. 
Question: """


def load_model(model_name):
    model_base = None
    if 'mistral' in model_name:
        model_path = f"{MODEL_DIRECTORY}/Mistral-7b-Instruct-v2"
    # elif 'llama_3_instruct' in model_name:
    #     model_path = f"{MODEL_DIRECTORY}/Meta-Llama-3-8B-Instruct"
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

def skywork_reward_calculation(question, response, rm, rm_tokenizer):
    conv1 = [{"role": "user", "content": question}, {"role": "assistant", "content": response}]

    conv1_tokenized = rm_tokenizer.apply_chat_template(conv1, tokenize=True, return_tensors="pt").to(device)

    # Get the reward scores
    with torch.no_grad():
        score = rm(conv1_tokenized).logits[0][0].item()
    return score

def escape_dollar_signs(text):
    return text.replace('$', '\\$')

if create_initial_prediction:
    from evaluation.eval import do_predict_llama_factory_unify
    from config.modify_config_on_current_job import set_config
    output_folder_name_temp = f'kkk'
    LLAMA_FACTORY_DIRECTORY_new = f"/gpfs/users/a1796450/llama_factory_temp/delete_later/perplexity_calculation_in_context_{model_name}"
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
for model_name in ['mistral', 'llama_3_instruct', 'qwen']:
# for model_name in ['qwen', 'llama_3_instruct']:
# for model_name in ['mistral']:
# for model_name in ['llama_3_instruct']:
# for model_name in ['qwen']:
# for model_name in ['mistral', 'llama_3_instruct']:
    suffix = ''
    if debug_mode:
        suffix = '_debug'
    file_name = ''
    if use_in_context_learning:
        file_name = f'in_context_'

    if similarity_compare_to_irrelevant_prediction:
        file_name += f'irrelevant_'

    if 'mistral' in model_name:
        model_name = 'mistral'
    elif 'llama_3_instruct' in model_name:
        model_name = 'llama_3_instruct'
    elif 'phi_4' in model_name:
        model_name = 'phi_4'
    elif 'qwen' in model_name:
        model_name = 'qwen'

    if 'mistral' in model_name:
        file_name += f'mistral_perplexity_record{suffix}'
    elif 'llama_3_instruct' in model_name:
        file_name += f'llama_3_instruct_perplexity_record{suffix}'
    elif 'phi_4' in model_name or 'phi-4' in model_name:
        file_name += f'phi_4_instruct_perplexity_record{suffix}'
    elif 'qwen' in model_name:
        file_name += f'qwen_perplexity_record{suffix}'
    else:
        file_name += f'unkown_model_perplexity_record{suffix}'

    if similarity_compare_to_initial_prediction_with_in_context_example:
        file_name += '_similarity_compare_to_initial_prediction_with_in_context_example'

    if hidden_states_layer_num != -1:
        file_name = f'layer_{hidden_states_layer_num}_' + file_name
    else:
        file_name = f'layer_all_' + file_name
    
    if use_correct_initial_prediction:
        file_name += '_use_correct_initial_prediction'

    

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not_cap_perplexity:
        suffix__ = f'not_cap_perplexity'
    else:
        suffix__ = f''  
    
    for initial_index, last_index in index_range:
        skywork_reward_score_book = {}
        for train_task_name in train_task_list:
            if not load_original_question:
                dataset_list, _, _, test_task_name, _ = perplexity_calculation_in_context_data_loader(train_task_name, n_train, use_in_context_learning, test_idx, end_template)
            else:
                dataset_list = load_original_data(train_task_name)
                if 'plan' in train_task_name and 'llama_3' in model_name:
                    for i_, iitem in enumerate(dataset_list):
                        q = iitem['question'] 
                        # print(q)
                        # The target string to replace
                        old_text = "\n\n[STATEMENT]"

                        # The replacement string
                        new_text = "\n\nThe statement and the plan above only help you to understand the task. now please solve the following problem. Please notice that the problem below is the problem you need to solve. Please solve the problem below.\n\n[STATEMENT]"

                        # Replace only the second occurrence
                        count = 0  # To track occurrences
                        def replace_second_occurrence(match):
                            global count
                            count += 1
                            return new_text if count == 2 else match.group(0)

                        # Use regex to replace the second occurrence
                        q = re.sub(re.escape(old_text), replace_second_occurrence, q)

                        q = q.replace('[STATEMENT]', 'This is an example. [STATEMENT]')
                        q = q.replace('[PLAN END]', '[PLAN END] The example end here.\n')
                        
                        # q = q + '\n\nPlease inference first then place the final answer at the end after Final Answer:'
                        dataset_list[i_]['question'] = q
                
                elif 'api_bank' in train_task_name:# and 'qwen' in model_name:
                    for i_, iitem in enumerate(dataset_list):
                        q = iitem['question'] 
                        q = q.replace('Generate API Request:', '')
                        q = q.replace('\nGenerate an API request ', 'Generate an API request ')
                        # q += '\nAccording to the available APIs, the previous API calling history(if available) and the user\'s utterance and the goal, please inference first then Generate API Request.'

                        # march 26 backup
                        # q += 'please inference first then Generate API Request'
                        # q = 'given the available information, please inference first, then ' + q

                        q += 'What should be the next API Request?'
                        q = 'given the available information, please answer the question, then ' + q

                        dataset_list[i_]['question'] = q

                dataset_list = dataset_list[:n_train]
                test_task_name = train_task_name
            if use_total:
                for api_type in ['anthropic', 'mini_gpt4', 'gpt4']:
                    for prompt_api in ['gpt4']:
                        for total_use_simple_structure in [True, False]:
                            dataset_list_total, _ = perplexity_calculation_in_context_total_data_loader(train_task_name, n_train, use_in_context_learning, test_idx, end_template, api_type = api_type, prompt_api=prompt_api, total_use_simple_structure = total_use_simple_structure)
                            dataset_list = dataset_list + dataset_list_total
        
            if similarity_compare_to_irrelevant_prediction:
                dataset_list = [dataset_list[-1]]
            
            initial_prediction_dict = {}

            if create_initial_prediction:
                if not load_original_question:
                    data_name, data_list, original_file_path, origianl_data_list, suffix = dataset_list[0]    
                else:
                    data_list = dataset_list      
                    origianl_data_list = data_list

                if create_initial_prediction:
                    print(f'----------------------------------------------------------initial prediction creation----------------------------------------------------------')
                    suffix_ = ''
                    # if use_correct_initial_prediction:
                    #     suffix_ += '_use_correct_initial_prediction'
                    # if similarity_compare_to_initial_prediction_with_in_context_example:
                    #     suffix_ += '_with_in_context_example'

                    if not load_original_question:
                        file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record/{train_task_name}_{model_name}_initial_prediction{suffix_}_{n_train}.json'
                    else:
                        file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/{train_task_name}_{model_name}_initial_prediction{suffix_}_{n_train}.json'

                    with open(file_path_temp, 'r') as json_file:
                        initial_prediction_list = json.load(json_file)

                    # n_train = 20
                    initial_prediction_list['initial_prediction'] = initial_prediction_list['initial_prediction'][:n_train]
                    dataset_list = dataset_list[:n_train]
                    train_config, test_config = set_config(test_task_name.lower(), device_num, seed_num, model_name = 'qwen')
                    gold_label_list, groundtruth_list, question_list = load_gold_label_and_question_list(train_task_name, n_train)
                    initial_prediction_correct_example_dict = Find_correct_initial_prediction(test_task_name, initial_prediction_list, dataset_list, gold_label_list, output_folder_name_temp, test_config,  LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)
                    file_path_correct_examples = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_correct_examples/{train_task_name}_{model_name}_{n_train}.json'
                    with open(file_path_correct_examples, 'w') as json_file:
                        json.dump(initial_prediction_correct_example_dict, json_file, indent=4)
                        
    index_range_list = [(0, 50)]
    for initial_index, last_index in index_range_list:
    # for initial_index, last_index in index_range:        
        skywork_reward_score_book = {}
        for train_task_name in train_task_list:
            record_book = {}
            layerwise_cos_similarity_record_book = {}
 
            print()
            print()
            print(f'----------------------------------------------------------{train_task_name}----------------------------------------------------------')

            suffix_ = ''
            if use_correct_initial_prediction and 'plan_bench' not in train_task_name:
                suffix_ += '_use_correct_initial_prediction'
            if similarity_compare_to_initial_prediction_with_in_context_example:
                suffix_ += '_with_in_context_example'
            if not load_original_question:
                file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record/{train_task_name}_{model_name}_initial_prediction{suffix_}_{n_train}.json'
            else:
                file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/{train_task_name}_{model_name}_initial_prediction{suffix_}_{n_train}.json'

            #############
            if use_correct_initial_prediction and load_original_question:# and 'plan_bench' not in train_task_name:
                file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_correct_examples/{train_task_name}_{model_name}_{n_train}.json'

            
            file_path_temp = file_path_temp.replace('.json', '_backup.json')

            with open(file_path_temp, 'r') as json_file:
                initial_prediction_list_total = json.load(json_file)
            if use_correct_initial_prediction:# and 'plan_bench' not in train_task_name:
                correct_index_list = initial_prediction_list_total['correct_index']
            else:
                correct_index_list = []
            

            dataset_list, train_config, test_config, test_task_name, gpt4_prediction_list = perplexity_calculation_in_context_data_loader(train_task_name, n_train, use_in_context_learning, test_idx, end_template, correct_index_list = correct_index_list)


            
            # file_path_temp_temp = file_path_temp.replace('.json', '_backup.json')
            # with open(file_path_temp_temp, 'w') as f:
            #     json.dump(initial_prediction_list_total, f, indent=4)


            for cccc in range(len(initial_prediction_list_total['initial_prediction'])):
                calc_index = correct_index_list[cccc]

                corresponding_initial_prediction = initial_prediction_list_total['initial_prediction'][cccc] 
                gold_label = dataset_list[0][3][cccc]['gold_label']

                if gold_label == '':
                    gold_label = '()'
                kkkkk = corresponding_initial_prediction + '\n\nFinal Answer: ' + gold_label
                initial_prediction_list_total['initial_prediction'][cccc] = kkkkk
            
            file_path_temp = file_path_temp.replace('_backup.json', '.json')
            with open(file_path_temp, 'w') as f:
                json.dump(initial_prediction_list_total, f, indent=4)