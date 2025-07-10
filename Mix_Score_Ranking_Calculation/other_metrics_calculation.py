from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
import pickle

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import HOME_DIRECTORY, MODEL_DIRECTORY
from utils.in_context_perplexity_measurement_function import original_perplexity_calculation, probability_in_context_perplexity_calculation
from utils.in_context_data_loader import perplexity_calculation_in_context_data_loader
import argparse

parser = argparse.ArgumentParser(description='train and evaluate')
parser.add_argument('--model_name', type=str, required=False, default='llama2-13b')
parser.add_argument('--suffix', type=str, default='', required=False, help='model_name')

args = parser.parse_args()
suffix_ = args.suffix

model_name = args.model_name


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

load_original_question = False

not_cap_perplexity = True
calc_IDF = True
CAR_beta = 3

# train_task_list = ['piqa', 'mmlu', 'winogrande', 'agieval', 'squad', 'gsm8k', 'math_algebra', 'ecqa', 'boolq', 'api_bank', 'mmlu_pro', 'hellaswag', 'arc_challenge', 'drop', 'math_geometry', 'mmlu_moral_scenarios', 'mmlu_pro_law']
train_task_list = ['api_bank' , 'gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality', 'plan_bench_reuse', 'plan_bench_verification', 'plan_bench_replaning', 'plan_bench_execution']
# train_task_list = ['plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality', 'plan_bench_reuse', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning']

test_idx = -1


# n_train = 500#00
n_train = 300#00

index_range_list = [(0, 50), (50, 100), (100, 150), (0, 10), (10, 20), (20, 30), (0,30), (30, 60), (60, 90), (0, 100), (0, 200), (0, 300), (100, 200), (200, 300)]

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

# for model_name in ['mistral', 'qwen', 'llama_3_instruct']:
# for model_name in ['llama_3_instruct']:
for model_name in [model_name]:
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
    
    skywork_reward_score_book = {}
    for train_task_name in train_task_list:
        print()
        print()
        print(f'----------------------------------------------------------{train_task_name}----------------------------------------------------------')

        correct_index_list = []

        dataset_list, train_config, test_config, test_task_name, gpt4_prediction_list = perplexity_calculation_in_context_data_loader(train_task_name, n_train, False, test_idx, end_template, correct_index_list = correct_index_list)

        skywork_path = f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/skywork_reward_record/{train_task_name}_{n_train}.pkl"

        with open(skywork_path, "rb") as f:
            skywork_dict = pickle.load(f)

        file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/{train_task_name}_{model_name}_initial_prediction_{500}.json'
        if 'plan' in train_task_name:
            file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_modified/{train_task_name}_{model_name}_initial_prediction_{500}_plan_bench.json'

        with open(file_path_temp, 'r') as json_file:
            initial_prediction_list_total = json.load(json_file)

        metrics_dict = {}
        data_name_list = []
        for data_name, data_list, original_file_path, origianl_data_list, _ in dataset_list:

            print()
            print(f'----------------------------------------------------------{data_name}----------------------------------------------------------')

            perplexity_list, IDF_list, loss_list, token_len_list = original_perplexity_calculation(data_list, model, tokenizer, model_name, device=device)
    
            _, seq_prob_list, log_seq_prob_list = probability_in_context_perplexity_calculation(data_list, model, tokenizer, model_name, device=device)

            metrics_dict[data_name] = {}
            metrics_dict[data_name]['ppl'] = perplexity_list
            metrics_dict[data_name]['idf'] = IDF_list
            metrics_dict[data_name]['loss'] = loss_list
            metrics_dict[data_name]['log_prob'] = log_seq_prob_list
            data_name_list.append(data_name)

        for initial_index, last_index in index_range_list:
            record_book = {}
            for data_name in data_name_list:
                sky_work_score_list = skywork_dict[data_name]
                perplexity_list = metrics_dict[data_name]['ppl'] 
                IDF_list = metrics_dict[data_name]['idf'] 
                loss_list = metrics_dict[data_name]['loss'] 
                log_seq_prob_list = metrics_dict[data_name]['log_prob'] 

                skywork_score_list_temp = sky_work_score_list[initial_index:last_index]
                perplexity_list_temp = perplexity_list[initial_index:last_index]
                IDF_list_temp = IDF_list[initial_index:last_index]
                loss_list_temp = loss_list[initial_index:last_index]
                seq_prob_list_temp = seq_prob_list[initial_index:last_index]
                log_seq_prob_list_temp = log_seq_prob_list[initial_index:last_index]

                skywork_reward_score = sum(skywork_score_list_temp)/len(skywork_score_list_temp)
                average_perplexity = sum(perplexity_list_temp) / len(perplexity_list_temp) if perplexity_list_temp else float('inf')
                average_IDF = sum(IDF_list_temp) / len(IDF_list_temp) if IDF_list_temp else float('inf')
                average_loss = sum(loss_list_temp) / len(loss_list_temp) if loss_list_temp else float('inf')
                avg_log_seq_prob = sum(log_seq_prob_list_temp) / len(log_seq_prob_list_temp)
                average_IDF_score = float(f"{average_IDF:.3g}")
                avg_log_seq_prob = float(f"{avg_log_seq_prob:.3g}")
                CAR_score = skywork_reward_score / (1 + CAR_beta * average_loss)
                CAR_score = float(f"{CAR_score:.3g}")
                
                key = f'{train_task_name}_{data_name}'
                element = {}
                element['perplexity'] = average_perplexity
                element['IDF_score'] = average_IDF_score
                element['log_propability'] = avg_log_seq_prob
                element['skywork_reward_score'] = skywork_reward_score
                element['CAR_score'] = CAR_score

                print(f'perplexity: {average_perplexity}')
                print(f'log_propability: {avg_log_seq_prob}')
                print(f'IDF: {average_IDF_score}')
                print(f"Skywork Reward Score: {skywork_reward_score}")
                print(f"CAR Score: {CAR_score}")

                record_book[key] = element

            # if load_original_question:
            #     with open(f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_other_metrics_correct_original_{model_name}_{train_task_name}_{data_name}_{suffix_}.json", 'w') as f:
            #         json.dump(record_book, f, indent=4)
            # else:
            #     with open(f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_other_metrics_{model_name}_{train_task_name}_{data_name}_{suffix_}.json", 'w') as f:
            #         json.dump(record_book, f, indent=4)
            
            with open(f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_other_metrics_{model_name}_{train_task_name}_{initial_index}_{last_index}_{suffix_}.json", 'w') as f:
                json.dump(record_book, f, indent=4)
            a = 1

    import gc
    del model
    del tokenizer
    del model_base

    # Trigger garbage collection
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()