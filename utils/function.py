import sys
import os
import json
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from config.config import HOME_DIRECTORY

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def record_json_data_to_file(output_folder_name, test_data_list, test_config, test_task_name):
    test_json_file = []
    for i in range(len(test_data_list)):
        temp = {}
        temp['question'] = test_data_list[i]['question']
        temp['answer'] = test_data_list[i]['answer']
        test_json_file.append(temp)
    destination_file = f"{HOME_DIRECTORY}/output/{output_folder_name}/{test_config['seed_num']}/{test_task_name}.jsonl"

    with open(destination_file, 'w') as json_file:
        json.dump(test_json_file, json_file, indent=4)


def extract_checkpoint_names(directory):
    # List to store checkpoint names
    checkpoint_names = []

    # Iterate over the files and directories in the given directory
    for item in os.listdir(directory):
        # Check if the item is a directory and starts with "checkpoint-"
        if os.path.isdir(os.path.join(directory, item)) and item.startswith("checkpoint-"):
            checkpoint_names.append(item)

    # Sort the checkpoint names by the number following "checkpoint-"
    checkpoint_names.sort(key=lambda name: int(name.split('-')[1]))

    return checkpoint_names



def load_experimental_result(model_name_list, task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = ''):
    variation_suffix_list_non_gpt4 = [""]
    variation_suffix_list_gpt4_gt_non_cot = ['', "variation_gpt4_style_in_context_examples", "variation_openai_human_written_examples", 'variation_step_by_step']#, 'variation_simple_response']#, 'variation_mistral_self_generated', 'variation_qwen_self_generated', 'variation_llama_3_instruct_self_generated']
    variation_suffix_list_gpt4_gt_cot = ['', "variation_gpt4_style_in_context_examples", "variation_openai_human_written_examples", 'variation_step_by_step', 'variation_rewrite_groundtruth_in_own_words']#, 'variation_simple_response']#, 'variation_mistral_self_generated', 'variation_qwen_self_generated', 'variation_llama_3_instruct_self_generated']

    # variation_suffix_list_gpt4_gt_non_cot = ['', "variation_gpt4_style_in_context_examples", 'variation_step_by_step']#, 'variation_simple_response']#, 'variation_mistral_self_generated', 'variation_qwen_self_generated', 'variation_llama_3_instruct_self_generated']
    # variation_suffix_list_gpt4_gt_cot = ['', "variation_gpt4_style_in_context_examples", 'variation_step_by_step', 'variation_rewrite_groundtruth_in_own_words']#, 'variation_simple_response']#, 'variation_mistral_self_generated', 'variation_qwen_self_generated', 'variation_llama_3_instruct_self_generated']


    variation_suffix_list_only_gold_label = ["variation_gold_label"]
    variation_suffix_list_only_groundtruth = [""]
    variation_suffix_list_both_gold_label_groundtruth = ["", "variation_gold_label"]
    method_list = ["groundtruth", "gpt4", "mini_gpt4", "claude"]

    experiment_result_dict = {}
    for model_name in model_name_list:
        experiment_result_dict[model_name] = {}
        for task_name in task_name_list:
            for method in method_list:
                non_cot_name_list = ['boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'agieval', 'api_bank', 'mmlu_pro', 'mmlu_pro_law', 'hellaswag', 'arc_challenge', 'drop', 'mmlu_moral_scenarios', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse']
                cot_name_list = ['mbpp', "gsm8k", 'math_algebra', 'math_intermediate_algebra', 'math_geometry', 'ecqa', 'esnli']

                # non_cot_name_list = ['boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'agieval', 'api_bank', 'mmlu_pro', 'mmlu_pro_law', 'hellaswag', 'arc_challenge', 'drop', 'mmlu_moral_scenarios']
                # cot_name_list = ['mbpp', "gsm8k", 'math_algebra', 'math_intermediate_algebra', 'math_geometry', 'ecqa', 'esnli', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse']

                # cot_name_list = ['mbpp', "gsm8k", 'math_algebra', 'math_intermediate_algebra', 'math_geometry', 'ecqa', 'esnli']

                only_gold_label_name_list = ['boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'agieval', 'api_bank', 'mmlu_pro', 'mmlu_pro_law', 'hellaswag', 'arc_challenge', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse', 'drop', 'mmlu_moral_scenarios']
                only_groundtruth_name_list = ['mbpp', "gsm8k", 'math_algebra', 'math_intermediate_algebra', 'math_geometry']
                both_gold_label_groundtruth_name_list = ['ecqa', 'esnli']
                if method == "gpt4":
                    if task_name in cot_name_list:
                        variation_suffix_list = variation_suffix_list_gpt4_gt_cot
                    if task_name in non_cot_name_list:
                        variation_suffix_list = variation_suffix_list_gpt4_gt_non_cot
                elif method == "groundtruth":
                    if task_name in only_gold_label_name_list:
                        variation_suffix_list = variation_suffix_list_only_gold_label
                    if task_name in only_groundtruth_name_list:
                        variation_suffix_list = variation_suffix_list_only_groundtruth
                    if task_name in both_gold_label_groundtruth_name_list:
                        variation_suffix_list = variation_suffix_list_both_gold_label_groundtruth
                else:
                    variation_suffix_list = variation_suffix_list_non_gpt4
                for variation_suffix in variation_suffix_list:
                    record_file_name = f"{task_name}_{method}_{variation_suffix}_{seed_num}.txt"
                    file_path = f'{HOME_DIRECTORY}/log_total/experiment_data_recorder/{model_name}/{n_train}/{sft_lr}/{epoch_num}/{record_file_name}'

                    # if not os.path.exists(file_path):
                    #     print(f"File not found: {file_path}. Skipping.")
                    content = ''
                    try:
                        # Open and read the file
                        with open(file_path, 'r') as file:
                            content = file.read()
                        
                        # Ensure nested dictionary structure exists
                        if task_name not in experiment_result_dict[model_name]:
                            experiment_result_dict[model_name][task_name] = {}
                        combined_method_name = method
                        if variation_suffix != '':
                            combined_method_name = variation_suffix.replace('variation_', '')
                        if f'{combined_method_name}' not in experiment_result_dict[model_name][task_name]:
                            experiment_result_dict[model_name][task_name][f'{combined_method_name}'] = {}

                        # Store content in the nested dictionary
                        experiment_result_dict[model_name][task_name][f'{combined_method_name}'] = f"{float(content):.3f}"
                        
                        # Print the content
                        # print(f"Processed {file_path}")
                    except Exception as e:
                        if task_name not in experiment_result_dict[model_name]:
                            experiment_result_dict[model_name][task_name] = {}
                        combined_method_name = method
                        if variation_suffix != '':
                            combined_method_name = variation_suffix.replace('variation_', '')
                        if f'{combined_method_name}' not in experiment_result_dict[model_name][task_name]:
                            experiment_result_dict[model_name][task_name][f'{combined_method_name}'] = load_none_as
    return experiment_result_dict

# with open(file_path, 'r') as file:
#     content = file.read()
# print(content)