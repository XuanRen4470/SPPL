from mix_score_ranking_utils.functions import *
import os
import json
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.function import load_experimental_result
from utils.data_recorder import write_to_table
from config.config import HOME_DIRECTORY


# rank_method_list = ['in_context_perplexity', 'perplexity', 'IDF', 'skywork', 'CAR']
# rank_method_list = ['IDF', 'skywork', 'CAR', 'perplexity', 'in_context_perplexity']
rank_method_list = ['perplexity', 'in_context_perplexity']

# rank_method_list = ['in_context_perplexity']



default_lr_task_name_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']

model_name_list = ['mistral', 'llama_3_instruct', 'qwen']

diff_threshold_list = [-1111]

n_train_recorder = 300
n_train_recorder_other_metrics = 300

index_range_list = [(0, 10), (10, 20), (20, 30), (0, 30), (30, 60), (60, 90), (0, 50), (50, 100), (100, 150), (0, 100), (100, 200), (200, 300), (0, 100), (0, 200), (0, 300)]

# index_range_list = [(0, 200), (0, 300)]


experimental_result_list = []
num_of_run = len(index_range_list)
suffix_ = ''
# suffix_ = '_may_18'
# suffix_ = '_simple'

# num_of_incontext_examples = 5
# num_of_incontext_examples = 4
num_of_incontext_examples = 3
# num_of_incontext_examples = 2
# num_of_incontext_examples = 1


# use_plan_prompt = True
use_plan_prompt = False

table_tex_name = f'icppl vs other metrics: average of {num_of_run} subsets{suffix_}'


def calculate_average(dictionary):
    total = 0
    count = 0
    max = -1
    min = 999999
    for key, value in dictionary.items():
        if value is not None:  # Skip if value is None
            if float(value)> 0:
                total += float(value)
                count += 1
                if float(value) < min:
                    min = float(value)
                if float(value) > max:
                    max = float(value)
        
    if count == 0:  # Handle the case where all values are None
        return None, None
    gap = max - (total / count)
    return total / count, gap


def calc_avg(lst):
    # Filter out None or 0 values
    filtered_lst = [x for x in lst if x not in (None, 0)]
    if filtered_lst:  # Avoid division by zero if the list is empty after filtering
        aaa = sum(filtered_lst) / len(filtered_lst)
        return aaa
    else:
        return 0  # Or handle the case where the list is empty after filtering


def calculate_diversity(data):
    """
    Calculate the diversity of values in a dictionary by computing the standard deviation.

    :param data: Dictionary where keys are methods and values are numerical strings.
    :return: Standard deviation of the values.
    """
    values = np.array(
        [float(v) for k, v in data.items() if k != "gold_label"], dtype=float
    )
    return np.std(values, ddof=1)  # Use ddof=1 for sample standard deviation


# def calculate_diversity(data, mode: str = "cv"):
# # def calculate_diversity(data, mode: str = "sd"):

#     """
#     Compute diversity of numeric values in a dict.

#     Parameters
#     ----------
#     data : dict
#         Keys are names, values are numeric-like strings.
#     mode : {"cv", "sd"}, optional
#         "cv" — coefficient of variation (相对离散度，默认)
#         "sd" — sample standard deviation (绝对离散度, ddof=1)

#     Returns
#     -------
#     float
#         Diversity score.

#     Notes
#     -----
#     * 键名为 'gold_label' 的项会被忽略。
#     * CV = SD / mean，可跨量级比较；SD 保留原单位。
#     """
#     # # 过滤掉非数值或特殊键
#     # values = np.array(
#     #     [float(v) for k, v in data.items() if k != "gold_label"], dtype=float
#     # )
#     values = np.array(
#         [float(v) for k, v in data.items() if k != "gold_label"], dtype=float
#     )

#     # values = np.array(
#     #     [float(v) for k, v in data.items() if k != "rewrite_groundtruth_in_own_words" and k != "gold_label"], dtype=float
#     # )


    

#     values = np.array(
#         [float(v) for k, v in data.items()], dtype=float
#     )

#     if values.size == 0:
#         raise ValueError("No numeric values found in `data`.")

#     if mode == "sd":
#         # 样本标准差 (ddof=1)
#         return float(np.std(values, ddof=1))

#     if mode == "cv":
#         mean = float(np.mean(values))
#         sd   = float(np.std(values, ddof=1))
#         return sd / mean if mean != 0 else 0.0

#     raise ValueError("`mode` must be 'cv' or 'sd'.")


import numpy as np
from scipy.stats import spearmanr

def weighted_spearman_correlation(spearman_list, diversity_list):
    """
    Calculate the weighted Spearman correlation between two lists.

    :param spearman_list: List of Spearman correlation values.
    :param diversity_list: List of diversity values used as weights.
    :return: Weighted Spearman correlation.
    """
    # Convert the lists to numpy arrays for easier manipulation
    spearman_array = np.array(spearman_list)
    diversity_array = np.array(diversity_list)

    # Check if both lists have the same length
    if len(spearman_array) != len(diversity_array):
        raise ValueError("The lists must have the same length.")

    # Calculate the weighted rank correlation
    weighted_rho = np.sum(spearman_array * diversity_array) / np.sum(diversity_array)

    return weighted_rho
    

def calculate_average_3(dictionary, dictionary_1, dictionary_2):
    total = 0
    count = 0
    max_val = float('-inf')
    min_val = float('inf')
    
    # Combine the three dictionaries in a list so we can iterate easily
    for d in [dictionary, dictionary_1, dictionary_2]:
        for key, value in d.items():
            if value is not None:  # Skip None values
                numeric_val = float(value)
                total += numeric_val
                count += 1
                
                if numeric_val < min_val:
                    min_val = numeric_val
                if numeric_val > max_val:
                    max_val = numeric_val
    
    # If there were no valid values in any of the dictionaries
    if count == 0:
        return None, None
    
    average = total / count
    gap = max_val - average
    return average, gap


diversity_list_dict = {}
for model_name in model_name_list:
    task_name_list = default_lr_task_name_list
    diversity_list = []
    for task_name in task_name_list:
        seed_num = 0
        sft_lr = '2e-05'
        n_train = 1000
        if 'plan_bench' in task_name:
            epoch_num = 40
        else:
            epoch_num = 20
        default_lr_experiment_result_dict = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
        seed_num = 1
        default_lr_experiment_result_dict_1 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
        seed_num = 2
        default_lr_experiment_result_dict_2 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)

        try:
            diversity = calculate_diversity(default_lr_experiment_result_dict[model_name][task_name])
        except:
            a = 1
    
        print("Diversity (Standard Deviation):", diversity)
        diversity_list.append(diversity)
    diversity_list_dict[model_name] = diversity_list



generation_strategy_name_list = default_lr_experiment_result_dict[model_name][task_name].keys()
strategy_name_list = []
for strategy_name in generation_strategy_name_list:
    strategy_name_list.append(strategy_name)

a = 1


experimental_result = {}
experimental_result['Upper bound'] = {}

for ranking_method in rank_method_list:
    if ranking_method == 'in_context_perplexity':
        experimental_result['Ours'] = {}
    else:
        experimental_result[ranking_method] = {}


experimental_result['Ours'] = {}
experimental_result['Ours - Perplexity'] = {}




column_name_list = ['STD Range', 'num of recorded data']
for model_name in model_name_list:
    column_name_list.append(model_name)
column_name_list.append('Avg Acc')
column_name_list.append('Weighted Spearman Pho')

for row_name in experimental_result.keys():
    for column_name in column_name_list:
        experimental_result[row_name][column_name] = ''


for kk, diff_threshold in enumerate(diff_threshold_list):
    
    experiment_recorder_different_seed_list = []
    for initial_index, last_index in index_range_list:
        experimental_result = {}
        experimental_result['Upper bound'] = {}
        for ranking_method in rank_method_list:
            if ranking_method == 'in_context_perplexity':
                experimental_result['Ours'] = {}
            else:
                experimental_result[ranking_method] = {}
        experimental_result['Ours - Perplexity'] = {}

        column_name_list = ['STD Range', 'num of recorded data']
        for model_name in model_name_list:
            column_name_list.append(model_name)
        column_name_list.append('Avg Acc')
        column_name_list.append('Weighted Spearman Pho')

        for row_name in experimental_result.keys():
            for column_name in column_name_list:
                experimental_result[row_name][column_name] = ''

        perplexity_avg_accuracy = 0
        perplexity_avg_pho = 0
        perplexity_avg_accuracy_per_model_dict = {}
        perplexity_avg_pho_per_moodel_dict = {}
        
        for ranking_method in rank_method_list:
            avg_accuracy_best = []
            avg_mix_score = []
            avg_gpt4 = []
            avg_claude = []
            avg_mini_gpt4 = []
            avg_human_example = []
            avg_gpt4_example = []
            avg_step_by_step = []
            diversity_list = []
            weighted_rho = []

            counted_task_num = 0
            for model_name in model_name_list:
                avg_accuracy_best_per_model = []
                avg_mix_score_per_model = []
                avg_gpt4_per_model = []
                avg_claude_per_model = []
                avg_mini_gpt4_per_model = []
                avg_human_example_per_model = []
                avg_gpt4_example_per_model = []
                avg_step_by_step_per_model = []
                diversity_list_per_model = []
                weighted_rho_per_model = []


                pho_for_each_model = []
                acc_for_each_model = []
                diversity_for_each_model = []
                task_name_list = default_lr_task_name_list
                avg_mix_score_subset_total = []
                method_names_with_ranks_total = []
                layerwise_best_record_book_list = []
                layerwise_best_record_book = {}
                log_line = ''
                best_log_line = ''
                
                for task_name in task_name_list:
                    in_context_ppl_cos_record_list = []
                    ppl_cos_record_list = []
                    if use_plan_prompt:
                        if 'plan' in task_name:
                            suffix_ = f'_{num_of_incontext_examples}_examples_use_plan_prompt'
                        else:
                            suffix_ = f'_{num_of_incontext_examples}_examples'
                    else:
                        suffix_ = f'_{num_of_incontext_examples}_examples'
                    if 'plan_bench' in task_name:
                        file_path = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train_recorder}_icppl_{model_name}_{task_name}_{initial_index}_{last_index}_plan_bench{suffix_}.json'
                        # file_path = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train_recorder}_icppl_{model_name}_{task_name}_{initial_index}_{last_index}.json'
                    else:
                        file_path = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train_recorder}_icppl_{model_name}_{task_name}_{initial_index}_{last_index}_main{suffix_}.json'
                        # file_path = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train_recorder}_icppl_{model_name}_{task_name}_{initial_index}_{last_index}.json'
                    with open(file_path, 'r') as f:
                        in_context_ppl_cos_record = json.load(f)

                    if last_index - initial_index > 100:
                        file_path = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{300}_other_metrics_{model_name}_{task_name}_{0}_{100}.json'      
                    else:
                        file_path = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train_recorder_other_metrics}_other_metrics_{model_name}_{task_name}_{initial_index}_{last_index}.json'                            
                    with open(file_path, 'r') as f:
                        ppl_cos_record = json.load(f)

                    

                    a = 1

                    seed_num = 0
                    sft_lr = '2e-05'
                    n_train = 1000
                    if 'plan_bench' not in task_name:
                        epoch_num = 20
                    else:
                        epoch_num = 40
                    default_lr_experiment_result_dict = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
                    seed_num = 1
                    default_lr_experiment_result_dict_1 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
                    seed_num = 2
                    default_lr_experiment_result_dict_2 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
                
                    diversity = calculate_diversity(default_lr_experiment_result_dict[model_name][task_name])              

                    if diversity > diff_threshold:
                        diversity_list.append(diversity)
                        diversity_for_each_model.append(diversity)
                        counted_task_num += 1
                        # print(f'===================={task_name}====================')
                        log_line += f'===================={task_name}====================\n'
                        best_log_line += f'===================={task_name}====================\n'

                        ppl_values, IDF_values, log_propability_values, skywork_reward_score_values, CAR_score_values, name_list = loading_other_metric_scores(task_name, ppl_cos_record)
                        
                        in_context_ppl_values, _ = loading_icppl_scores(task_name, in_context_ppl_cos_record)

                        
                        if task_name in default_lr_task_name_list:
                            experiment_result_dict = default_lr_experiment_result_dict
                            experiment_result_dict_1 = default_lr_experiment_result_dict_1
                            experiment_result_dict_2 = default_lr_experiment_result_dict_2

                
                        ranked_method_name_list, method_names_with_ranks = rank_on_metrics(in_context_ppl_values, ppl_values, IDF_values, log_propability_values, skywork_reward_score_values, CAR_score_values, name_list, ranking_method = ranking_method)


                        rho, gt_accuracy_record_detail, key_name_list, accuracy_best, chosen_method_accuracy = calc_spearman_coefficient_3(experiment_result_dict[model_name][task_name], experiment_result_dict_1[model_name][task_name], experiment_result_dict_2[model_name][task_name], ranked_method_name_list, task_name)
                        avg_accuracy_best.append(float(accuracy_best))
                        avg_accuracy_best_per_model.append(float(accuracy_best))

                        if chosen_method_accuracy is not None:
                            avg_mix_score.append(float(chosen_method_accuracy))
                            avg_mix_score_per_model.append(float(chosen_method_accuracy))

                        if experiment_result_dict[model_name][task_name]['gpt4'] is not None:
                            avg_gpt4.append(float(experiment_result_dict[model_name][task_name]['gpt4']))
                            avg_gpt4_per_model.append(float(experiment_result_dict[model_name][task_name]['gpt4']))

                        if experiment_result_dict[model_name][task_name]['claude'] is not None:
                            avg_claude.append(float(experiment_result_dict[model_name][task_name]['claude']))
                            avg_claude_per_model.append(float(experiment_result_dict[model_name][task_name]['claude']))

                        if experiment_result_dict[model_name][task_name]['mini_gpt4'] is not None:
                            avg_mini_gpt4.append(float(experiment_result_dict[model_name][task_name]['mini_gpt4']))
                            avg_mini_gpt4_per_model.append(float(experiment_result_dict[model_name][task_name]['mini_gpt4']))
                        
                        if experiment_result_dict[model_name][task_name]['openai_human_written_examples'] is not None:
                            avg_human_example.append(float(experiment_result_dict[model_name][task_name]['openai_human_written_examples']))
                            avg_human_example_per_model.append(float(experiment_result_dict[model_name][task_name]['openai_human_written_examples']))

                        if experiment_result_dict[model_name][task_name]['gpt4_style_in_context_examples'] is not None:
                            avg_gpt4_example.append(float(experiment_result_dict[model_name][task_name]['gpt4_style_in_context_examples']))
                            avg_gpt4_example_per_model.append(float(experiment_result_dict[model_name][task_name]['gpt4_style_in_context_examples']))

                        if experiment_result_dict[model_name][task_name]['step_by_step'] is not None:
                            avg_step_by_step.append(float(experiment_result_dict[model_name][task_name]['step_by_step']))
                            avg_step_by_step_per_model.append(float(experiment_result_dict[model_name][task_name]['step_by_step']))
                    
                        # for item in method_names_with_ranks:
                        #     print(f'response rank:    {item}')
                        # print('----------------------')
                        
                        weighted_rho.append(rho)

                        pho_for_each_model.append(rho)
                        acc_for_each_model.append(float(chosen_method_accuracy))

                        # for iiiii, item in enumerate(gt_accuracy_record_detail):
                        #     print(f'gt rank: {iiiii + 1}   {item}')
                        # print('Spearman: ', rho)
                        # print()
                

                    

                # print('-------------------------------------------------------------------')
                aveavg_mix_scorerage_per_model = sum(avg_mix_score_per_model) / len(avg_mix_score_per_model)
                avg_rho_score_average_per_model = weighted_spearman_correlation(pho_for_each_model, diversity_for_each_model)
                # print(f'weighted pho for {model_name}: {avg_rho_score_average_per_model}')
                # print(f'avg acc for {model_name}: {aveavg_mix_scorerage_per_model}')
                # print()
                # print()


                if ranking_method == 'perplexity':
                    perplexity_avg_accuracy_per_model_dict[model_name] = aveavg_mix_scorerage_per_model

                experimental_result['Upper bound'][model_name] = calc_avg(avg_accuracy_best_per_model)


                if ranking_method == 'in_context_perplexity':
                    experimental_result['Ours'][model_name] = calc_avg(avg_mix_score_per_model)
                    experimental_result['Ours - Perplexity'][model_name] = calc_avg(avg_mix_score_per_model) - perplexity_avg_accuracy_per_model_dict[model_name]
                else:
                    experimental_result[ranking_method][model_name] = calc_avg(avg_mix_score_per_model)
                                        
            num_of_recorded_data = len(weighted_rho)
            avg_pho = sum(weighted_rho) / len(weighted_rho)
            weighted_rho = weighted_spearman_correlation(weighted_rho, diversity_list)
            num_samples = len(avg_mix_score)
            avg_mix_score = calc_avg(avg_mix_score)
            avg_gpt4 = calc_avg(avg_gpt4)
            avg_claude = calc_avg(avg_claude)
            avg_mini_gpt4 = calc_avg(avg_mini_gpt4)
            avg_human_example = calc_avg(avg_human_example)
            avg_gpt4_example = calc_avg(avg_gpt4_example)
            avg_step_by_step = calc_avg(avg_step_by_step)
            avg_accuracy_best = calc_avg(avg_accuracy_best)


            # print('////////////////////////////////')

            # print(f'avg_accuracy_best: {avg_accuracy_best}')
            # print(f'avg_mix_score: {avg_mix_score}')
            
            aveavg_mix_scorerage = avg_mix_score 


            # print('---------------------')
            # print(f'avg pho: {avg_pho}')
            # print(f'weighted_rho: {weighted_rho}')



            experimental_result['Upper bound']['Avg Acc'] = avg_accuracy_best



            
            if diff_threshold < 0:
                experimental_result['Upper bound']['STD Range'] = 'All Data'

            if diff_threshold > 0:
                experimental_result['Upper bound']['STD Range']  = f"$STD > {diff_threshold * 100:.2f}\\%$"

            experimental_result['Upper bound']['num of recorded data']  = str(num_of_recorded_data)
            
            if ranking_method == 'in_context_perplexity':
                experimental_result['Ours']['Avg Acc'] = avg_mix_score
                experimental_result['Ours - Perplexity']['Avg Acc'] = avg_mix_score - perplexity_avg_accuracy
                experimental_result['Ours - Perplexity']['Weighted Spearman Pho'] = weighted_rho - perplexity_avg_pho
                experimental_result['Ours']['Weighted Spearman Pho'] = weighted_rho
            else:
                experimental_result[ranking_method]['Avg Acc'] = avg_mix_score
                experimental_result[ranking_method]['Weighted Spearman Pho'] = weighted_rho
                    
            if ranking_method == 'perplexity':
                perplexity_avg_accuracy = avg_mix_score
                perplexity_avg_pho= weighted_rho
            

            if 'in_context_perplexity' == ranking_method:
                print()
                print()
                print('initial index :',  str(initial_index))
                print('last index :',  str(last_index))
                print('acc: ', "{:.2f}\\%".format(experimental_result['Ours']['Avg Acc'] * 100))
                print('pho: ', "{:.3f}".format(experimental_result['Ours']['Weighted Spearman Pho']))

    # import copy
    # experimental_result_update = copy.deepcopy(experiment_recorder_different_seed_list[0])
    # for method_name in experimental_result_update.keys():
    #     for metric_name in experimental_result_update[method_name].keys():
    #         record_enable = False
    #         try:
    #             float(experimental_result_update[method_name][metric_name])
    #             record_enable = True
    #         except:
    #             record_enable = False
    #         if record_enable:
    #             total_value = 0
    #             for experiment_recorder_different_seed_item in experiment_recorder_different_seed_list:
    #                 total_value += float(experiment_recorder_different_seed_item[method_name][metric_name])
    #             total_value /= len(index_range_list)
    #             experimental_result_update[method_name][metric_name] = total_value
    # experimental_result = experimental_result_update


    
    # experimental_result_list.append(experimental_result)


# print()
# print()

# print()

# print()

# print()

# print()

# print()

# print('Ours - Perplexity: mistral', experimental_result['Ours - Perplexity']['mistral'] * 100)
# print('Ours - Perplexity: llama_3_instruct', experimental_result['Ours - Perplexity']['llama_3_instruct']* 100)
# print('Ours - Perplexity: qwen', experimental_result['Ours - Perplexity']['qwen']* 100)
# print('Ours Avg Acc', experimental_result['Ours - Perplexity']['Avg Acc']* 100)
# print('Ours Weighted Spearman Pho', experimental_result['Ours - Perplexity']['Weighted Spearman Pho'])

# write_to_table(experimental_result_list, table_tex_name)