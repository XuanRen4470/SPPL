from mix_score_ranking_utils.functions import *
import os
import json
import sys
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.function import load_experimental_result
from config.config import HOME_DIRECTORY



disable_extra_line = False
no_normalization = False

initial_index = 0
# last_index = 10
last_index = 100

# total_num = 20
total_num = 100



import numpy as np

def calculate_diversity(data, mode: str = "cv"):
# def calculate_diversity(data, mode: str = "sd"):

    """
    Compute diversity of numeric values in a dict.

    Parameters
    ----------
    data : dict
        Keys are names, values are numeric-like strings.
    mode : {"cv", "sd"}, optional
        "cv" — coefficient of variation (相对离散度，默认)
        "sd" — sample standard deviation (绝对离散度, ddof=1)

    Returns
    -------
    float
        Diversity score.

    Notes
    -----
    * 键名为 'gold_label' 的项会被忽略。
    * CV = SD / mean，可跨量级比较；SD 保留原单位。
    """

    # values = np.array(
    #     [float(v) for k, v in data.items()], dtype=float
    # )

    values = np.array(
        [float(v) for k, v in data.items() if k != "gold_label"], dtype=float
    )

    if values.size == 0:
        raise ValueError("No numeric values found in `data`.")

    if mode == "sd":
        # 样本标准差 (ddof=1)
        return float(np.std(values, ddof=1))

    if mode == "cv":
        mean = float(np.mean(values))
        sd   = float(np.std(values, ddof=1))
        return sd / mean if mean != 0 else 0.0

    raise ValueError("`mode` must be 'cv' or 'sd'.")


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


def calculate_average(dictionary):
    total = 0
    count = 0
    max = -1
    min = 999999
    for key, value in dictionary.items():
        if value is not None:  # Skip if value is None
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


def save_bar_plot_acc(labels, values, model_name, ax=None):
    """
    生成一个竖向柱状图，并保存到指定路径或展示在指定的子图中。
    
    参数：
    - labels: List[str]，柱子的标签
    - values: List[float]，对应的数值
    - model_name: str，模型名称
    - ax: matplotlib.axes.Axes，指定的子图（如果为 None，创建新的图）
    """
    if ax is None:
        # 如果没有传递 ax 参数，创建新的图
        plt.figure(figsize=(8, 6))
        ax = plt.gca()  # 获取当前轴

    # 使用一个离散的颜色集合来区分每个 bar
    colors = plt.cm.tab10(np.linspace(0, 1, len(values)))

    # 创建竖向柱状图
    bars = ax.bar(labels, values, color=colors, width=0.6)

    # 设置 y 轴范围以拉大数值差距，使区别更明显
    min_val = min(values)
    max_val = max(values)
    buffer = (max_val - min_val) * 0.3  # 添加30%的间距
    ax.set_ylim(0.55, 0.70)

    # 添加数值标签（显示在柱子顶部）
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f'{height:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置标题和轴标签
    ax.set_title(model_name, fontsize=14, fontweight='bold')
    # ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)

    # 旋转 x 轴标签，以防止重叠
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)

    # legend_labels = [f'{label}' for label in labels]
    # ax.legend(bars, legend_labels, loc='upper right', fontsize=10)


def save_bar_plot_pho(labels, values, model_name, ax=None):
    """
    生成一个竖向柱状图，并保存到指定路径或展示在指定的子图中。
    
    参数：
    - labels: List[str]，柱子的标签
    - values: List[float]，对应的数值
    - model_name: str，模型名称
    - ax: matplotlib.axes.Axes，指定的子图（如果为 None，创建新的图）
    """
    if ax is None:
        # 如果没有传递 ax 参数，创建新的图
        plt.figure(figsize=(8, 6))
        ax = plt.gca()  # 获取当前轴

    # 使用一个离散的颜色集合来区分每个 bar
    colors = plt.cm.tab10(np.linspace(0, 1, len(values)))

    # 创建竖向柱状图
    bars = ax.bar(labels, values, color=colors, width=0.6)

    # 设置 y 轴范围以拉大数值差距，使区别更明显
    min_val = min(values)
    max_val = max(values)
    buffer = (max_val - min_val) * 0.3  # 添加30%的间距
    ax.set_ylim(-0.2, 0.5)

    # 添加数值标签（显示在柱子顶部）
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f'{height:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置标题和轴标签
    ax.set_title(model_name, fontsize=14, fontweight='bold')
    # ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Average Spearman Correlation', fontsize=12)

    # 旋转 x 轴标签，以防止重叠
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)

    # legend_labels = [f'{label}' for label in labels]
    # ax.legend(bars, legend_labels, loc='upper right', fontsize=10)





diversity_list_dict = {}
for model_name in ['mistral', 'llama_3_instruct', 'qwen']:
    diversity_list = []
    for task_name in ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']:
        seed_num = 0
        sft_lr = '2e-05'
        n_train = 1000
        if 'plan_bench' in task_name:
            epoch_num = 40
        else:
            epoch_num = 20
        default_lr_experiment_result_dict = load_experimental_result([model_name],['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank'], n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
        seed_num = 1
        default_lr_experiment_result_dict_1 = load_experimental_result([model_name],['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank'], n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
        seed_num = 2
        default_lr_experiment_result_dict_2 = load_experimental_result([model_name],['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank'], n_train, sft_lr, seed_num, epoch_num, load_none_as = None)

        try:
            diversity = calculate_diversity(default_lr_experiment_result_dict[model_name][task_name])
        except:
            a = 1
    
        print("Diversity (Standard Deviation):", diversity)
        diversity_list.append(diversity)
    diversity_list_dict[model_name] = diversity_list


# for only_record_designed_prompt in [False]:#, True]:
for only_record_designed_prompt in [True]:
    # for model_name in ['qwen']: 
    # for model_name in ['llama_3_instruct']:
    # for model_name in ['mistral']:


    # beta_value = 0
    beta_value = 0.5
    # beta_value = -1

    # diff_threshold = 0
    diff_threshold = -10000
    # diff_threshold = 0.1
    # gap_threshold = 0.02
    gap_threshold = -1

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # for kk, diff_threshold in enumerate([0.05, 0]):
    for kk, diff_threshold in enumerate([diff_threshold]):
    # for kk, diff_threshold in enumerate([0.05]):
        # ranking_method_list = ['both', 'perplexity', 'similarity', 'complexity', 'length', 'IDF']
        ranking_method_list = ['both']
        ranking_method_dict = {}
        ii = 0    

        values_rho_dict = {}
        values_accuracy_dict = {}
        
        for model_name in ['mistral', 'llama_3_instruct', 'qwen']:
        # for model_name in ['mistral']:
        # for model_name in ['llama_3_instruct']:
        # for model_name in ['qwen']:
            values_rho_total = []
            values_accuracy_total = []
            values_rho = []
            values_accuracy = []

            # default_lr_task_name_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'api_bank', 'mmlu_moral_scenarios', 'math_geometry']
            default_lr_task_name_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']

            task_name_list = default_lr_task_name_list 
            
            
            for ranking_method in ranking_method_list:
                weighted_rho = []
                best_rho = -9999999
                best_layer_num = 0
                spear_man_coefficient_record_book = {}
                avg_accuracy_best = 0
                avg_mix_score = 0
                counted_task_num = 0

                avg_rho_total = 0
                avg_mix_score_total = 0

                
                # avg_groundtruth = 0
                # avg_in_own_words = 0
                # avg_gold_label = 0

                # for layer_num in [15]:
                for layer_num in ['all']:
                    # for use_token_length in [True, False]:
                    for use_token_length in [False]:
                    # for use_token_length in [True]:
                        method_names_with_ranks_total = []
                        layerwise_best_record_book_list = []
                        layerwise_best_record_book = {}
                        log_line = ''
                        best_log_line = ''
                        total_rho = 0
                        for task_name in task_name_list:
                            seed_num = 0
                            sft_lr = '2e-05'
                            n_train = 100
                            epoch_num = 20
                            default_lr_experiment_result_dict_100 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)

                            seed_num = 0
                            sft_lr = '2e-05'
                            n_train = 1000
                            epoch_num = 20
                            default_lr_experiment_result_dict = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
                            seed_num = 1
                            default_lr_experiment_result_dict_1 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
                            seed_num = 2
                            default_lr_experiment_result_dict_2 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
                        

                            diff_ = 1
                            average_result_100, gap_100 = calculate_average(default_lr_experiment_result_dict_100[model_name][task_name])

                            if average_result_100:
                                average_result, gap = calculate_average(default_lr_experiment_result_dict[model_name][task_name])
                                diff_ = average_result - average_result_100
                                improved_percentage = diff_ / average_result_100
                            if improved_percentage < diff_threshold:
                                a = 1
                            if improved_percentage > diff_threshold:
                                counted_task_num += 1
                                if not use_token_length:
                                    print(f'===================={task_name}: not use token length ====================')
                                else:
                                    print(f'===================={task_name}: use token length ====================')
                                log_line += f'===================={task_name}====================\n'
                                best_log_line += f'===================={task_name}====================\n'

                                if task_name in default_lr_task_name_list:
                                    experiment_result_dict = default_lr_experiment_result_dict
                                    experiment_result_dict_1 = default_lr_experiment_result_dict_1
                                    experiment_result_dict_2 = default_lr_experiment_result_dict_2
                                

                                seed_num = 0
                                n_train = 100

                                sft_lr = '0.0002'
                                epoch_num = 40
                                
                                # sft_lr = '2e-05'
                                # epoch_num = 20
                                # # 
                                default_lr_experiment_result_dict_100 = load_experimental_result([model_name],task_name_list, n_train, sft_lr, seed_num, epoch_num, load_none_as = None)
                                record_100 = default_lr_experiment_result_dict_100[model_name][task_name]
                                # Convert the values to float and sort in descending order
                                # record_100_item = sorted(record_100.items(), key=lambda item: float(item[1]), reverse=True)

                                # # Extract the ranked methods (keys)
                                # ranked_method_name_list = [key for key, value in record_100_item]


                                record_100_item = sorted(record_100.items(), key=lambda item: float(item[1]) if item[1] is not None else float('-inf'), reverse=True)
                                record_100_item = [item for item in record_100_item if item[0] != "gold_label"]
                                ranked_method_name_list = [key for key, value in record_100_item if value is not None]

                                rho, gt_accuracy_record_detail, key_name_list, accuracy_best, chosen_method_accuracy = calc_spearman_coefficient_3(experiment_result_dict[model_name][task_name], experiment_result_dict_1[model_name][task_name], experiment_result_dict_2[model_name][task_name], ranked_method_name_list, task_name)
                                avg_accuracy_best += float(accuracy_best)

                                if chosen_method_accuracy is not None:
                                    avg_mix_score += float(chosen_method_accuracy)
                                

                                total_rho += rho
                                print(rho)
                                weighted_rho.append(rho)


                                for item in gt_accuracy_record_detail:
                                    print(f'response rank:    {item}')
                                print('----------------------')
                                for iiiii, item in enumerate(gt_accuracy_record_detail):
                                    print(f'gt rank: {iiiii + 1}   {item}')

                                if layerwise_best_record_book:
                                    save_best_varient(model_name, layerwise_best_record_book, method_names_with_ranks_total, use_token_length, only_record_designed_prompt)

                                if 'plan_bench' in task_name:
                                    a = 1
                            
                        avg_mix_score /= counted_task_num
                        print(f'{ranking_method} avg_mix_score: {avg_mix_score}')

                        ranking_method_dict[ranking_method] = {}
                        ranking_method_dict[ranking_method]['accuracy'] = avg_accuracy_best
                    
                    avg_rho = total_rho/counted_task_num
                    ranking_method_dict[ranking_method]['rho'] = avg_rho
                    print(f'{ranking_method} layer_num: {layer_num}     avg_rho: {avg_rho}')


                    weighted_rho = weighted_spearman_correlation(weighted_rho, diversity_list)

                    # total_rho += rho
                    # print(rho)
                    # weighted_rho.append(rho)
                    avg_rho = weighted_rho
                    values_rho.append(avg_rho)
                    values_accuracy.append(avg_mix_score)
                    avg_rho_total += avg_rho
                    avg_mix_score_total += avg_mix_score
                values_rho_total.append(weighted_rho)
                values_accuracy_total.append(avg_mix_score_total)
            values_rho_dict[model_name] = values_rho_total
            values_accuracy_dict[model_name] = values_accuracy_total
            ranking_method_list_temp = []
            for item in ranking_method_list:
                if item == 'both':
                    ranking_method_list_temp.append('ours')
                else:
                    ranking_method_list_temp.append(item)
            ii += 1
        values_accuracy_list = []
        values_rho_list = []

        for iiiiii in range(len(ranking_method_list_temp)):
            sum = values_accuracy_dict['mistral'][iiiiii] + values_accuracy_dict['llama_3_instruct'][iiiiii] + values_accuracy_dict['qwen'][iiiiii]
            avg = sum/3
            values_accuracy_list.append(avg)

            # print('mistral avg: ' , values_accuracy_dict['mistral'][iiiiii])
            # print('llama_3_instruct avg: ' , values_accuracy_dict['llama_3_instruct'][iiiiii])
            # print('qwen avg: ' , values_accuracy_dict['qwen'][iiiiii])

            sum = values_rho_dict['mistral'][iiiiii] + values_rho_dict['llama_3_instruct'][iiiiii] + values_rho_dict['qwen'][iiiiii]
            avg = sum/3
            values_rho_list.append(avg)

            print('mistral pho: ' , values_rho_dict['mistral'][iiiiii])
            print('llama_3_instruct pho: ' , values_rho_dict['llama_3_instruct'][iiiiii])
            print('qwen pho: ' , values_rho_dict['qwen'][iiiiii])
        a = 1

a = 1