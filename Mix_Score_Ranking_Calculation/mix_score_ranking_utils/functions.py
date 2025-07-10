import numpy as np
from scipy.stats import rankdata
import os
import json
import math
import re
import statistics

def extract_prompt_generation_method(input_string):
    method = {}
    
    try:
        # Extract the number between '_total_' and '_generation_api'
        number_match = re.search(r'_total_(.*?)_prompt_api', input_string)
        prompt_number = number_match.group(1) if number_match else None

        prompt_api_match = re.search(r'_prompt_api_(.*?)_generation_api_', input_string)
        prompt_api = prompt_api_match.group(1) if prompt_api_match else None

        if 'simple_prompt' in input_string:
            # Extract the API between '_generation_api_' and '_simple_prompt'
            method['simple_prompt'] = True
            answer_api_match = re.search(r'_generation_api_(.*?)_simple_prompt', input_string)
            answer_api = answer_api_match.group(1) if answer_api_match else None
        else:
            # Extract the API between '_generation_api_' and '_simple_prompt'
            method['simple_prompt'] = False
            answer_api_match = re.search(r'_generation_api_(.*?)  ', input_string)
            answer_api = answer_api_match.group(1) if answer_api_match else None

        method['prompt_api'] = prompt_api
        method['prompt_number'] = int(prompt_number)
        method['answer_api'] = answer_api
    except:
        method = 'not_total_method'

    return method



def compute_beta(ppl_std_dev, cos_similarity_std_dev, mean_ppl, mean_cos_similarity):
    beta = 0
    ppl_ratio = ppl_std_dev/mean_ppl
    cos_similarity_ratio = cos_similarity_std_dev/mean_cos_similarity

    beta = ppl_ratio/(ppl_ratio + cos_similarity_ratio) 
    return beta

def compute_length_punishment_value(avg_initial_prediction_length, avg_target_response_length):
    """
    Computes the punishment coefficient based on the length difference between
    the target response and the initial prediction.

    Parameters:
    - avg_initial_prediction_length (float): Average length of the initial predictions.
    - avg_target_response_length (float): Average length of the target responses.

    Returns:
    - float: Punishment coefficient. 0 if the length difference is within 20%, 
             linearly increases otherwise.
    """
    # Calculate the absolute length difference ratio
    length_difference_ratio = abs(avg_target_response_length - avg_initial_prediction_length) / avg_initial_prediction_length
    
    # If the difference is within 20%, no punishment
    if length_difference_ratio <= 0.2:
        return 0.0
    
    # Linearly increase punishment for differences beyond 20%
    punishment_coefficient = (length_difference_ratio - 0.2) * 5  # Adjust scaling factor (e.g., 5) if needed
    
    return punishment_coefficient

def rank_on_mix_score(task_name, ppl_values, in_context_ppl_values, name_list, method_names_with_ranks_total, log_line, best_log_line, total_layer_best_record_book, IDF_values = '', token_length_values = '', ranking_method = 'both'):#, 


    # Proceed with normalization
    ppl_min = min(ppl_values)
    ppl_max = max(ppl_values)


    # Proceed with normalization
    in_context_ppl_min = min(in_context_ppl_values)
    in_context_ppl_max = max(in_context_ppl_values)


    IDF_min = min(IDF_values)
    IDF_max = max(IDF_values)

    # if use_token_length:
    token_length_min = min(token_length_values)
    token_length_max = max(token_length_values)


    # Normalize values
    normalized_ppl_list = []
    normalized_in_context_perp_list = []
    normalized_token_length_list = []
    IDF_score_list = []

    for perp, in_context_perp, token_length, IDF_item in zip(ppl_values, in_context_ppl_values, token_length_values, IDF_values):
        normalized_perp = 0.0
        normalized_token_length = 0.0
        
        
        if ppl_max != ppl_min:
            normalized_perp = (perp - ppl_min) / (ppl_max - ppl_min)
        if in_context_ppl_max != in_context_ppl_min:
            normalized_in_context_perp = (in_context_perp - in_context_ppl_min) / (in_context_ppl_max - in_context_ppl_min)
        if token_length_max != token_length_min:
            normalized_token_length = (token_length - token_length_min) / (token_length_max - token_length_min)
        if IDF_max != IDF_min:
            IDF_score = (IDF_item - IDF_min) / (IDF_max - IDF_min)
        
        normalized_ppl_list.append(normalized_perp)
        normalized_in_context_perp_list.append(normalized_in_context_perp)
        normalized_token_length_list.append(normalized_token_length)
        IDF_score_list.append(IDF_score)

        # if avg_initial_prediction_token_length:
        #     length_punishment_coefficient = compute_length_punishment_value(avg_initial_prediction_token_length, token_length)
        #     length_punishment_coefficient_list.append(length_punishment_coefficient)

    total_score_list = []
    for normalized_perp, normalized_in_context_perp, normalized_token_length, normalized_IDF in zip(normalized_ppl_list, normalized_in_context_perp_list, normalized_token_length_list, IDF_score_list):
        if 'in_context_perplexity' in ranking_method:
            total_score = normalized_in_context_perp
        elif 'perplexity' in ranking_method:
            total_score = normalized_perp
        elif 'length' in ranking_method:
            total_score = normalized_token_length
        elif 'IDF' in ranking_method:
            total_score = normalized_IDF
        else:
            raise ValueError("Invalid ranking method. Choose 'both', 'perplexity', or 'similarity'.")
        total_score_list.append(total_score)

    # Rank based on total score
    total_score_array = np.array(total_score_list)
    overall_rank = rankdata(total_score_array, method='min')

    # Prepare method names with ranks
    method_names_with_ranks = []
    ranked_method_name_list = []
    sorted_indices = np.argsort(total_score_array)  # Sort indices based on total score
    for idx in sorted_indices:
        rank = overall_rank[idx]
        if 'complexity' in ranking_method or 'three' in ranking_method:
            method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}         in context perplexity: {in_context_ppl_values[idx]:.4f}             Token Length: {token_length_values[idx]:.2f}'
        else:
            method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}         in context perplexity: {in_context_ppl_values[idx]:.4f}            Token Length: {token_length_values[idx]:.2f}'#        IDF: {IDF_values[idx]:.2f}'
        # if not use_token_length:
        #     method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}        similarity: {cos_similarity_values[idx]:.4f}'#        IDF: {IDF_values[idx]:.2f}'
        # else:
        #     method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}        similarity: {cos_similarity_values[idx]:.4f}         Token Length: {token_length_values[idx]:.2f}'#        IDF: {IDF_values[idx]:.2f}'
        ranked_method_name_list.append(name_list[idx])
        log_line += (method_name_with_rank + '\n')
        method_names_with_ranks.append(method_name_with_rank)
        # print(method_name_with_rank)
    method_names_with_ranks_total.append(method_names_with_ranks)

    method = extract_prompt_generation_method(method_names_with_ranks[0])
    if method != 'not_total_method':
        total_layer_best_record_book[task_name] = method

    best_log_line += (method_names_with_ranks[0] + '\n')
    return total_layer_best_record_book, ranked_method_name_list, method_names_with_ranks_total, log_line, best_log_line




def rank_on_metrics(in_context_ppl_values, ppl_values, IDF_values, log_propability_values, skywork_reward_score_values, CAR_score_values, name_list, ranking_method = 'both'):#, 
    if 'in_context_perplexity' in ranking_method:
        total_score_list = in_context_ppl_values
    elif 'calibrated_perplexity' in ranking_method:
        total_score_list = in_context_ppl_values
    elif 'perplexity' in ranking_method:
        total_score_list = ppl_values
    elif 'IDF' in ranking_method:
        total_score_list = IDF_values
    elif 'log_probability' in ranking_method:
        total_score_list = log_propability_values
    elif 'skywork' in ranking_method:
        total_score_list = skywork_reward_score_values
    elif 'CAR' in ranking_method:
        total_score_list = CAR_score_values
    else:
        raise ValueError("Invalid ranking method. Choose 'both', 'perplexity', or 'similarity'.")

    # Rank based on total score
    total_score_array = np.array(total_score_list)
    if 'in_context_perplexity' in ranking_method or 'perplexity' in ranking_method or 'IDF' in ranking_method or 'calibrated_perpelxity' in ranking_method:
        overall_rank = rankdata(total_score_array, method='min')
        sorted_indices = np.argsort(total_score_array)  # Sort indices based on total score
    if 'log_probability' in ranking_method or 'skywork' in ranking_method or 'CAR' in ranking_method:
        overall_rank = rankdata([-x for x in total_score_array], method='max')
        sorted_indices = np.argsort(-total_score_array)
        sorted_indices = np.argsort(total_score_array)[::-1]


    # Prepare method names with ranks
    method_names_with_ranks = []
    ranked_method_name_list = []
    for idx in sorted_indices:
        rank = overall_rank[idx]
        method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}         in context perplexity: {in_context_ppl_values[idx]:.4f}            IDF: {IDF_values[idx]:.2f}           log_probability: {log_propability_values[idx]:.2f}           skywork: {skywork_reward_score_values[idx]:.2f}          CAR: {CAR_score_values[idx]:.2f}'
        ranked_method_name_list.append(name_list[idx])
        method_names_with_ranks.append(method_name_with_rank)
    # method = extract_prompt_generation_method(method_names_with_ranks[0])
    return ranked_method_name_list, method_names_with_ranks



def rank_on_metrics_all(in_context_ppl_values, in_context_ppl_filter_removed_values, ppl_values, IDF_values, log_propability_values, skywork_reward_score_values, CAR_score_values, name_list, ranking_method = 'both'):#, 
    if 'in_context_perplexity' in ranking_method:
        total_score_list = in_context_ppl_values
    elif 'calibrated_perplexity' in ranking_method:
        total_score_list = in_context_ppl_values
    elif 'perplexity' in ranking_method:
        total_score_list = ppl_values
    elif 'IDF' in ranking_method:
        total_score_list = IDF_values
    elif 'log_probability' in ranking_method:
        total_score_list = log_propability_values
    elif 'skywork' in ranking_method:
        total_score_list = skywork_reward_score_values
    elif 'CAR' in ranking_method:
        total_score_list = CAR_score_values
    elif 'ours_filter_removed' in ranking_method:
        total_score_list = in_context_ppl_filter_removed_values
    else:
        raise ValueError("Invalid ranking method. Choose 'both', 'perplexity', or 'similarity'.")

    # Rank based on total score
    total_score_array = np.array(total_score_list)
    if 'in_context_perplexity' in ranking_method or 'ours_filter_removed' in ranking_method or 'perplexity' in ranking_method or 'IDF' in ranking_method or 'calibrated_perpelxity' in ranking_method:
        overall_rank = rankdata(total_score_array, method='min')
        sorted_indices = np.argsort(total_score_array)  # Sort indices based on total score
    if 'log_probability' in ranking_method or 'skywork' in ranking_method or 'CAR' in ranking_method:
        overall_rank = rankdata([-x for x in total_score_array], method='max')
        sorted_indices = np.argsort(-total_score_array)
        sorted_indices = np.argsort(total_score_array)[::-1]

    # Prepare method names with ranks
    method_names_with_ranks = []
    ranked_method_name_list = []
    for idx in sorted_indices:
        rank = overall_rank[idx]
        method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}         in context perplexity: {in_context_ppl_values[idx]:.4f}         in context perplexity remove filter: {in_context_ppl_filter_removed_values[idx]:.4f}            IDF: {IDF_values[idx]:.2f}           log_probability: {log_propability_values[idx]:.2f}           skywork: {skywork_reward_score_values[idx]:.2f}          CAR: {CAR_score_values[idx]:.2f}'
        ranked_method_name_list.append(name_list[idx])
        method_names_with_ranks.append(method_name_with_rank)
    # method = extract_prompt_generation_method(method_names_with_ranks[0])
    return ranked_method_name_list, method_names_with_ranks


def rank_on_score(task_name, ppl_values, in_context_ppl_values, in_context_ppl_plus_values, total_layer_cos_similarity_values, name_list, method_names_with_ranks_total, total_layer_best_record_book, log_line, best_log_line, token_length_values = '', complexity_score_values = '', ranking_method = 'perplexity'):
    
    if 'in_context_perplexity_plus' in ranking_method:
        total_score_list = in_context_ppl_plus_values
    elif 'in_context_perplexity' in ranking_method:
        total_score_list = in_context_ppl_values
    elif 'perplexity' in ranking_method:
        total_score_list = ppl_values
    elif 'similarity' in ranking_method:
        total_score_list = total_layer_cos_similarity_values
    elif 'length' in ranking_method:
        total_score_list = token_length_values
    elif 'complexity' in ranking_method:
        total_score_list = complexity_score_values

    total_score_array = np.array(total_score_list)
    overall_rank = rankdata(total_score_array, method='min')

    method_names_with_ranks = []
    ranked_method_name_list = []
    sorted_indices = np.argsort(total_score_array)  # Sort indices based on total score
    for idx in sorted_indices:
        rank = overall_rank[idx]
        # method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}         in context perplexity: {in_context_ppl_values[idx]:.4f}         in context perplexity plus: {total_score_list[idx]:.4f}        similarity: {total_layer_cos_similarity_values[idx]:.4f}        homogeneity: {complexity_score_values[idx]:.4f}         Token Length: {token_length_values[idx]:.2f}'
        method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        in context perplexity plus: {total_score_list[idx]:.4f}        perplexity: {ppl_values[idx]:.2f}         in context perplexity: {in_context_ppl_values[idx]:.4f}        similarity: {total_layer_cos_similarity_values[idx]:.4f}        homogeneity: {complexity_score_values[idx]:.4f}         Token Length: {token_length_values[idx]:.2f}'
        ranked_method_name_list.append(name_list[idx])
        log_line += (method_name_with_rank + '\n')
        method_names_with_ranks.append(method_name_with_rank)

    method_names_with_ranks_total.append(method_names_with_ranks)

    method = extract_prompt_generation_method(method_names_with_ranks[0])
    total_layer_best_record_book[task_name] = method

    best_log_line += (method_names_with_ranks[0] + '\n')
    return total_layer_best_record_book, ranked_method_name_list, method_names_with_ranks_total, log_line, best_log_line



def rank(task_name, ppl_values, in_context_ppl_values, name_list, method_names_with_ranks_total, total_layer_best_record_book, log_line, best_log_line, IDF_values = '', token_length_values = '', complexity_score_values = '', ranking_method = 'both', perplexity_cap = 10, use_token_length = False):#, initial_prediction_token_length_record = []):
    # if not initial_prediction_token_length_record:
    #     avg_initial_prediction_token_length = sum(initial_prediction_token_length_record)
    #     avg_initial_prediction_token_length = avg_initial_prediction_token_length/len(initial_prediction_token_length_record)
    # Cap perplexity values at 10

    ppl_values = [min(value, perplexity_cap) for value in ppl_values]

    # Proceed with normalization
    ppl_min = min(ppl_values)
    ppl_max = max(ppl_values)



    in_context_ppl_values = [min(value, perplexity_cap) for value in in_context_ppl_values]

    # Proceed with normalization
    in_context_ppl_min = min(in_context_ppl_values)
    in_context_ppl_max = max(in_context_ppl_values)


    


    IDF_min = min(IDF_values)
    IDF_max = max(IDF_values)

    # if use_token_length:
    token_length_min = min(token_length_values)
    token_length_max = max(token_length_values)

    complexity_score_min = min(complexity_score_values)
    complexity_score_max = max(complexity_score_values)

    # Normalize values
    normalized_ppl_list = []
    normalized_in_context_perp_list = []
    normalized_token_length_list = []
    normalized_complexity_list = []
    length_punishment_coefficient_list = []
    IDF_score_list = []
    # for perp, cos_similarity, token_length, IDF in zip(ppl_values, cos_similarity_values, token_length_values, IDF_values):

    for perp, in_context_perp, token_length, complexity_score, IDF_item in zip(ppl_values, in_context_ppl_values, token_length_values, complexity_score_values, IDF_values):
        normalized_perp = 0.0
        normalized_in_context_perp = 0.0
        normalized_token_length = 0.0
        normalized_complexity_score = 0.0
        
        if 'advance' in ranking_method:
            if ppl_max != ppl_min:
                normalized_perp = (perp - ppl_min) / ppl_min
                normalized_perp = min(normalized_perp, 1)  # Cap the value at 1
            if in_context_ppl_max != in_context_ppl_min:
                normalized_in_context_perp = (in_context_perp - in_context_ppl_min) / in_context_ppl_min
                normalized_in_context_perp = min(normalized_in_context_perp, 1)  # Cap the value at 1
            if token_length_max != token_length_min:
                normalized_token_length = (token_length - token_length_min) / token_length_min
                normalized_token_length = min(normalized_token_length, 1)  # Cap the value at 1
            if IDF_max != IDF_min:
                IDF_score = (IDF_item - IDF_min) / IDF_min
            # if complexity_score_max != complexity_score_min:
            #     normalized_complexity_score = (complexity_score - complexity_score_min) / complexity_score_min
            #     normalized_complexity_score = min(normalized_complexity_score, 1)  # Cap the value at 1
        else:
            if ppl_max != ppl_min:
                normalized_perp = (perp - ppl_min) / (ppl_max - ppl_min)
            if in_context_ppl_max != in_context_ppl_min:
                normalized_in_context_perp = (in_context_perp - in_context_ppl_min) / (in_context_ppl_max - in_context_ppl_min)
            if token_length_max != token_length_min:
                normalized_token_length = (token_length - token_length_min) / (token_length_max - token_length_min)
            if IDF_max != IDF_min:
                IDF_score = (IDF_item - IDF_min) / (IDF_max - IDF_min)
            # if complexity_score_max != complexity_score_min:
            #     normalized_complexity_score = (complexity_score - complexity_score_min) / (complexity_score_max - complexity_score_min)
        
        if complexity_score_max != complexity_score_min:
            normalized_complexity_score = (complexity_score - complexity_score_min) / complexity_score_min
            normalized_complexity_score = min(normalized_complexity_score, 1)  # Cap the value at 1
        
        normalized_ppl_list.append(normalized_perp)
        normalized_in_context_perp_list.append(normalized_in_context_perp)
        normalized_token_length_list.append(normalized_token_length)
        normalized_complexity_list.append(normalized_complexity_score)
        IDF_score_list.append(IDF_score)

        # if avg_initial_prediction_token_length:
        #     length_punishment_coefficient = compute_length_punishment_value(avg_initial_prediction_token_length, token_length)
        #     length_punishment_coefficient_list.append(length_punishment_coefficient)


    total_score_list = []
    # for normalized_perp, normalized_cos_similarity, normalized_token_length, normalized_complexity, length_punishment_coefficient in zip(normalized_ppl_list, normalized_cos_similarity_list, normalized_token_length_list, normalized_complexity_list, length_punishment_coefficient_list):
    for normalized_perp, normalized_in_context_perp, normalized_token_length, normalized_complexity, normalized_IDF in zip(normalized_ppl_list, normalized_in_context_perp_list, normalized_token_length_list, normalized_complexity_list, IDF_score_list):#, length_punishment_coefficient_list):
    # for normalized_perp, normalized_cos_similarity in zip(normalized_ppl_list, normalized_cos_similarity_list):
        if 'in_context_perplexity' in ranking_method:
            total_score = normalized_in_context_perp
        elif 'perplexity' in ranking_method:
            total_score = normalized_perp
        elif 'length' in ranking_method:
            total_score = normalized_token_length
        elif 'complexity' in ranking_method:
            total_score = normalized_complexity
        elif 'IDF' in ranking_method:
            total_score = normalized_IDF
        
        # elif ranking_method == 'both_with_length':
        #     total_score = 0.33 * normalized_perp + 0.33 * normalized_cos_similarity + 0.33 * (length_punishment_coefficient)
        else:
            raise ValueError("Invalid ranking method. Choose 'both', 'perplexity', or 'similarity'.")
        total_score_list.append(total_score)

    # Rank based on total score
    total_score_array = np.array(total_score_list)
    overall_rank = rankdata(total_score_array, method='min')

    # Prepare method names with ranks
    method_names_with_ranks = []
    ranked_method_name_list = []
    sorted_indices = np.argsort(total_score_array)  # Sort indices based on total score
    for idx in sorted_indices:
        rank = overall_rank[idx]
        # method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}        similarity: {total_layer_cos_similarity_values[idx]:.4f}'        
        if 'complexity' in ranking_method or 'three' in ranking_method:
            method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}         in context perplexity: {in_context_ppl_values[idx]:.4f}        homogeneity: {complexity_score_values[idx]:.4f}         Token Length: {token_length_values[idx]:.2f}'
        else:
            method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}        in context perplexity: {in_context_ppl_values[idx]:.4f}        homogeneity: {complexity_score_values[idx]:.4f}         Token Length: {token_length_values[idx]:.2f}'#        IDF: {IDF_values[idx]:.2f}'
        # if not use_token_length:
        #     method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}        similarity: {cos_similarity_values[idx]:.4f}'#        IDF: {IDF_values[idx]:.2f}'
        # else:
        #     method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}        similarity: {cos_similarity_values[idx]:.4f}         Token Length: {token_length_values[idx]:.2f}'#        IDF: {IDF_values[idx]:.2f}'
        ranked_method_name_list.append(name_list[idx])
        log_line += (method_name_with_rank + '\n')
        method_names_with_ranks.append(method_name_with_rank)
        # print(method_name_with_rank)
    method_names_with_ranks_total.append(method_names_with_ranks)

    method = extract_prompt_generation_method(method_names_with_ranks[0])
    if method != 'not_total_method':
        total_layer_best_record_book[task_name] = method

    best_log_line += (method_names_with_ranks[0] + '\n')
    return total_layer_best_record_book, ranked_method_name_list, method_names_with_ranks_total, log_line, best_log_line



def rank_on_mix_score_car(sky_ppl_cos_record, name_list, method_names_with_ranks_total, log_line, best_log_line, ranking_method = 'both'):
    skywork_values = []
    CAR_values = []
    for name_key in name_list:
        skywork_values.append(sky_ppl_cos_record[name_key]['skywork_reward_score'])
    
    for name_key in name_list:
        CAR_values.append(sky_ppl_cos_record[name_key]['CAR_score'])

    CAR_score_list = []
    Skywork_SCORE_LIST = []
    for CAR_item, Skywork_item in zip(CAR_values, skywork_values):
        CAR_score_list.append(CAR_item)
        Skywork_SCORE_LIST.append(Skywork_item)
        

    total_score_list = []
    for skywork, car in zip(Skywork_SCORE_LIST, CAR_score_list):
        if 'car' in ranking_method.lower():
            total_score = car
        elif 'skywork' in ranking_method.lower():
            total_score = skywork
        else:
            raise ValueError("Invalid ranking method. Choose 'car', 'skywork'")
        total_score_list.append(total_score)

    # Rank based on total score
    total_score_array = np.array(total_score_list)
    # overall_rank = rankdata(total_score_array, method='max')
    overall_rank = rankdata([-x for x in total_score_array], method='max')
    # overall_rank = rankdata(total_score_array, method='min')


    # Prepare method names with ranks
    method_names_with_ranks = []
    ranked_method_name_list = []
    sorted_indices = np.argsort(total_score_array) [::-1] # Sort indices based on total score
    for idx in sorted_indices:
        rank = overall_rank[idx]
        method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        sky: {Skywork_SCORE_LIST[idx]:.2f}        car: {CAR_score_list[idx]:.4f}'

        ranked_method_name_list.append(name_list[idx])
        log_line += (method_name_with_rank + '\n')
        method_names_with_ranks.append(method_name_with_rank)

    method_names_with_ranks_total.append(method_names_with_ranks)
    best_log_line += (method_names_with_ranks[0] + '\n')
    return ranked_method_name_list, method_names_with_ranks_total, log_line, best_log_line



def rank_on_sky_car(sky_car_dict, name_list, method_names_with_ranks_total, log_line, best_log_line, ranking_method = 'sky'):#, 
   
    Skywork_SCORE_LIST = []
    CAR_score_list = []
    
    for name_key in name_list:
        Skywork_SCORE_LIST.append(sky_car_dict['skywork_reward_score'])
    
    for name_key in name_list:
        CAR_score_list.append(sky_car_dict['CAR_score'])


    total_score_list = []
    # for normalized_perp, normalized_cos_similarity, normalized_token_length, normalized_complexity, length_punishment_coefficient in zip(normalized_ppl_list, normalized_cos_similarity_list, normalized_token_length_list, normalized_complexity_list, length_punishment_coefficient_list):
    for skywork, car in zip(Skywork_SCORE_LIST, CAR_score_list):
 
        if 'car' in ranking_method.lower():
            total_score = car
        elif 'skywork' in ranking_method.lower():
            total_score = skywork
        
        # elif ranking_method == 'both_with_length':
        #     total_score = 0.33 * normalized_perp + 0.33 * normalized_cos_similarity + 0.33 * (length_punishment_coefficient)
        else:
            raise ValueError("Invalid ranking method. Choose 'both', 'perplexity', or 'similarity'.")
        total_score_list.append(total_score)

    # Rank based on total score
    total_score_array = np.array(total_score_list)
    overall_rank = rankdata(total_score_array, method='min')

    # Prepare method names with ranks
    method_names_with_ranks = []
    ranked_method_name_list = []
    sorted_indices = np.argsort(total_score_array)  # Sort indices based on total score
    for idx in sorted_indices:
        rank = overall_rank[idx]
        # method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        perplexity: {ppl_values[idx]:.2f}        similarity: {total_layer_cos_similarity_values[idx]:.4f}'        
        if 'sky' in ranking_method or 'car' in ranking_method:
            method_name_with_rank = f'Rank {int(rank)}: {name_list[idx]}        car: {CAR_score_list[idx]:.2f}        sky: {Skywork_SCORE_LIST[idx]:.4f}'
   
        ranked_method_name_list.append(name_list[idx])
        log_line += (method_name_with_rank + '\n')
        method_names_with_ranks.append(method_name_with_rank)
        # print(method_name_with_rank)
    method_names_with_ranks_total.append(method_names_with_ranks)


    best_log_line += (method_names_with_ranks[0] + '\n')
    return ranked_method_name_list, method_names_with_ranks_total, log_line, best_log_line



def loading_calculated_values(task_name, total_num, ppl_cos_record, only_record_designed_prompt):
    name_list = []
    # Define suffixes for each task
    # suffixes = ['anthropic', 'gpt4', 'mini_gpt4', 'step_by_step', 'human_written_examples', 'provide_gpt4_style_example']
    # if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli']:#, 'plan_bench']:
    #     suffixes.append('in_own_words')
    #     suffixes.append('groundtruth')
    # if task_name in ['plan_bench']:
    #     suffixes.append('in_own_words')

    suffixes = ['claude', 'gpt4', 'mini_gpt4', 'step_by_step', 'openai_human_written_examples', 'gpt4_style_in_context_examples']#, 'simple_response']
    if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli', 'math_geometry', 'math_intermediate']:
        # suffixes.append('rewrite_groundtruth_in_own_words')
        suffixes.append('groundtruth')
    if task_name in ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        # suffixes.append('groundtruth')

    # Collect values and names
    ppl_values = [ppl_cos_record[f'{task_name}_{suffix}']['perplexity'] for suffix in suffixes]
    IDF_values = [ppl_cos_record[f'{task_name}_{suffix}']['IDF_score'] for suffix in suffixes]
    name_list = [f'{task_name}_{suffix}' for suffix in suffixes]
    token_length_values = [ppl_cos_record[f'{task_name}_{suffix}']['average_token_len'] for suffix in suffixes]
    if not only_record_designed_prompt:
        for iii in range(total_num):
            for prompt_api in ['gpt4']:
                for generation_api in ['gpt4', 'anthropic', 'mini_gpt4']:
                    for enable_simple_structure in [True, False]:
                        if enable_simple_structure:
                            simple_suffix = '_simple_prompt'
                        else:
                            simple_suffix = ''
                        ppl_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['perplexity'])
                        IDF_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['IDF_score'])
                        name_list.append(f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}')
                        token_length_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['average_token_len'])
    return ppl_values, IDF_values, name_list, token_length_values


def loading_other_metric_scores(task_name, ppl_cos_record):
    name_list = []

    suffixes = ['claude', 'gpt4', 'mini_gpt4', 'step_by_step', 'openai_human_written_examples', 'gpt4_style_in_context_examples']#, 'simple_response']
    if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli', 'math_geometry', 'math_intermediate']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        suffixes.append('groundtruth')
    if task_name in ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        # suffixes.append('gold_label')

    # Collect values and names
    ppl_values = [ppl_cos_record[f'{task_name}_{suffix}']['perplexity'] for suffix in suffixes]
    IDF_values = [ppl_cos_record[f'{task_name}_{suffix}']['IDF_score'] for suffix in suffixes]
    log_propability_values = [ppl_cos_record[f'{task_name}_{suffix}']['log_propability'] for suffix in suffixes]
    skywork_reward_score_values = [ppl_cos_record[f'{task_name}_{suffix}']['skywork_reward_score'] for suffix in suffixes]
    CAR_score_values = [ppl_cos_record[f'{task_name}_{suffix}']['CAR_score'] for suffix in suffixes]
    name_list = [f'{task_name}_{suffix}' for suffix in suffixes]
    return ppl_values, IDF_values, log_propability_values, skywork_reward_score_values, CAR_score_values, name_list

def loading_icppl_scores(task_name, ppl_cos_record):
    name_list = []

    suffixes = ['claude', 'gpt4', 'mini_gpt4', 'step_by_step', 'openai_human_written_examples', 'gpt4_style_in_context_examples']#, 'simple_response']
    if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli', 'math_geometry', 'math_intermediate']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        suffixes.append('groundtruth')
    if task_name in ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        # suffixes.append('groundtruth')

    # Collect values and names
    ppl_values = [ppl_cos_record[f'{task_name}_{suffix}'] for suffix in suffixes]
    name_list = [f'{task_name}_{suffix}' for suffix in suffixes]
    return ppl_values, name_list




def loading_calculated_values_new(task_name, total_num, ppl_cos_record, in_context_ppl_record, only_record_designed_prompt):
    name_list = []
    # Define suffixes for each task
    # suffixes = ['anthropic', 'gpt4', 'mini_gpt4', 'step_by_step', 'human_written_examples', 'provide_gpt4_style_example']
    # if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli']:#, 'plan_bench']:
    #     suffixes.append('in_own_words')
    #     suffixes.append('groundtruth')
    # if task_name in ['plan_bench']:
    #     suffixes.append('in_own_words')

    suffixes = ['claude', 'gpt4', 'mini_gpt4', 'step_by_step', 'openai_human_written_examples', 'gpt4_style_in_context_examples']#, 'simple_response']
    if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli', 'math_geometry', 'math_intermediate']:
        # suffixes.append('rewrite_groundtruth_in_own_words')
        suffixes.append('groundtruth')
    if task_name in ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        # suffixes.append('groundtruth')

    # Collect values and names
    in_context_ppl_values = [in_context_ppl_record[f'{task_name}_{suffix}']['in_context_perplexity'] for suffix in suffixes]
    if not only_record_designed_prompt:
        for iii in range(total_num):
            for prompt_api in ['gpt4']:
                for generation_api in ['gpt4', 'anthropic', 'mini_gpt4']:
                    for enable_simple_structure in [True, False]:
                        if enable_simple_structure:
                            simple_suffix = '_simple_prompt'
                        else:
                            simple_suffix = ''
                        in_context_ppl_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['perplexity'])
    return in_context_ppl_values


def loading_calculated_values_calibrated_perplexity(task_name, total_num, ppl_cos_record, in_context_ppl_record, only_record_designed_prompt):
    name_list = []

    suffixes = ['claude', 'gpt4', 'mini_gpt4', 'step_by_step', 'openai_human_written_examples', 'gpt4_style_in_context_examples']#, 'simple_response']
    if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli', 'math_geometry', 'math_intermediate']:
        # suffixes.append('rewrite_groundtruth_in_own_words')
        suffixes.append('groundtruth')
    if task_name in ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        # suffixes.append('groundtruth')

    # Collect values and names
    ppl_values = [ppl_cos_record[f'{task_name}_{suffix}']['perplexity'] for suffix in suffixes]
    in_context_ppl_values = [in_context_ppl_record[f'{task_name}_{suffix}']['in_context_perplexity'] for suffix in suffixes]
    IDF_values = [ppl_cos_record[f'{task_name}_{suffix}']['IDF_score'] for suffix in suffixes]
    name_list = [f'{task_name}_{suffix}' for suffix in suffixes]
    token_length_values = [ppl_cos_record[f'{task_name}_{suffix}']['average_token_len'] for suffix in suffixes]
    complexity_score_values = [ppl_cos_record[f'{task_name}_{suffix}']['complexity_score'] for suffix in suffixes]
    if not only_record_designed_prompt:
        for iii in range(total_num):
            for prompt_api in ['gpt4']:
                for generation_api in ['gpt4', 'anthropic', 'mini_gpt4']:
                    for enable_simple_structure in [True, False]:
                        if enable_simple_structure:
                            simple_suffix = '_simple_prompt'
                        else:
                            simple_suffix = ''

                        ppl_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['perplexity'])

                        in_context_ppl_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['perplexity'])

                        
                        IDF_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['IDF_score'])
                        name_list.append(f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}')
                        token_length_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['average_token_len'])
                        complexity_score_values.append(ppl_cos_record[f'{task_name.lower()}_total_{iii}_prompt_api_{prompt_api}_generation_api_{generation_api}{simple_suffix}']['complexity_score'])
    return ppl_values, in_context_ppl_values, IDF_values, name_list, token_length_values, complexity_score_values


def loading_calculated_values_plus(task_name, in_context_ppl_plus_record):
    suffixes = ['claude', 'gpt4', 'mini_gpt4', 'step_by_step', 'openai_human_written_examples', 'gpt4_style_in_context_examples']#, 'simple_response']
    if task_name in ['gsm8k', 'math_algebra', 'ecqa', 'esnli', 'math_geometry', 'math_intermediate']:
        # suffixes.append('rewrite_groundtruth_in_own_words')
        suffixes.append('groundtruth')
    if task_name in ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization']:
        suffixes.append('rewrite_groundtruth_in_own_words')
        # suffixes.append('groundtruth')

    # Collect values and names
    in_context_ppl_values = [in_context_ppl_plus_record[f'{task_name}_{suffix}']['in_context_perplexity_plus'] for suffix in suffixes]
    return in_context_ppl_values




def calculate_spearman(rank_x, rank_y):
    """
    Calculate the Spearman correlation coefficient given pre-sorted ranks.
    
    Parameters:
    rank_x: List of ranks for variable X
    rank_y: List of ranks for variable Y

    Returns:
    Spearman correlation coefficient (rho)
    """
    # Ensure the lengths match
    if len(rank_x) != len(rank_y):
        raise ValueError("Rank lists must have the same length")
    
    n = len(rank_x)  # Number of observations
    
    # Calculate the differences in ranks
    d = [rank_x[i] - rank_y[i] for i in range(n)]
    d_squared = [di ** 2 for di in d]
    
    # Apply the Spearman correlation formula
    rho = 1 - (6 * sum(d_squared)) / (n * (n**2 - 1))
    
    return rho

# def calc_spearman_coefficient(gt_accuracy_record, name_list, task_name):
#     """
#     For each (dataset+model) in data_dict, sort its methods by descending accuracy
#     and print the ranking.
#     """
#     # data_dict = data_dict[task_name]
#     key_name_list = []
#     for name_item in name_list:
#         temp = task_name + '_'
#         name_item = name_item.replace(temp, '')
#         key_name_list.append(name_item)
#     rank_response = range(1, len(key_name_list)+1)
#     rank_gt = [0] * len(key_name_list)

#     # 1. Filter out methods that have None values
#     valid_methods = [(m, s) for m, s in gt_accuracy_record.items() if s is not None]

#     # 2. Sort by score descending
#     valid_methods.sort(key=lambda x: x[1], reverse=True)
    
#     i = 0
#     for key_name, value in valid_methods:
#         for index, key_name_item in enumerate(key_name_list):
#             if key_name_item == key_name:
#                 rank_gt[index] = i + 1
#                 i += 1
#                 break

#     rho = calculate_spearman(rank_gt, rank_response)

#     # print(f"Spearman correlation coefficient: {rho:.4f}")
    
#     # # 3. Print top results
#     # print(f"--- {task_name} ---")
#     # if not valid_methods:
#     #     print("  (No numeric scores available)")
#     # else:
#     #     for rank, (method, score) in enumerate(valid_methods, start=1):
#     #         print(f"  {rank}. {method}: {score}")
#     # print()


#     # total_dict = {}
#     # total_dict[task_name] = valid_methods
#     return rho



def calc_spearman_coefficient(gt_accuracy_record, name_list, task_name):
    """
    For each (dataset+model) in data_dict, sort its methods by descending accuracy
    and print the ranking.
    """
    # data_dict = data_dict[task_name]
    key_name_list = []
    for name_item in name_list:
        temp = task_name + '_'
        name_item = name_item.replace(temp, '')
        key_name_list.append(name_item)
    
    

    # 1. Filter out methods that have None values
    valid_methods = [(m, s) for m, s in gt_accuracy_record.items() if s is not None]

    # 2. Sort by score descending
    valid_methods.sort(key=lambda x: x[1], reverse=True)

    key_name_list_temp = []
    valid_name_list = []
    
    for key_name, value in valid_methods:
        valid_name_list.append(key_name)
    for key_name in key_name_list:
        if key_name in valid_name_list:
            key_name_list_temp.append(key_name)
    key_name_list = key_name_list_temp
    rank_response = range(1, len(key_name_list)+1)


    gt_accuracy_record_detail = []
    chosen_method_accuracy = 0
    current_ccc = 99999999
    for key_name, value in valid_methods:
        if key_name in key_name_list:
            for ccc, kkk in enumerate(key_name_list):
                if kkk == key_name:
                    gt_accuracy_record_detail.append(f'{key_name}     {value}          strategy rank: {ccc + 1}')
                    if len(gt_accuracy_record_detail) == 1:
                        accuracy_best = value
                    if ccc < current_ccc:
                        current_ccc = ccc
                        chosen_method_accuracy = value

    rank_gt = [0] * len(key_name_list)

    i = 0
    previous_value = None  # 初始化 previous_value
    previous_rank_indices = []  # 存储需要更新的排名在 key_name_list 中的索引

    for key_name, value in valid_methods:
        for index, key_name_item in enumerate(key_name_list):
            if key_name_item == key_name:
                current_rank = i + 1  # 当前排名

                if previous_value != value:
                    # 如果之前有同分的方法，需要计算平均排名
                    if previous_rank_indices:
                        avg_rank = sum(rank_gt[idx] for idx in previous_rank_indices) / len(previous_rank_indices)
                        for idx in previous_rank_indices:
                            rank_gt[idx] = avg_rank
                        previous_rank_indices = []
                    
                    # 赋予当前方法的排名
                    rank_gt[index] = current_rank
                    previous_rank_indices.append(index)
                else:
                    # 同分的方法，赋予当前排名
                    rank_gt[index] = current_rank
                    previous_rank_indices.append(index)

                i += 1
                previous_value = value
                break

    # 处理最后一组同分的方法
    if previous_rank_indices:
        avg_rank = sum(rank_gt[idx] for idx in previous_rank_indices) / len(previous_rank_indices)
        for idx in previous_rank_indices:
            rank_gt[idx] = avg_rank
    
    rho = calculate_spearman(rank_gt, rank_response)
    return rho, gt_accuracy_record_detail, key_name_list, accuracy_best, chosen_method_accuracy



def calc_spearman_coefficient_3(gt_accuracy_record, gt_accuracy_record_1, gt_accuracy_record_2, name_list, task_name):
    """
    For each (dataset+model) in data_dict, sort its methods by descending accuracy
    and print the ranking.
    """
    # data_dict = data_dict[task_name]
    key_name_list = []
    for name_item in name_list:
        temp = task_name + '_'
        name_item = name_item.replace(temp, '')
        key_name_list.append(name_item)

    key_list = gt_accuracy_record.keys()
    for key_name in key_list:
        value = gt_accuracy_record[key_name]
        value_1 = gt_accuracy_record_1[key_name]
        value_2 = gt_accuracy_record_2[key_name]

        value_num = 0
        count = 0
        if value:
            if float(value) > 0:
                count += 1
                value_num += float(value)
        if value_1:
            if float(value_1) > 0:
                count += 1
                value_num += float(value_1)
        if value_2:
            if float(value_2) > 0:
                count += 1
                value_num += float(value_2)
        value_num /= count
        gt_accuracy_record[key_name] = str(value_num)

    # 1. Filter out methods that have None values
    valid_methods = [(m, s) for m, s in gt_accuracy_record.items() if s is not None]

    # 2. Sort by score descending
    valid_methods.sort(key=lambda x: x[1], reverse=True)

    key_name_list_temp = []
    valid_name_list = []
    
    for key_name, value in valid_methods:
        valid_name_list.append(key_name)
    for key_name in key_name_list:
        if key_name in valid_name_list:
            key_name_list_temp.append(key_name)
    key_name_list = key_name_list_temp
    rank_response = range(1, len(key_name_list)+1)

    gt_accuracy_record_detail = []
    chosen_method_accuracy = 0
    current_ccc = 99999999
    for key_name, value in valid_methods:
        if key_name in key_name_list:
            for ccc, kkk in enumerate(key_name_list):
                if kkk == key_name:
                    gt_accuracy_record_detail.append(f'{key_name}     {value}          strategy rank: {ccc + 1}')
                    if len(gt_accuracy_record_detail) == 1:
                        accuracy_best = value
                    if ccc < current_ccc:
                        current_ccc = ccc
                        chosen_method_accuracy = value

    rank_gt = [0] * len(key_name_list)

    i = 0
    previous_value = None  # 初始化 previous_value
    previous_rank_indices = []  # 存储需要更新的排名在 key_name_list 中的索引

    for key_name, value in valid_methods:
        for index, key_name_item in enumerate(key_name_list):
            if key_name_item == key_name:
                current_rank = i + 1  # 当前排名

                if previous_value != value:
                    # 如果之前有同分的方法，需要计算平均排名
                    if previous_rank_indices:
                        avg_rank = sum(rank_gt[idx] for idx in previous_rank_indices) / len(previous_rank_indices)
                        for idx in previous_rank_indices:
                            rank_gt[idx] = avg_rank
                        previous_rank_indices = []
                    
                    # 赋予当前方法的排名
                    rank_gt[index] = current_rank
                    previous_rank_indices.append(index)
                else:
                    # 同分的方法，赋予当前排名
                    rank_gt[index] = current_rank
                    previous_rank_indices.append(index)

                i += 1
                previous_value = value
                break

    # 处理最后一组同分的方法
    if previous_rank_indices:
        avg_rank = sum(rank_gt[idx] for idx in previous_rank_indices) / len(previous_rank_indices)
        for idx in previous_rank_indices:
            rank_gt[idx] = avg_rank
    
    rho = calculate_spearman(rank_gt, rank_response)
    return rho, gt_accuracy_record_detail, key_name_list, accuracy_best, chosen_method_accuracy



def save_best_varient(model_name, layerwise_best_record_book, method_names_with_ranks_total, use_token_length, only_record_designed_prompt):
    if use_token_length:
        fill_in = ''
    else:
        fill_in = 'not_'
    best_record_book_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/best_total_record_book/best_{model_name}_{fill_in}use_token_length.json'
    best_record_best_record_book_path_designed_prompt_only = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/best_record_book_not_include_total_varient/best_{model_name}_{fill_in}use_token_length_designed_prompt_only.json'
    
    if not only_record_designed_prompt:
        with open(best_record_book_path, 'w') as f:
            json.dump(layerwise_best_record_book, f, indent=4)
    else:
        with open(best_record_best_record_book_path_designed_prompt_only, 'w') as f:
            json.dump(method_names_with_ranks_total, f, indent=4)





def save_best_varient_log(model_name, log_line, best_log_line, use_token_length, only_record_designed_prompt):
    if use_token_length:
        fill_in = ''
    else:
        fill_in = 'not_'
    log_file_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/log/{model_name}_{fill_in}use_token_length.txt'
    best_log_file_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/Mix_Score_Ranking_Calculation/log/best_{model_name}_{fill_in}use_token_length.txt'
    
    if not only_record_designed_prompt:
        with open(log_file_path, 'w') as f:
            log_line += '\n\n\n'
            f.write(log_line)
        with open(best_log_file_path, 'w') as f:
            best_log_line += '\n\n\n'
            f.write(best_log_line)