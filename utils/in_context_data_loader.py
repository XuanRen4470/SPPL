import sys
import os
import json
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from config.config import HOME_DIRECTORY
import copy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import json
from utils.data_loader import load_GSM8K, load_MATH, load_ESNLI, load_PIQA, load_BOOLQ, load_MMLU, load_AGIEVAL, load_ECQA, load_SQUAD, load_API_BANK, load_WINOGRANDE, load_DROP, load_plan_bench_with_proportion, load_MBPP
from utils.data_loader_in_context import random_select_in_context_learning_examples, initial_prediction_random_select_in_context_learning_examples
from config.modify_config_on_current_job import set_config
import random
import os
import math


def process_data_list(data_list, prompt_style = '', test_task_name = '',sub_samples_num_list = '', end_template = '', use_in_context_learning = False, n_train = 1, test_idx = -1, vairation_num = -1, api_type = 'mini_gpt4', prompt_api = 'gpt4', total_use_simple_structure = False, correct_index_list = []):#, plot_record_mispredicted_samples = False):
    for i, item in enumerate(data_list):
        prompt_style = prompt_style
        question_item = item['question']
        if 'original_question' in item:
            original_question = item['original_question']
        else:
            original_question = item['question']

        if use_in_context_learning:
            formated_question = random_select_in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = test_task_name.lower(), sub_samples_num_list = sub_samples_num_list[i], end_template = end_template, vairation_num = vairation_num, api_type = api_type, prompt_api = prompt_api, total_use_simple_structure = total_use_simple_structure, correct_index_list = correct_index_list)
        else:
            formated_question = question_item
        data_list[i]['question'] = formated_question
        data_list[i]['original_question'] = original_question
    data_list = data_list[:n_train]

    # if not correct_index_list:
    #     in_own_words_data_list = in_own_words_data_list[:n_train]
    # else:
    #     in_own_words_data_list = [in_own_words_data_list[i] for i in correct_index_list]

    if test_idx != -1:
        data_list = [data_list[test_idx]]
    return data_list

def initial_prediction_in_context_process_data_list(data_list, initial_prediction_list, prompt_style = '', test_task_name = '',sub_samples_num_list = '', end_template = '', n_train = 1, test_idx = -1, vairation_num = -1, api_type = 'mini_gpt4', prompt_api = 'gpt4', total_use_simple_structure = False, correct_index_list = []):#, plot_record_mispredicted_samples = False):
    for i, item in enumerate(data_list):
        prompt_style = prompt_style
        question_item = item['question']
        if 'original_question' in item:
            original_question = item['original_question']
        else:
            original_question = item['question']

        formated_question = initial_prediction_random_select_in_context_learning_examples(question_item, original_question, initial_prediction_list, prompt_style = prompt_style, task = test_task_name.lower(), sub_samples_num_list = sub_samples_num_list[i], end_template = end_template, vairation_num = vairation_num, api_type = api_type, prompt_api = prompt_api, total_use_simple_structure = total_use_simple_structure, correct_index_list = correct_index_list)

        data_list[i]['question'] = formated_question
        data_list[i]['original_question'] = original_question
    data_list = data_list[:n_train]

    # if not correct_index_list:
    #     in_own_words_data_list = in_own_words_data_list[:n_train]
    # else:
    #     in_own_words_data_list = [in_own_words_data_list[i] for i in correct_index_list]

    if test_idx != -1:
        data_list = [data_list[test_idx]]
    return data_list

def perplexity_calculation_in_context_data_loader(train_task_name, n_train, use_in_context_learning, test_idx, end_template, correct_index_list = []):
    train_task_name = train_task_name.lower()

    if train_task_name.lower() == 'gsm8k':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/train_filtered.json'
        train_data_list = load_GSM8K(train_path, 1000, zeroshot = True)
    if 'math_algebra' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_ALGEBRA/train_algebra_total_filtered.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = True)
    if 'math_geometry' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_GEOMETRY/groundtruth.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = True)
    if 'math_intermediate_algebra' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_INTERMEDIATE_ALGEBRA/groundtruth.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = True)
    if train_task_name.lower() == 'esnli':
        train_path = f'{HOME_DIRECTORY}/dataset/ESNLI/train.json'
        train_data_list = load_ESNLI(train_path, 1000)
    if train_task_name.lower() == 'ecqa':
        train_path = f'{HOME_DIRECTORY}/dataset/ECQA/train.json'
        train_data_list = load_ECQA(train_path, 1000, use_gt_rationale = True)
    if 'plan_bench' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as f:
            train_data_list = json.load(f)
    if train_task_name.lower() == 'boolq':
        train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
        train_data_list = load_BOOLQ(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'mmlu':
        train_path = f'{HOME_DIRECTORY}/dataset/MMLU/groundtruth.json'
        train_data_list = load_MMLU(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'mmlu_moral_scenarios':
        train_path = f'{HOME_DIRECTORY}/dataset/MMLU_MORAL_SCENARIOS/groundtruth.json'
        train_data_list = load_MMLU(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'winogrande':
        train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
        train_data_list = load_WINOGRANDE(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'piqa':
        train_path = f'{HOME_DIRECTORY}/dataset/PIQA/train.json'
        train_data_list = load_PIQA(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'squad':
        train_path = f'{HOME_DIRECTORY}/dataset/SQUAD/train.json'
        train_data_list = load_SQUAD(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'agieval':
        train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/train.json'
        train_data_list = load_AGIEVAL(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'api_bank':
        train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-api-train.json'
        train_data_list = load_API_BANK(train_path, 1000)
    if train_task_name.upper() =='DROP':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        train_data_list = load_DROP(train_path, 1000, finetune_with_gt=True)
    elif train_task_name.upper() == 'MBPP':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        train_data_list = load_MBPP(train_path, 1000)



    if train_task_name.lower() == 'mmlu_pro' or train_task_name.lower() == 'arc_challenge' or train_task_name.lower() == 'hellaswag' or train_task_name.lower() == 'theoremqa' or train_task_name.lower() == 'mmlu_pro_law':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:1000]

    if not correct_index_list:
        train_data_list = train_data_list[:n_train]
    else:
        train_data_list = [train_data_list[i] for i in correct_index_list]

    # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/mistral_minimum_change.json'
    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4.json'
    anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/claude.json'
    mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/mini_gpt4.json'
    step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/gpt4_generated_step_by_step_1000.json'
    provide_gpt4_style_example_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_provide_gpt4_example_1000.json'
    human_written_examples_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_human_written_examples.json'
    # simple_response_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/simple_response.json'
    
    
    if train_task_name.lower() in 'esnli':
        gold_label_data_list = load_ESNLI(train_path, 1000, use_gold_label = True)
        if not correct_index_list:
            gold_label_data_list = gold_label_data_list[:n_train]
        else:
            gold_label_data_list = [gold_label_data_list[i] for i in correct_index_list]
        redundant_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_generated_{train_task_name.lower()}_redundant_1000.json'
    
    if train_task_name.lower() in 'ecqa':
        gold_label_data_list = load_ECQA(train_path, n_train, finetune = True, use_gt_rationale = False)
        if not correct_index_list:
            gold_label_data_list = gold_label_data_list[:n_train]
        else:
            gold_label_data_list = [gold_label_data_list[i] for i in correct_index_list]
        redundant_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_generated_{train_task_name.lower()}_redundant_1000.json'

    # if 'plan_bench' in train_task_name.lower():
    #     rewrite_in_natural_language_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_generated_in_natural_language_1000.json'
    #     with open(rewrite_in_natural_language_data_train_path, 'r') as file:
    #         rewrite_in_natural_language_data_list = json.load(file)
    #     if not correct_index_list:
    #         rewrite_in_natural_language_data_list = rewrite_in_natural_language_data_list[:n_train]
    #     else:
    #         rewrite_in_natural_language_data_list = [rewrite_in_natural_language_data_list[i] for i in correct_index_list]
    #     original_rewrite_in_natural_language_data_list = copy.deepcopy(rewrite_in_natural_language_data_list)
    
    if train_task_name.lower() == 'ecqa' or train_task_name.lower() == 'esnli':
        with open(redundant_data_train_path, 'r') as file:
            redundant_data_train_data_list = json.load(file)
        if not correct_index_list:
            redundant_data_train_data_list = redundant_data_train_data_list[:n_train]
        else:
            redundant_data_train_data_list = [redundant_data_train_data_list[i] for i in correct_index_list]
        original_redundant_data_train_data_list = copy.deepcopy(redundant_data_train_data_list)

    # if test_task_name.lower() != 'api_bank' and test_task_name.lower() != 'plan_bench' and test_task_name.lower() != 'mmlu' and test_task_name.lower() != 'winogrande' and test_task_name.lower() != 'piqa' and test_task_name.lower() != 'agieval' and test_task_name.lower() != 'squad':
    #     with open(minimum_change_train_path, 'r') as file:
    #         minimum_change_train_data_list = json.load(file)
    #     minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
    #     original_minimum_change_train_data_list = copy.deepcopy(minimum_change_train_data_list)
    with open(human_written_examples_data_train_path, 'r') as file:
        human_written_examples_data_list = json.load(file)
    if not correct_index_list:
        human_written_examples_data_list = human_written_examples_data_list[:n_train]
    else:
        human_written_examples_data_list = [human_written_examples_data_list[i] for i in correct_index_list]

    # with open(simple_response_data_train_path, 'r') as file:
    #     simple_response_data_list = json.load(file)
    # if not correct_index_list:
    #     simple_response_data_list = simple_response_data_list[:n_train]
    # else:
    #     simple_response_data_list = [simple_response_data_list[i] for i in correct_index_list]
        
    with open(gpt4_generated_data_train_path, 'r') as file:
        gpt4_generated_train_data_list = json.load(file)
    if not correct_index_list:
        gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
    else:
        gpt4_generated_train_data_list = [gpt4_generated_train_data_list[i] for i in correct_index_list]
    gpt4_prediction_list = []
    for item in gpt4_generated_train_data_list:
        gpt4_prediction_list.append(item['answer'])

    with open(anthropic_generated_data_train_path, 'r') as file:
        anthropic_data_list = json.load(file)
    if not correct_index_list:
        anthropic_data_list = anthropic_data_list[:n_train]
    else:
        anthropic_data_list = [anthropic_data_list[i] for i in correct_index_list]

    with open(mini_gpt4_generated_data_train_path, 'r') as file:
        mini_gpt4_data_list = json.load(file)
    if not correct_index_list:
        mini_gpt4_data_list = mini_gpt4_data_list[:n_train]
    else:
        mini_gpt4_data_list = [mini_gpt4_data_list[i] for i in correct_index_list]

    with open(step_by_step_data_train_path, 'r') as file:
        step_by_step_data_list = json.load(file)
    if not correct_index_list:
        step_by_step_data_list = step_by_step_data_list[:n_train]
    else:
        step_by_step_data_list = [step_by_step_data_list[i] for i in correct_index_list]
    
    with open(provide_gpt4_style_example_data_train_path, 'r') as file:
        provide_gpt4_style_example_data_list = json.load(file)
    if not correct_index_list:
        provide_gpt4_style_example_data_list = provide_gpt4_style_example_data_list[:n_train]
    else:
        provide_gpt4_style_example_data_list = [provide_gpt4_style_example_data_list[i] for i in correct_index_list]
    
    if 'boolq' not in train_task_name.lower() and 'api_bank' not in train_task_name.lower() and 'mmlu' not in train_task_name.lower() and 'winogrande' not in train_task_name.lower() and 'piqa' not in train_task_name.lower() and 'agieval' not in train_task_name.lower() and 'squad' not in train_task_name.lower() and 'mmlu_pro' not in train_task_name.lower() and 'mmlu_pro_law' not in train_task_name.lower() and 'arc_challenge' not in train_task_name.lower() and 'hellaswag' not in train_task_name.lower() and 'theoremqa' not in train_task_name.lower() and 'drop' not in train_task_name.lower():
        in_own_words_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        with open(in_own_words_train_path, 'r') as file:
            in_own_words_data_list = json.load(file)
        if not correct_index_list:
            in_own_words_data_list = in_own_words_data_list[:n_train]
        else:
            in_own_words_data_list = [in_own_words_data_list[i] for i in correct_index_list]
        original_in_own_words_data_list = copy.deepcopy(in_own_words_data_list)
    
    original_train_data_list = copy.deepcopy(train_data_list)
    original_mini_gpt4_data_list = copy.deepcopy(mini_gpt4_data_list)
    original_anthropic_data_list = copy.deepcopy(anthropic_data_list)
    original_gpt4_generated_train_data_list = copy.deepcopy(gpt4_generated_train_data_list)
    original_step_by_step_data_list = copy.deepcopy(step_by_step_data_list)
    original_provide_gpt4_style_example_data_list = copy.deepcopy(provide_gpt4_style_example_data_list)
    original_human_written_examples_data_list = copy.deepcopy(human_written_examples_data_list)
    # original_simple_response_data_list = copy.deepcopy(simple_response_data_list)

    if 'ecqa' in train_task_name.lower() or 'esnli' in train_task_name.lower():
        original_gold_label_data_list = copy.deepcopy(gold_label_data_list)
    
    def generate_random_lists(num_lists, sample_num=3, sub_sample_num=10):
        random_lists = []
        for i in range(num_lists):
            # Create a value range excluding the current index 'i'
            value_range = [x for x in range(sub_sample_num) if x != i]
            random_list = random.sample(value_range, sample_num)
            random_lists.append(random_list)
        return random_lists
    
    if 'BOOLQ' in train_task_name.upper():
        sample_num = 6
    elif 'PLAN_BENCH' in train_task_name.upper() or 'API_BANK' in train_task_name.upper():
        sample_num = 1
    else:
        sample_num = 3
    sub_sample_num = 10
    
    sub_samples_num_list = generate_random_lists(n_train, sample_num = sample_num, sub_sample_num = sub_sample_num)
    
    # if 'plan_bench' in train_task_name.lower():
    #     rewrite_in_natural_language_data_list = process_data_list(rewrite_in_natural_language_data_list, prompt_style = 'rewrite_in_natural_language', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)


    # if test_task_name.lower() != 'api_bank' and test_task_name.lower() != 'plan_bench' and test_task_name.lower() != 'mmlu' and test_task_name.lower() != 'winogrande' and test_task_name.lower() != 'piqa' and test_task_name.lower() != 'agieval' and test_task_name.lower() != 'squad':

    #     minimum_change_train_data_list = process_data_list(minimum_change_train_data_list, prompt_style = 'minimum_change', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx)

        # if plot_record_mispredicted_samples:
        #     minimum_change_train_data_list = minimum_change_train_data_list_filtered[:n_train]
        # else:
        #     minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
    human_written_examples_data_list = process_data_list(human_written_examples_data_list, prompt_style = 'human_written_examples', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    provide_gpt4_style_example_data_list = process_data_list(provide_gpt4_style_example_data_list, prompt_style = 'provide_gpt4_style_example', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    # if test_task_name.lower() != 'code' and test_task_name.lower() != 'mbpp':
    step_by_step_data_list = process_data_list(step_by_step_data_list, prompt_style = 'step_by_step', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    train_data_list = process_data_list(train_data_list, prompt_style = 'gt_style', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)
    
    gpt4_generated_train_data_list = process_data_list(gpt4_generated_train_data_list, prompt_style = 'gpt4', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    anthropic_data_list = process_data_list(anthropic_data_list, prompt_style = 'anthropic', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    mini_gpt4_data_list = process_data_list(mini_gpt4_data_list, prompt_style = 'mini_gpt4', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    # simple_response_data_list = process_data_list(simple_response_data_list, prompt_style = 'simple_response', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    # if test_task_name.lower() == 'esnli' or test_task_name.lower() == 'boolq':
    #     mini_gpt4_style_data_list = process_data_list(mini_gpt4_style_data_list, prompt_style = 'mini_gpt4_style', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx)

    if train_task_name.lower() == 'ecqa' or train_task_name.lower() == 'esnli':
        gold_label_data_list = process_data_list(gold_label_data_list, prompt_style = 'gold_label', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    if 'boolq' not in train_task_name.lower() and 'api_bank' not in train_task_name.lower() and 'mmlu' not in train_task_name.lower() and 'winogrande' not in train_task_name.lower() and 'piqa' not in train_task_name.lower() and 'agieval' not in train_task_name.lower() and 'squad' not in train_task_name.lower() and 'mmlu_pro' not in train_task_name.lower() and 'mmlu_pro_law' not in train_task_name.lower() and 'arc_challenge' not in train_task_name.lower() and 'hellaswag' not in train_task_name.lower() and 'theoremqa' not in train_task_name.lower() and 'drop' not in train_task_name.lower():
        in_own_words_data_list = process_data_list(in_own_words_data_list, prompt_style = 'in_own_words', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    if 'esnli' in train_task_name.lower() or 'ecqa' in train_task_name.lower():
        redundant_data_train_data_list = process_data_list(redundant_data_train_data_list, prompt_style = 'redundant', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    current_task_name = train_task_name.lower()
    train_config, test_config = set_config(current_task_name, 0, 0, model_name = 'mistral')

    if 'api_bank' in train_task_name.lower() or 'plan_bench' in train_task_name.lower():
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 16
    else:
        per_device_train_batch_size = train_config['per_device_train_batch_size']
        per_device_train_batch_size = math.floor(per_device_train_batch_size / 2)
        gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        gradient_accumulation_steps *= 2

    train_config['per_device_train_batch_size'] = per_device_train_batch_size
    train_config['gradient_accumulation_steps'] = gradient_accumulation_steps
    
    test_config['max_length'] = 4024
    test_config['max_input_length'] = 3000
    test_config['max_new_tokens'] = 1024
    test_config['per_device_eval_batch_size'] = 2

    # dataset_list = [['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['claude', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['gpt4_style_in_context_examples', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['openai_human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, '']]#, ['simple_response', simple_response_data_list, simple_response_data_train_path, original_simple_response_data_list, '']]

    # dataset_list = [['gpt4_style_in_context_examples', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['claude', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['openai_human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, '']]

    dataset_list = [['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['claude', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['gpt4_style_in_context_examples', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['openai_human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, '']]


    if 'ecqa' in train_task_name.lower() or 'esnli' in train_task_name.lower() or 'plan_bench' in train_task_name.lower() or 'gsm8k' in train_task_name.lower() or 'math' in train_task_name.lower() or 'mbpp' in train_task_name.lower():
        dataset_list.append(['rewrite_groundtruth_in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, ''])


    # dataset_list = [['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['claude', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]
    return dataset_list, train_config, test_config, train_task_name, gpt4_prediction_list


def initial_prediction_in_context_data_loader(train_task_name, n_train, test_idx, end_template, initial_prediction_list, correct_index_list):
    train_task_name = train_task_name.lower()

    if train_task_name.lower() == 'gsm8k':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/train_filtered.json'
        train_data_list = load_GSM8K(train_path, 1000, zeroshot = True)
    if 'math_algebra' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_ALGEBRA/train_algebra_total_filtered.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = True)
    if 'math_geometry' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_GEOMETRY/groundtruth.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = True)
    if 'math_intermediate_algebra' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_INTERMEDIATE_ALGEBRA/groundtruth.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = True)
    if train_task_name.lower() == 'esnli':
        train_path = f'{HOME_DIRECTORY}/dataset/ESNLI/train.json'
        train_data_list = load_ESNLI(train_path, 1000)
    if train_task_name.lower() == 'ecqa':
        train_path = f'{HOME_DIRECTORY}/dataset/ECQA/train.json'
        train_data_list = load_ECQA(train_path, 1000, use_gt_rationale = True)
    if 'plan_bench' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as f:
            train_data_list = json.load(f)
    if train_task_name.lower() == 'boolq':
        train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
        train_data_list = load_BOOLQ(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'mmlu':
        train_path = f'{HOME_DIRECTORY}/dataset/MMLU/groundtruth.json'
        train_data_list = load_MMLU(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'winogrande':
        train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
        train_data_list = load_WINOGRANDE(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'piqa':
        train_path = f'{HOME_DIRECTORY}/dataset/PIQA/train.json'
        train_data_list = load_PIQA(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'squad':
        train_path = f'{HOME_DIRECTORY}/dataset/SQUAD/train.json'
        train_data_list = load_SQUAD(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'agieval':
        train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/train.json'
        train_data_list = load_AGIEVAL(train_path, 1000, finetune = True)
    if train_task_name.lower() == 'api_bank':
        train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-api-train.json'
        train_data_list = load_API_BANK(train_path, 1000)
    if train_task_name.upper() =='DROP':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        train_data_list = load_DROP(train_path, 1000, finetune_with_gt=True)
    elif train_task_name.upper() == 'MBPP':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        train_data_list = load_MBPP(train_path, 1000)



    if train_task_name.lower() == 'mmlu_pro' or train_task_name.lower() == 'mmlu_pro_law' or train_task_name.lower() == 'arc_challenge' or train_task_name.lower() == 'hellaswag' or train_task_name.lower() == 'theoremqa':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:1000]

    if not correct_index_list:
        train_data_list = train_data_list[:n_train]
    else:
        train_data_list = [train_data_list[i] for i in correct_index_list]

    # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/mistral_minimum_change.json'
    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4.json'
    anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/claude.json'
    mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/mini_gpt4.json'
    step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/gpt4_generated_step_by_step_1000.json'
    provide_gpt4_style_example_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_provide_gpt4_example_1000.json'
    human_written_examples_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_human_written_examples.json'
    simple_response_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/simple_response.json'
    
    
    if train_task_name.lower() in 'esnli':
        gold_label_data_list = load_ESNLI(train_path, 1000, use_gold_label = True)
        if not correct_index_list:
            gold_label_data_list = gold_label_data_list[:n_train]
        else:
            gold_label_data_list = [gold_label_data_list[i] for i in correct_index_list]
        redundant_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_generated_{train_task_name.lower()}_redundant_1000.json'
    
    if train_task_name.lower() in 'ecqa':
        gold_label_data_list = load_ECQA(train_path, n_train, finetune = True, use_gt_rationale = False)
        if not correct_index_list:
            gold_label_data_list = gold_label_data_list[:n_train]
        else:
            gold_label_data_list = [gold_label_data_list[i] for i in correct_index_list]
        redundant_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_generated_{train_task_name.lower()}_redundant_1000.json'

    # if 'plan_bench' in train_task_name.lower():
    #     rewrite_in_natural_language_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/openai_gpt4_generated_in_natural_language_1000.json'
    #     with open(rewrite_in_natural_language_data_train_path, 'r') as file:
    #         rewrite_in_natural_language_data_list = json.load(file)
    #     if not correct_index_list:
    #         rewrite_in_natural_language_data_list = rewrite_in_natural_language_data_list[:n_train]
    #     else:
    #         rewrite_in_natural_language_data_list = [rewrite_in_natural_language_data_list[i] for i in correct_index_list]
    #     original_rewrite_in_natural_language_data_list = copy.deepcopy(rewrite_in_natural_language_data_list)
    
    if train_task_name.lower() == 'ecqa' or train_task_name.lower() == 'esnli':
        with open(redundant_data_train_path, 'r') as file:
            redundant_data_train_data_list = json.load(file)
        if not correct_index_list:
            redundant_data_train_data_list = redundant_data_train_data_list[:n_train]
        else:
            redundant_data_train_data_list = [redundant_data_train_data_list[i] for i in correct_index_list]
        original_redundant_data_train_data_list = copy.deepcopy(redundant_data_train_data_list)

    # if test_task_name.lower() != 'api_bank' and test_task_name.lower() != 'plan_bench' and test_task_name.lower() != 'mmlu' and test_task_name.lower() != 'winogrande' and test_task_name.lower() != 'piqa' and test_task_name.lower() != 'agieval' and test_task_name.lower() != 'squad':
    #     with open(minimum_change_train_path, 'r') as file:
    #         minimum_change_train_data_list = json.load(file)
    #     minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
    #     original_minimum_change_train_data_list = copy.deepcopy(minimum_change_train_data_list)
    with open(human_written_examples_data_train_path, 'r') as file:
        human_written_examples_data_list = json.load(file)
    if not correct_index_list:
        human_written_examples_data_list = human_written_examples_data_list[:n_train]
    else:
        human_written_examples_data_list = [human_written_examples_data_list[i] for i in correct_index_list]

    with open(simple_response_data_train_path, 'r') as file:
        simple_response_data_list = json.load(file)
    if not correct_index_list:
        simple_response_data_list = simple_response_data_list[:n_train]
    else:
        simple_response_data_list = [simple_response_data_list[i] for i in correct_index_list]
        
    with open(gpt4_generated_data_train_path, 'r') as file:
        gpt4_generated_train_data_list = json.load(file)
    if not correct_index_list:
        gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
    else:
        gpt4_generated_train_data_list = [gpt4_generated_train_data_list[i] for i in correct_index_list]
    gpt4_prediction_list = []
    for item in gpt4_generated_train_data_list:
        gpt4_prediction_list.append(item['answer'])

    with open(anthropic_generated_data_train_path, 'r') as file:
        anthropic_data_list = json.load(file)
    if not correct_index_list:
        anthropic_data_list = anthropic_data_list[:n_train]
    else:
        anthropic_data_list = [anthropic_data_list[i] for i in correct_index_list]

    with open(mini_gpt4_generated_data_train_path, 'r') as file:
        mini_gpt4_data_list = json.load(file)
    if not correct_index_list:
        mini_gpt4_data_list = mini_gpt4_data_list[:n_train]
    else:
        mini_gpt4_data_list = [mini_gpt4_data_list[i] for i in correct_index_list]

    with open(step_by_step_data_train_path, 'r') as file:
        step_by_step_data_list = json.load(file)
    if not correct_index_list:
        step_by_step_data_list = step_by_step_data_list[:n_train]
    else:
        step_by_step_data_list = [step_by_step_data_list[i] for i in correct_index_list]
    
    with open(provide_gpt4_style_example_data_train_path, 'r') as file:
        provide_gpt4_style_example_data_list = json.load(file)
    if not correct_index_list:
        provide_gpt4_style_example_data_list = provide_gpt4_style_example_data_list[:n_train]
    else:
        provide_gpt4_style_example_data_list = [provide_gpt4_style_example_data_list[i] for i in correct_index_list]
    
    if 'boolq' not in train_task_name.lower() and 'api_bank' not in train_task_name.lower() and 'mmlu' not in train_task_name.lower() and 'winogrande' not in train_task_name.lower() and 'piqa' not in train_task_name.lower() and 'agieval' not in train_task_name.lower() and 'squad' not in train_task_name.lower() and 'mmlu_pro' not in train_task_name.lower() and 'mmlu_pro_law' not in train_task_name.lower() and 'arc_challenge' not in train_task_name.lower() and 'hellaswag' not in train_task_name.lower() and 'theoremqa' not in train_task_name.lower() and 'drop' not in train_task_name.lower():
        in_own_words_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        with open(in_own_words_train_path, 'r') as file:
            in_own_words_data_list = json.load(file)
        if not correct_index_list:
            in_own_words_data_list = in_own_words_data_list[:n_train]
        else:
            in_own_words_data_list = [in_own_words_data_list[i] for i in correct_index_list]
        original_in_own_words_data_list = copy.deepcopy(in_own_words_data_list)
    
    original_train_data_list = copy.deepcopy(train_data_list)
    original_mini_gpt4_data_list = copy.deepcopy(mini_gpt4_data_list)
    original_anthropic_data_list = copy.deepcopy(anthropic_data_list)
    original_gpt4_generated_train_data_list = copy.deepcopy(gpt4_generated_train_data_list)
    original_step_by_step_data_list = copy.deepcopy(step_by_step_data_list)
    original_provide_gpt4_style_example_data_list = copy.deepcopy(provide_gpt4_style_example_data_list)
    original_human_written_examples_data_list = copy.deepcopy(human_written_examples_data_list)
    original_simple_response_data_list = copy.deepcopy(simple_response_data_list)

    if 'ecqa' in train_task_name.lower() or 'esnli' in train_task_name.lower():
        original_gold_label_data_list = copy.deepcopy(gold_label_data_list)
    
    def generate_random_lists(num_lists, sample_num=3, sub_sample_num=10):
        random_lists = []
        for i in range(num_lists):
            # Create a value range excluding the current index 'i'
            value_range = [x for x in range(sub_sample_num) if x != i]
            random_list = random.sample(value_range, sample_num)
            random_lists.append(random_list)
        return random_lists
    
    if 'BOOLQ' in train_task_name.upper():
        sample_num = 6
    elif 'PLAN_BENCH' in train_task_name.upper() or 'API_BANK' in train_task_name.upper():
        sample_num = 1
    else:
        sample_num = 3
        # sample_num = 1
    sub_sample_num = min(10, len(initial_prediction_list))
    
    sub_samples_num_list = generate_random_lists(n_train, sample_num = sample_num, sub_sample_num = sub_sample_num)
    
    # if 'plan_bench' in train_task_name.lower():
    #     rewrite_in_natural_language_data_list = process_data_list(rewrite_in_natural_language_data_list, prompt_style = 'rewrite_in_natural_language', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)


    # if test_task_name.lower() != 'api_bank' and test_task_name.lower() != 'plan_bench' and test_task_name.lower() != 'mmlu' and test_task_name.lower() != 'winogrande' and test_task_name.lower() != 'piqa' and test_task_name.lower() != 'agieval' and test_task_name.lower() != 'squad':

    #     minimum_change_train_data_list = process_data_list(minimum_change_train_data_list, prompt_style = 'minimum_change', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx)

        # if plot_record_mispredicted_samples:
        #     minimum_change_train_data_list = minimum_change_train_data_list_filtered[:n_train]
        # else:
        #     minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
    human_written_examples_data_list = initial_prediction_in_context_process_data_list(human_written_examples_data_list, initial_prediction_list, prompt_style = 'human_written_examples', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    provide_gpt4_style_example_data_list = initial_prediction_in_context_process_data_list(provide_gpt4_style_example_data_list, initial_prediction_list, prompt_style = 'provide_gpt4_style_example', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    # if test_task_name.lower() != 'code' and test_task_name.lower() != 'mbpp':
    step_by_step_data_list = initial_prediction_in_context_process_data_list(step_by_step_data_list, initial_prediction_list, prompt_style = 'step_by_step', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    train_data_list = initial_prediction_in_context_process_data_list(train_data_list, initial_prediction_list, prompt_style = 'gt_style', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)
    
    gpt4_generated_train_data_list = initial_prediction_in_context_process_data_list(gpt4_generated_train_data_list, initial_prediction_list, prompt_style = 'gpt4', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    anthropic_data_list = initial_prediction_in_context_process_data_list(anthropic_data_list, initial_prediction_list, prompt_style = 'anthropic', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    mini_gpt4_data_list = initial_prediction_in_context_process_data_list(mini_gpt4_data_list, initial_prediction_list, prompt_style = 'mini_gpt4', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    simple_response_data_list = initial_prediction_in_context_process_data_list(simple_response_data_list, initial_prediction_list, prompt_style = 'simple_response', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    # if test_task_name.lower() == 'esnli' or test_task_name.lower() == 'boolq':
    #     mini_gpt4_style_data_list = process_data_list(mini_gpt4_style_data_list, prompt_style = 'mini_gpt4_style', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx)

    if train_task_name.lower() == 'ecqa' or train_task_name.lower() == 'esnli':
        gold_label_data_list = initial_prediction_in_context_process_data_list(gold_label_data_list, initial_prediction_list, prompt_style = 'gold_label', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    if 'boolq' not in train_task_name.lower() and 'api_bank' not in train_task_name.lower() and 'mmlu' not in train_task_name.lower() and 'winogrande' not in train_task_name.lower() and 'piqa' not in train_task_name.lower() and 'agieval' not in train_task_name.lower() and 'squad' not in train_task_name.lower() and 'mmlu_pro' not in train_task_name.lower() and 'mmlu_pro_law' not in train_task_name.lower() and 'arc_challenge' not in train_task_name.lower() and 'hellaswag' not in train_task_name.lower() and 'theoremqa' not in train_task_name.lower() and 'drop' not in train_task_name.lower():
        in_own_words_data_list = initial_prediction_in_context_process_data_list(in_own_words_data_list, initial_prediction_list, prompt_style = 'in_own_words', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    if 'esnli' in train_task_name.lower() or 'ecqa' in train_task_name.lower():
        redundant_data_train_data_list = initial_prediction_in_context_process_data_list(redundant_data_train_data_list, initial_prediction_list, prompt_style = 'redundant', test_task_name = train_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, n_train = n_train, test_idx = test_idx, correct_index_list = correct_index_list)

    current_task_name = train_task_name.lower()
    train_config, test_config = set_config(current_task_name, 0, 0, model_name = 'mistral')

    if 'api_bank' in train_task_name.lower() or 'plan_bench' in train_task_name.lower():
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 16
    else:
        per_device_train_batch_size = train_config['per_device_train_batch_size']
        per_device_train_batch_size = math.floor(per_device_train_batch_size / 2)
        gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        gradient_accumulation_steps *= 2

    train_config['per_device_train_batch_size'] = per_device_train_batch_size
    train_config['gradient_accumulation_steps'] = gradient_accumulation_steps
    
    test_config['max_length'] = 4024
    test_config['max_input_length'] = 3000
    test_config['max_new_tokens'] = 1024
    test_config['per_device_eval_batch_size'] = 2

    # dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['claude', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['gpt4_style_in_context_examples', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['openai_human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, ''], ['simple_response', simple_response_data_list, simple_response_data_train_path, original_simple_response_data_list, '']]

    # dataset_list = [['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['claude', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['gpt4_style_in_context_examples', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['openai_human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, ''], ['simple_response', simple_response_data_list, simple_response_data_train_path, original_simple_response_data_list, '']]

    dataset_list = [['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['claude', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['gpt4_style_in_context_examples', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['openai_human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, ''], ['simple_response', simple_response_data_list, simple_response_data_train_path, original_simple_response_data_list, '']]

    if 'ecqa' in train_task_name.lower() or 'esnli' in train_task_name.lower() or 'plan_bench' in train_task_name.lower() or 'gsm8k' in train_task_name.lower() or 'math' in train_task_name.lower() or 'mbpp' in train_task_name.lower():
        dataset_list.append(['rewrite_groundtruth_in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, ''])

    return dataset_list, train_config, test_config, train_task_name, gpt4_prediction_list


def perplexity_calculation_in_context_total_data_loader(train_task_name, n_train, use_in_context_learning, test_idx, end_template, api_type = 'mini_gpt4', prompt_api = 'gpt4', total_use_simple_structure = False, correct_index_list = []):
    test_task_name = train_task_name.lower()
    # if test_task_name.lower() == 'plan_bench' or test_task_name.lower() == 'api_bank':
    if total_use_simple_structure:
        total_data_path = f'{HOME_DIRECTORY}/diverse_data_creation/generated_diverse_target_response/data/{test_task_name.lower()}/{api_type}/{prompt_api}_for_prompt_total_20_10_simple_structure.json'
    else:
        total_data_path = f'{HOME_DIRECTORY}/diverse_data_creation/generated_diverse_target_response/data/{test_task_name.lower()}/{api_type}/{prompt_api}_for_prompt_total_20_10.json'
    with open(total_data_path, 'r') as file:
        total_data_list = json.load(file)
    
    if not correct_index_list:
        train_data_list = train_data_list[:n_train]
    else:
        train_data_list = [train_data_list[i] for i in correct_index_list]
    
    def generate_random_lists(num_lists, sample_num=3, sub_sample_num=10):
        random_lists = []
        for i in range(num_lists):
            # Create a value range excluding the current index 'i'
            value_range = [x for x in range(sub_sample_num) if x != i]
            random_list = random.sample(value_range, sample_num)
            random_lists.append(random_list)
        return random_lists
    
    if 'BOOLQ' in test_task_name.upper():
        sample_num = 6
    elif 'PLAN_BENCH' in test_task_name.upper() or 'API_BANK' in test_task_name.upper():
        sample_num = 1
    else:
        sample_num = 3
    sub_sample_num = 10
    
    sub_samples_num_list = generate_random_lists(n_train, sample_num = sample_num, sub_sample_num = sub_sample_num)

    vairation_num_range = len(total_data_list)
    dataset_list = []
    for vairation_num in range(vairation_num_range):
        total_data_list_item = process_data_list(total_data_list[vairation_num], prompt_style = 'total', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, use_in_context_learning = use_in_context_learning, n_train = n_train, test_idx = test_idx, vairation_num = vairation_num, api_type = api_type, prompt_api = prompt_api, total_use_simple_structure = total_use_simple_structure, correct_index_list = correct_index_list)
        original_total_data_list_item = copy.deepcopy(total_data_list_item)
        # if 'plan_bench' in test_task_name.lower() or 'api_bank' in test_task_name.lower():
        if total_use_simple_structure:
            dataset_list.append([f'total_{vairation_num}_prompt_api_{prompt_api}_generation_api_{api_type}_simple_prompt', total_data_list_item, total_data_path, original_total_data_list_item, ''])
        else:
            dataset_list.append([f'total_{vairation_num}_prompt_api_{prompt_api}_generation_api_{api_type}', total_data_list_item, total_data_path, original_total_data_list_item, ''])

    return dataset_list, test_task_name

def load_original_data(train_task_name):
    train_task_name = train_task_name.lower()

    if train_task_name.lower() == 'gsm8k':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/train_filtered.json'
        train_data_list = load_GSM8K(train_path, 1000, zeroshot = True, load_original_question = True)
    if 'math_algebra' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_ALGEBRA/train_algebra_total_filtered.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = False, load_original_question = True)
    if 'math_geometry' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_GEOMETRY/groundtruth.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = False, load_original_question = True)
    if 'math_intermediate_algebra' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_INTERMEDIATE_ALGEBRA/groundtruth.json'
        train_data_list = load_MATH(train_path, 1000, zeroshot = False, load_original_question = True)
    if train_task_name.lower() == 'esnli':
        train_path = f'{HOME_DIRECTORY}/dataset/ESNLI/train.json'
        train_data_list = load_ESNLI(train_path, 1000, load_original_question = True)
    if train_task_name.lower() == 'ecqa':
        train_path = f'{HOME_DIRECTORY}/dataset/ECQA/train.json'
        train_data_list = load_ECQA(train_path, 1000, finetune = True, use_gt_rationale = True, load_original_question = True)
    if 'plan_bench' in train_task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as f:
            train_data_list = json.load(f)
    if train_task_name.lower() == 'boolq':
        train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
        train_data_list = load_BOOLQ(train_path, 1000, finetune = True, load_original_question = True)
    if train_task_name.lower() == 'mmlu':
        train_path = f'{HOME_DIRECTORY}/dataset/MMLU/groundtruth.json'
        train_data_list = load_MMLU(train_path, 1000, finetune = True, load_original_question = True)
    if train_task_name.lower() == 'mmlu_moral_scenarios':
        train_path = f'{HOME_DIRECTORY}/dataset/MMLU_MORAL_SCENARIOS/groundtruth.json'
        train_data_list = load_MMLU(train_path, 1000, finetune = True, load_original_question = True)
    if train_task_name.lower() == 'winogrande':
        train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
        train_data_list = load_WINOGRANDE(train_path, 1000, finetune = True, load_original_question = True)
    if train_task_name.lower() == 'piqa':
        train_path = f'{HOME_DIRECTORY}/dataset/PIQA/train.json'
        train_data_list = load_PIQA(train_path, 1000, finetune = True, load_original_question = True)
    if train_task_name.lower() == 'squad':
        train_path = f'{HOME_DIRECTORY}/dataset/SQUAD/train.json'
        train_data_list = load_SQUAD(train_path, 1000, finetune = True, load_original_question = True)
    if train_task_name.lower() == 'drop':
        train_path = f'{HOME_DIRECTORY}/dataset/DROP/groundtruth.json'
        train_data_list = load_DROP(train_path, 1000, finetune_with_gt = True, load_original_question = True)
    if train_task_name.lower() == 'agieval':
        train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/train.json'
        train_data_list = load_AGIEVAL(train_path, 1000, finetune = True, load_original_question = True)
    if train_task_name.lower() == 'api_bank':
        train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-api-train.json'
        # train_data_list = load_API_BANK_optimized(train_path, 1000)
        train_data_list = load_API_BANK(train_path, 1000, load_original_question = True)
    if train_task_name.upper() == 'MBPP':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        train_data_list = load_MBPP(train_path, 1000)


    if train_task_name.lower() == 'mmlu_pro' or train_task_name.lower() == 'mmlu_pro_law' or train_task_name.lower() == 'arc_challenge' or train_task_name.lower() == 'hellaswag' or train_task_name.lower() == 'theoremqa':
        train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:1000]

    return train_data_list