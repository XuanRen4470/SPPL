import json
from utils.data_loader import load_training_dataset, add_gold_label, load_gold_label_and_question_list, load_variation_path
from utils.function_synthetic_data_generation import create_answer_directly_response, create_response_varient
from evaluation.eval import Check_Correctness
from utils.initialization import initial_output_folder
from config.config import HOME_DIRECTORY



task_name_list = ['gsm8k', 'math_algebra', 'math_geometry', 'ecqa', 'boolq', 'winogrande', 'piqa', 'agieval', 'squad', 'arc_challenge', 'drop', 'mbpp', 'api_bank', 'hellaswag', 'mmlu_pro_law', 'mmlu_moral_scenarios']

task_name_list = ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_replaning', 'plan_bench_reuse', 'plan_bench_verification']#, 'plan_bench_execution']



# api_list = ["gpt4", "mini_gpt4o", "claude"]
# api_list = ["claude"]
# api_list = ["mini_gpt4o"]
api_list = ["gpt4"]

variation_suffix_list_non_gpt4_gt_non_cot = ['none']
variation_suffix_list_non_gpt4_gt_cot = ['none']
variation_suffix_list_gpt4_gt_cot = ['variation_rewirte_groundtruth_in_own_words', 'variation_step_by_step', 'variation_openai_human_written_examples', 'variation_gpt4_style_in_context_examples']
variation_suffix_list_gpt4_gt_non_cot = ['variation_rewirte_groundtruth_in_own_words', 'variation_step_by_step', 'variation_openai_human_written_examples', 'variation_gpt4_style_in_context_examples']

# variation_suffix_list_gpt4_gt_cot = ['variation_redundant']
# variation_suffix_list_gpt4_gt_non_cot = ['variation_redundant']



n_data_creation = 1000
output_folder_name = 'error_correction'
seed_num = 0
initial_output_folder(output_folder_name, seed_num)
enable_error_correction = True

for task_name in task_name_list:
    for api in api_list:
        non_cot_name_list = ['boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'agieval', 'api_bank', 'mmlu_pro', 'mmlu_pro_law', 'mmlu_moral_scenarios', 'hellaswag', 'arc_challenge', 'drop']
        cot_name_list = ['mbpp', "gsm8k", 'math_algebra', 'ecqa', 'esnli', 'plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality', 'plan_bench_reuse', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'math_intermediate_algebra', 'math_geometry', 'gsm8k_paraphrase', 'math_algebra_paraphrase']

        if api == "gpt4":
            if task_name in cot_name_list :
                variation_suffix_list = variation_suffix_list_gpt4_gt_cot
            if task_name in non_cot_name_list:
                variation_suffix_list = variation_suffix_list_gpt4_gt_non_cot
        else:
            # variation_suffix_list = variation_suffix_list_non_gpt4
            if task_name in cot_name_list :
                variation_suffix_list = variation_suffix_list_non_gpt4_gt_cot
            if task_name in non_cot_name_list:
                variation_suffix_list = variation_suffix_list_non_gpt4_gt_non_cot

        for variation_suffix in variation_suffix_list:
            job_name = task_name
            train_task_name = task_name

            if 'step_by_step' in variation_suffix:
                step_by_step = True
            else:
                step_by_step = False
            
            if task_name == 'math_geometry' or task_name == 'math_intermediate_algebra' or task_name == 'mbpp':
                provide_groundtruth_with_inference_steps = True
            else:
                provide_groundtruth_with_inference_steps = False
            # answer_without_groundtruth=True
            answer_without_groundtruth=False
            temperature = 0.7
            # temperature = 1.0
            api_type = api
            
            # if task_name == 'mmlu_pro' or task_name == 'hellaswag' or task_name == 'arc_challenge':
            #     create_initial_synthetic_response = True
            # else:
            #     create_initial_synthetic_response = False

            create_initial_synthetic_response = True
            # create_initial_synthetic_response = False

            gold_label_list, groundtruth_list, question_list = load_gold_label_and_question_list(task_name, n_data_creation)
            if not create_initial_synthetic_response:
                data_list, full_path = load_training_dataset(api, HOME_DIRECTORY, train_task_name, n_data_creation, variation_suffix)
            
            original_question_list = question_list.copy()
            original_gold_label_list = gold_label_list.copy()
            original_groundtruth_list = groundtruth_list.copy()
            root_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}'
            if variation_suffix == 'none':
                if 'gpt4' == api_type:
                    full_path = f'{root_path}/gpt4.json'
                elif 'claude' == api_type:
                    full_path = f'{root_path}/claude.json'
                elif 'mini_gpt4' in api_type:
                    full_path = f'{root_path}/varient/mini_gpt4.json'
                elif 'anthropic_thinking' in api_type:
                    full_path = f'{root_path}/varient/claude_thinking.json'
            elif 'variation' in variation_suffix:# and api_type == 'gpt4':
                full_path = load_variation_path(variation_suffix, root_path, api_type = api_type)

            if create_initial_synthetic_response:
                print(f'create_initial_synthetic_response---------------{task_name}: {api_type} {variation_suffix}---------------')
                if 'step_by_step' in variation_suffix or 'none' in variation_suffix:
                    # data_list = create_answer_directly_response(question_list, gold_label_list = gold_label_list, step_by_step = step_by_step, task_name = task_name.upper(), train_data_list = data_list, provide_groundtruth_with_inference_steps = provide_groundtruth_with_inference_steps, answer_without_groundtruth = answer_without_groundtruth, api_type = api_type, temperature = temperature)
                    
                    data_list = create_answer_directly_response(question_list, gold_label_list = gold_label_list, step_by_step = step_by_step, task_name = task_name.upper(), provide_groundtruth_with_inference_steps = provide_groundtruth_with_inference_steps, answer_without_groundtruth = answer_without_groundtruth, api_type = api_type, temperature = temperature, groundtruth_list = original_groundtruth_list)

                elif 'paraphrase' in variation_suffix:
                    with open(f'{root_path}/gpt4.json', 'r') as f:
                        gpt4_prediction_list = json.load(f)
                    data_list = create_response_varient(question_list, variation_suffix, api_type, task_name.upper(), gold_label_list, groundtruth_list, temperature = temperature, original_question_list = original_question_list, original_gold_label_list = original_gold_label_list, gpt4_prediction_list = gpt4_prediction_list)
                else:
                    # data_list, full_path = load_training_dataset('gpt4', HOME_DIRECTORY, train_task_name, 2, 'none')
                    # print('gpt4_1: ' + data_list[0]['answer'])
                    # print()
                    # print()
                    # print()
                    # print()
                    # print('gpt4_2: ' + data_list[1]['answer'])

                    data_list = create_response_varient(question_list, variation_suffix, api_type, task_name.upper(), gold_label_list, groundtruth_list, temperature = temperature, original_question_list = original_question_list, original_gold_label_list = original_gold_label_list)
                    a = 1
                
                if 'plan_bench' in task_name.lower():
                    for iiii, itemm in enumerate(data_list):
                        data_list[iiii]['domain'] = groundtruth_list[iiii]['domain']
                        data_list[iiii]['instance_id'] = groundtruth_list[iiii]['instance_id']
                        data_list[iiii]['task'] = groundtruth_list[iiii]['task']
                        try:
                            data_list[iiii]['prompt_type'] = groundtruth_list[iiii]['prompt_type']
                        except:
                            a = 1
                        try:
                            data_list[iiii]['example_instance_ids'] = groundtruth_list[iiii]['example_instance_ids']
                        except:
                            a = 1
                        try:
                            data_list[iiii]['new_instance'] = groundtruth_list[iiii]['new_instance']
                        except:
                            a = 1

                if 'mbpp' in task_name.lower():
                    for iiii, itemm in enumerate(data_list):
                        data_list[iiii]['task_id'] = groundtruth_list[iiii]['task_id']
                        data_list[iiii]['test_list'] = groundtruth_list[iiii]['test_list']
                        data_list[iiii]['test_setup_code'] = groundtruth_list[iiii]['test_setup_code']
                        data_list[iiii]['challenge_test_list'] = groundtruth_list[iiii]['challenge_test_list']

                with open(full_path, 'w') as file:
                    json.dump(data_list, file, indent=4)
            
            if enable_error_correction:
                correct_counter = 0
                incorrect_counter = 0
                current_groundtruth = []
                for index, item in enumerate(data_list):
                    # print(f'enable_error_correction 1-----------index {index}----{task_name}: {variation_suffix}---------------')
                    pred_temp = []
                    data_temp = []
                    aa = item['answer']
                    pred_temp.append(aa)
                    item = add_gold_label(train_task_name, item, gold_label_list[index])

                    if 'plan_bench' in train_task_name:
                        item['domain'] = groundtruth_list[index]['domain']
                        item['task'] = groundtruth_list[index]['task']
                        # item['prompt_type'] = groundtruth_list[index]['prompt_type']
                        item['prompt_type'] = 'oneshot'

                        if 'instance_id' in groundtruth_list[index].keys():
                            item['instance_id'] = groundtruth_list[index]['instance_id']
                        if 'example_instance_ids' in groundtruth_list[index].keys():
                            item['example_instance_ids'] = groundtruth_list[index]['example_instance_ids']
                        if 'new_instance' in groundtruth_list[index].keys():
                            item['new_instance'] = groundtruth_list[index]['new_instance']
                        item['instance_id'] = groundtruth_list[index]['instance_id']

                        if 'ground_truth_plan' in groundtruth_list[index].keys():
                            item['ground_truth_plan'] = groundtruth_list[index]['ground_truth_plan']
                            

                    if 'mbpp' in task_name.lower():
                        item['task_id'] = groundtruth_list[index]['task_id']
                        item['test_list'] = groundtruth_list[index]['test_list']
                        item['test_setup_code'] = groundtruth_list[index]['test_setup_code']
                        item['challenge_test_list'] = groundtruth_list[index]['challenge_test_list']

                    keys_to_remove = [key for key in item.keys() if 'perplexity' in key]
                    for key in keys_to_remove:
                        item.pop(key)
                    
                    keys_to_remove = [key for key in item.keys() if 'original_gpt4_prediction' in key]
                    for key in keys_to_remove:
                        item.pop(key)
                    
                    data_temp.append(item)
                    accuracy, cover_ratio = Check_Correctness(pred_temp, data_temp, train_task_name, output_folder_name, task_name = 'error_correction', extract_gold_label_as_gt = True, simple_evaluation = True)
                    if index == 50:
                        a = 1
                    if accuracy == 1:    
                        correct_counter += 1                    
                        item['correct'] = True
                        data_list[index] = item
                    else:
                        incorrect_counter += 1
                        current_gold_label = [gold_label_list[index]]
                        current_groundtruth = [groundtruth_list[index]]
                        incorrect_question_list = [item['question']]
                        
                        if 'step_by_step' in variation_suffix or 'none' in variation_suffix:
                            gpt4_answer_list = create_answer_directly_response(incorrect_question_list, gold_label_list = current_gold_label, step_by_step = step_by_step, task_name = task_name.upper(), train_data_list = [data_list[index]], provide_groundtruth_with_inference_steps = provide_groundtruth_with_inference_steps, answer_without_groundtruth = answer_without_groundtruth, api_type = api_type, temperature = temperature, groundtruth_list = current_groundtruth)
                        else:
                            gpt4_answer_list = create_response_varient(incorrect_question_list, variation_suffix, api_type, task_name.upper(), current_gold_label, current_groundtruth, temperature = temperature, original_question_list = original_question_list, original_gold_label_list = original_gold_label_list)

                        gpt4_prediction = gpt4_answer_list[0]['answer']
                        accuracy, cover_ratio = Check_Correctness([gpt4_prediction], data_temp, train_task_name, output_folder_name, task_name = 'error_correction', extract_gold_label_as_gt = True, simple_evaluation = True)

                        if accuracy == 1:
                            item['answer_old'] = item.pop('answer')
                            item['answer'] = gpt4_prediction
                            item['correct'] = True
                            data_list[index] = item

                            # print('NEW GT******* ', data_temp[0]['gold_label'])
                            # print('NEW PRED******* ', gpt4_prediction)
                            # print('OLD PRED ******* ', aa)
                            # print()
                            # print()
                        else:
                            if 'step_by_step' in variation_suffix or 'none' in variation_suffix:
                                gpt4_answer_list = create_answer_directly_response(incorrect_question_list, gold_label_list = current_gold_label, step_by_step = step_by_step, task_name = task_name.upper(), train_data_list = [data_list[index]], provide_groundtruth_with_inference_steps = provide_groundtruth_with_inference_steps, answer_without_groundtruth = answer_without_groundtruth, api_type = api_type, temperature = temperature, groundtruth_list = current_groundtruth)
                            else:
                                gpt4_answer_list = create_response_varient(incorrect_question_list, variation_suffix, api_type, task_name.upper(), current_gold_label, current_groundtruth, temperature = temperature, original_question_list = original_question_list, original_gold_label_list = original_gold_label_list)

                            gpt4_prediction = gpt4_answer_list[0]['answer']
                            accuracy, cover_ratio = Check_Correctness([gpt4_prediction], data_temp, train_task_name, output_folder_name, task_name = 'error_correction', extract_gold_label_as_gt = True, simple_evaluation = True)

                            if accuracy == 1:
                                item['answer_old'] = item.pop('answer')
                                item['answer'] = gpt4_prediction
                                item['correct'] = True
                                data_list[index] = item

                                # print('NEW GT******* ', data_temp[0]['gold_label'])
                                # print('NEW PRED******* ', gpt4_prediction)
                                # print('OLD PRED ******* ', aa)
                                # print()
                                # print()
                            else:
                                item['correct'] = False                            
                                data_list[index] = item

                                # print('NEW GT******* ', data_temp[0]['gold_label'])
                                # print('NEW PRED******* ', gpt4_prediction)
                                # print('OLD PRED ******* ', aa)
                                # print()
                                # print()
                        a = 1
                a = 1
            with open(full_path, 'w') as file:
                json.dump(data_list, file, indent=4)

            print()
            print()
            print('---------------------------------------------------')
            print('PATH: ' + full_path)
            print()
            # print('QUESTION: ' + data_list[0]['question'])
            print('len correct: ' + str(correct_counter))
            print()
            print('len incorrect: ' + str(incorrect_counter))
            # print('ANSWER: ' + data_list[0]['answer'])
            a = 1
