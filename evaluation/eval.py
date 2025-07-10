import os
import re
import sys
import json
from fractions import Fraction
import subprocess
from multiprocessing import Process, Manager
import math

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.log_writter import *
from config.config import HOME_DIRECTORY, HOME_DIRECTORY, MODEL_DIRECTORY
from utils.llama_factory_data_file_processor import put_json_list_to_data_info

def parse_number_with_commas(num_str):
    """
    统一解析数字字符串：
      - 若只出现一次 ',' 并且没有 '.', 则将 ',' 当作小数点
      - 否则，将所有 ',' 视为千分位分隔符去掉
      - 最后转成 float，如果是整数则再转成 int
    """
    # 去掉前后空格
    num_str = num_str.strip()
    is_percentage = '%' in num_str
    # 去掉百分号
    num_str = num_str.replace('%', '')

    num_str = num_str.replace(',', '')

    # 转 float
    val = float(num_str)
    
    # 如果是百分数就除以100
    if is_percentage:
        val /= 100
    
    # 如果小数部分为 0，就转成 int
    if val.is_integer():
        return str(int(val))
    else:
        return str(val)

def extract_boxed_content(s):
    start = s.rfind('\\boxed{')
    if start == -1:
        return None

    # 从 \boxed{ 往后，搜索第一个 }
    end = s.find('}', start + 7)
    if end == -1:
        return None

    answer = s[start + 7 : end] # 7 is the length of '\\boxed{'
    return answer

def evaluate_expression_try_best(expr):
    try:
        # Handle LaTeX-style fractions and square roots
        expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
        expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

        expr = re.sub(r'(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)', r'(\1) / (\2)', expr)



        # Evaluate the expression
        result = eval(expr)
        result = float(result)
        return str(result)
    except:
        return False
    
def evaluate_expression(expr):
    if 'sqrt' in expr or '^' in expr or '(' in expr:
        return False
    try:
        # Handle LaTeX-style fractions and square roots
        expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        # expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
        expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

        # Evaluate the expression
        result = eval(expr)
        
        return float(result)
    except:
        return False
    
def extract_last_number(text):
    # New pattern to include LaTeX-style expressions
    # pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\}|\\dfrac\{.*?\}\{.*?\}|\\cfrac\{.*?\}\{.*?\}|\\sqrt\{.*?\})'
    # text = text.replace(',', '')
    pattern = r'(-?\d+\/-?\d+|-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\}|\\dfrac\{.*?\}\{.*?\}|\\cfrac\{.*?\}\{.*?\}|\\sqrt\{.*?\})'


    founded_text = extract_boxed_content(text)
    if founded_text:
        if '\\frac' in founded_text or '\\dfrac' in founded_text or '\\cfrac' in founded_text or '\\sqrt' in founded_text or '/' in founded_text:
            extracted_num = evaluate_expression_try_best(founded_text)
            if not extracted_num:
                return -3333333333333 
            else:
                return extracted_num
        else: 
            text = founded_text

    # Find all numbers and expressions in the string
    all_numbers = re.findall(pattern, text)

    # Process the last number or expression
    if all_numbers:
        number = all_numbers[-1]
        # Evaluate LaTeX-style expressions
        if '\\frac' in number or '\\dfrac' in number or '\\cfrac' in number or '\\sqrt' in number or '/' in number:
            extracted_num = evaluate_expression_try_best(number)
            if not extracted_num:
                return -3333333333333 
            else:
                return extracted_num
        else:
            return parse_number_with_commas(str(number))
        
        # Handle percentages and remove commas from numbers
        is_percentage = '%' in number
        number = number.replace('%', '').replace(',', '')
        
        # Convert to float and adjust for percentage if needed
        number = float(number)
        if is_percentage:
            number /= 100

        return str(number)
    else:
        return -3333333333333 



def calc_accuracy_GSM8K(question_list, output_list, groundtruth_list, output_folder_name, task_name = ''):
    eval_data_list_updated = []
    mispredict_eval_data_list_updated = []
    eval_num = 0 # how many data is evaluated?
    accuracy = 0
    cover_ratio = 0

    for i in range(len(output_list)):
        if i % 50 ==0:
            print(i)
        temp = output_list[i]
        # extracted_final_answer = extract_boxed_content(temp) 
        extracted_final_answer = extract_last_number(temp)
        final_answer = extracted_final_answer

        # temp = groundtruth_list[i]
        # extracted_groundtruth = extract_gsm8k_num(temp)
        # extracted_groundtruth = extract_last_number(temp)
        # extracted_groundtruth = extract_boxed_content(temp) 
        # groundtruth_num = extracted_groundtruth
        groundtruth_num = groundtruth_list[i]

        result = float(groundtruth_num)
        groundtruth_num = f"{result:.2f}"
        final_answer = float(final_answer)
        final_answer = f"{final_answer:.2f}"
        eval_num += 1
        if final_answer == groundtruth_num:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = groundtruth_num
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Correct'
            eval_data_list_updated.append(item_temp)
        else:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = groundtruth_num
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Incorrect'
            mispredict_eval_data_list_updated.append(item_temp)
           
    accuracy = len(eval_data_list_updated)/eval_num      
    cover_ratio = eval_num/len(output_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(eval_data_list_updated, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(mispredict_eval_data_list_updated, file, indent=4)
    return accuracy, cover_ratio

def calc_accuracy_MATH(question_list, output_list, groundtruth_list, output_folder_name, task_name = ''):
    eval_data_list_updated = []
    mispredict_eval_data_list_updated = []
    eval_num = 0 # how many data is evaluated?
    accuracy = 0
    cover_ratio = 0

    for i in range(len(output_list)):
        if i % 50 ==0:
            print(i)
        temp = output_list[i]
        extracted_final_answer = extract_last_number(temp)
        final_answer = extracted_final_answer

        temp = groundtruth_list[i]
        extracted_groundtruth = extract_last_number(temp)
        groundtruth_num = extracted_groundtruth

        result = float(groundtruth_num)
        groundtruth_num = f"{result:.2f}"
        final_answer = float(final_answer)
        final_answer = f"{final_answer:.2f}"

        eval_num += 1
        if final_answer == groundtruth_num:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = extracted_groundtruth
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Correct'
            eval_data_list_updated.append(item_temp)
        else:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = extracted_groundtruth
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Incorrect'
            mispredict_eval_data_list_updated.append(item_temp)
           
    accuracy = len(eval_data_list_updated)/eval_num      
    cover_ratio = eval_num/len(output_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(eval_data_list_updated, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(mispredict_eval_data_list_updated, file, indent=4)
    return accuracy, cover_ratio

def calc_accuracy_API_BANK(API_BANK_test_data_list, predict_list, output_folder_name, task_name=''):
    sys.path.append(f'{HOME_DIRECTORY}/DAMO_ConvAI')
    from api_bank.lv3_evaluator_new import eval_api_bank

    for i in range(len(API_BANK_test_data_list)):
        API_BANK_test_data_list[i]['pred'] = predict_list[i]
    task_name = task_name.upper()
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_predictions.json", 'w') as file:
        json.dump(API_BANK_test_data_list, file, indent=4)

    accuracy, lv12_accuracy, lv3_accuracy = eval_api_bank(API_BANK_test_data_list, HOME_DIRECTORY) 
    cover_ratio = 1
    return accuracy, cover_ratio, lv12_accuracy, lv3_accuracy

def calc_accuracy_API_BANK_simple(API_BANK_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    
    for i, answer in enumerate(predict_list):
        gold_label = API_BANK_test_data_list[i]['gold_label']
        if 'final answer' not in gold_label.lower():
            gold_label = 'Final Answer: ' + gold_label
        final_answer = extract_text_span(answer)
        gold_label = extract_text_span(gold_label)

        if final_answer != 'null':
            count += 1

        if final_answer[0] == '"':  # remove the last period
            final_answer = final_answer[1:]

        if final_answer[-1] == '"':  # remove the last period
            final_answer = final_answer[:-1]
        
        if final_answer[-1] == '.':  # remove the last period
            final_answer = final_answer[:-1]


        if gold_label[0] == '"':  # remove the last period
            gold_label = gold_label[1:]

        if gold_label[-1] == '"':  # remove the last period
            gold_label = gold_label[:-1]

        if gold_label[-1] == '.':  # remove the last period
            gold_label = gold_label[:-1]


        if gold_label.lower() == final_answer.lower():
            correct_count += 1
        
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    lv12_accuracy = 0
    lv3_accuracy = 0
    return accuracy, cover_ratio, lv12_accuracy, lv3_accuracy



def calc_accuracy_PLAN_BENCH(PLAN_BENCH_test_data_list_total, predict_list, output_folder_name, task_name=''):
    sys.path.append(f'{HOME_DIRECTORY}/LLMs-Planning-main/plan-bench')
    from response_evaluation_modified import eval_plan_generation

    def extract_llm_raw_response(answer):
        # Split the answer by 'Final Answer', case-insensitive
        parts = re.split(r'(?i)final answer\s*:', answer)
        if len(parts) > 1:
            # Get the content after the last 'Final Answer'
            llm_raw_response = parts[-1].strip()
        else:
            # If 'Final Answer' not found, return the whole answer or handle accordingly
            llm_raw_response = answer.strip()
        return llm_raw_response

    PLAN_BENCH_test_data_list_temp_total = []
    PLAN_BENCH_test_data_list_temp = []
    previous_domain = ''
    for item in PLAN_BENCH_test_data_list_total:
        if not previous_domain:
            previous_domain = item['domain']
        if item['domain'] != previous_domain:
            PLAN_BENCH_test_data_list_temp_total.append(PLAN_BENCH_test_data_list_temp)
            previous_domain = item['domain']
            PLAN_BENCH_test_data_list_temp = []
        PLAN_BENCH_test_data_list_temp.append(item)
    PLAN_BENCH_test_data_list_temp_total.append(PLAN_BENCH_test_data_list_temp)
    PLAN_BENCH_test_data_list_total = PLAN_BENCH_test_data_list_temp_total
    correct_count = 0
    total = 0
    current_id = 0
    for PLAN_BENCH_test_data_list in PLAN_BENCH_test_data_list_total:
        total += len(PLAN_BENCH_test_data_list)
        domain_name = PLAN_BENCH_test_data_list[0]["domain"]
        # Load the initial JSON data
        output_data = {
            "task": PLAN_BENCH_test_data_list[0]["task"],
            "prompt_type": PLAN_BENCH_test_data_list[0]["prompt_type"],
            "domain": domain_name,
            "instances": []
        }

        # Process each instance in the initial data
        for i, instance in enumerate(PLAN_BENCH_test_data_list):
            answer = predict_list[current_id]
            current_id += 1
            answer = extract_llm_raw_response(answer)
            answer = answer + '\n'
                
            new_instance = {
                "instance_id": instance["instance_id"],
                "query": instance["question"],
                "llm_raw_response": answer,
            }
            if "example_instance_ids" in instance:
                new_instance["example_instance_ids"] = instance["example_instance_ids"]
            
            if 'ground_truth_plan' in instance.keys():
                new_instance["ground_truth_plan"] = instance["ground_truth_plan"]
            else:
                new_instance["ground_truth_plan"] = instance["gold_label"]
            if 'new_instance' in instance.keys():
                new_instance["new_instance"] = instance["new_instance"]

            output_data["instances"].append(new_instance)

        task_name = task_name.upper()
        modified_path = f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_{domain_name}.json"

        with open(modified_path, 'w') as file:
            json.dump(output_data, file, indent=4)
        modified_path = modified_path.replace('.json', '')
        task_ = PLAN_BENCH_test_data_list[0]['task']
        if task_ == 'task_1_plan_generation':
            task_ = 't1'
        elif task_ == 'task_2_plan_optimality':
            task_ = 't2'
        elif task_ == 'task_3_plan_verification':
            task_ = 't3'
        elif task_ == 'task_3_plan_verification_with_llm_plans':  
            task_ = 't3_1'
        elif task_ == 'task_4_plan_reuse':
            task_ = 't4'
        elif task_ == 'task_5_plan_generalization':
            task_ = 't5'
        elif task_ == 'task_6_replanning':
            task_ = 't6'
        elif task_ == 'task_7_plan_execution':
            task_ = 't7'
        elif task_ == 'task_8_1_goal_shuffling':
            task_ = 't8_1'
        elif task_ == 'task_8_2_full_to_partial':
            task_ = 't8_2'
        elif task_ == 'task_8_3_partial_to_full':
            task_ = 't8_3'
        accuracy = eval_plan_generation(modified_path, HOME_DIRECTORY, task = task_, config = domain_name, engine = "gpt-3.5-turbo_chat")
        try:
            correct_count += accuracy*(len(PLAN_BENCH_test_data_list))
        except Exception as err:
            print('accuracy:', accuracy)
            print('PLAN_BENCH_test_data_list:', PLAN_BENCH_test_data_list)
            print('caught error:', err)
            raise  # 重新抛出同一异常

    cover_ratio = 1
    accuracy = correct_count/total
    accuracy = round(accuracy, 3)
    return accuracy, cover_ratio

def calc_accuracy_ANLI(ANLI_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    ANLI_test_data_mispredict_list = []
    ANLI_test_data_correct_predict_list = []
    for i in range(len(ANLI_test_data_list)):
        ANLI_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = ANLI_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            ANLI_test_data_item = ANLI_test_data_list[i]
            ANLI_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                ANLI_test_data_correct_predict_list.append(ANLI_test_data_item)
            else:
                ANLI_test_data_mispredict_list.append(ANLI_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(ANLI_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(ANLI_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_MNLI(MNLI_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    MNLI_test_data_mispredict_list = []
    MNLI_test_data_correct_predict_list = []
    for i in range(len(MNLI_test_data_list)):
        MNLI_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = MNLI_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            MNLI_test_data_item = MNLI_test_data_list[i]
            MNLI_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                MNLI_test_data_correct_predict_list.append(MNLI_test_data_item)
            else:
                MNLI_test_data_mispredict_list.append(MNLI_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(MNLI_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(MNLI_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio



def calc_accuracy_ESNLI(ESNLI_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    ESNLI_test_data_mispredict_list = []
    ESNLI_test_data_correct_predict_list = []
    for i in range(len(ESNLI_test_data_list)):
        ESNLI_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = ESNLI_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            ESNLI_test_data_item = ESNLI_test_data_list[i]
            ESNLI_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                ESNLI_test_data_correct_predict_list.append(ESNLI_test_data_item)
            else:
                ESNLI_test_data_mispredict_list.append(ESNLI_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(ESNLI_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(ESNLI_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio



def calc_accuracy_SCITAIL(SCITAIL_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    SCITAIL_test_data_mispredict_list = []
    SCITAIL_test_data_correct_predict_list = []
    for i in range(len(SCITAIL_test_data_list)):
        SCITAIL_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = SCITAIL_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            SCITAIL_test_data_item = SCITAIL_test_data_list[i]
            SCITAIL_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                SCITAIL_test_data_correct_predict_list.append(SCITAIL_test_data_item)
            else:
                SCITAIL_test_data_mispredict_list.append(SCITAIL_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(SCITAIL_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(SCITAIL_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_BOOLQ(BOOLQ_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    BOOLQ_test_data_mispredict_list = []
    BOOLQ_test_data_correct_predict_list = []
    for i in range(len(BOOLQ_test_data_list)):
        BOOLQ_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(BOOLQ_test_data_list[i]['gold_label'])
        if answer:
            final_answer = extract_bool(answer)
            BOOLQ_test_data_item = BOOLQ_test_data_list[i]
            BOOLQ_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                BOOLQ_test_data_correct_predict_list.append(BOOLQ_test_data_item)
            else:
                BOOLQ_test_data_mispredict_list.append(BOOLQ_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(BOOLQ_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(BOOLQ_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_SQUAD(SQUAD_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    SQUAD_test_data_mispredict_list = []
    SQUAD_test_data_correct_predict_list = []
    for i in range(len(predict_list)):
        SQUAD_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        answer_list = SQUAD_test_data_list[i]['answer_list']
        final_answer = extract_text_span(answer)

        if final_answer != 'null':
            count += 1
        found_answer = False

        if final_answer[0] == '"':  # remove the last period
            final_answer = final_answer[1:]

        if final_answer[-1] == '"':  # remove the last period
            final_answer = final_answer[:-1]
        
        if final_answer[-1] == '.':  # remove the last period
            final_answer = final_answer[:-1]

        for answer_item in answer_list:

            if answer_item[0] == '"':  # remove the last period
                answer_item = answer_item[1:]

            if answer_item[-1] == '"':  # remove the last period
                answer_item = answer_item[:-1]

            if answer_item[-1] == '.':  # remove the last period
                answer_item = answer_item[:-1]


            if answer_item.lower() == final_answer.lower():
                correct_count += 1
                found_answer = True
                break
            else:
                found_answer = False
        
        squad_test_data_item = SQUAD_test_data_list[i]
        squad_test_data_item['extracted_answer'] = final_answer
        if found_answer:
            SQUAD_test_data_correct_predict_list.append(squad_test_data_item)
        else:
            SQUAD_test_data_mispredict_list.append(squad_test_data_item)
            
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(SQUAD_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(SQUAD_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio

def calc_accuracy_SQUAD_simplified(SQUAD_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    SQUAD_test_data_mispredict_list = []
    SQUAD_test_data_correct_predict_list = []
    for i in range(len(predict_list)):
        SQUAD_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gt = SQUAD_test_data_list[i]['gold_label']
        final_answer = extract_text_span(answer)

        if final_answer != 'null':
            count += 1
        found_answer = False

        if final_answer[0] == '"':  # remove the last period
            final_answer = final_answer[1:]

        if final_answer[-1] == '"':  # remove the last period
            final_answer = final_answer[:-1]
        
        if final_answer[-1] == '.':  # remove the last period
            final_answer = final_answer[:-1]


        if gt[0] == '"':  # remove the last period
            gt = gt[1:]

        if gt[-1] == '"':  # remove the last period
            gt = gt[:-1]

        if gt[-1] == '.':  # remove the last period
            gt = gt[:-1]


        if gt.lower() == final_answer.lower():
            correct_count += 1
            found_answer = True
        else:
            found_answer = False
        
        squad_test_data_item = SQUAD_test_data_list[i]
        squad_test_data_item['extracted_answer'] = final_answer
        if found_answer:
            SQUAD_test_data_correct_predict_list.append(squad_test_data_item)
        else:
            SQUAD_test_data_mispredict_list.append(squad_test_data_item)
            
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    return accuracy, cover_ratio

def calc_accuracy_PIQA(PIQA_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    PIQA_test_data_mispredict_list = []
    PIQA_test_data_correct_predict_list = []
    # for i in range(len(PIQA_test_data_list)):
    #     PIQA_test_data_list[i]['pred'] = predict_list[i]
    
    for i in range(min(len(PIQA_test_data_list), len(predict_list))):
        PIQA_test_data_list[i]['pred'] = predict_list[i]

    for i, answer in enumerate(predict_list):
        gold_label = str(PIQA_test_data_list[i]['gold_label'])

        sol12_content = PIQA_test_data_list[i]['sol'+gold_label]
        gold_label = gold_label.lower()
        option1 = PIQA_test_data_list[i]['sol1']
        option2 = PIQA_test_data_list[i]['sol2']

        sol12_content = sol12_content.strip().lower().rstrip('.')
        option1 = option1.strip().lower().rstrip('.')
        option2 = option2.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_option(answer)
            PIQA_test_data_item = PIQA_test_data_list[i]
            PIQA_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                PIQA_test_data_correct_predict_list.append(PIQA_test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer == option1.lower() or final_answer == option2.lower():
                    count += 1
                    if final_answer == option1.lower():
                        PIQA_test_data_item['extracted_answer'] = '1'
                    if final_answer == option2.lower():
                        PIQA_test_data_item['extracted_answer'] = '2'
                    if final_answer == sol12_content:
                        correct_count += 1
                        PIQA_test_data_correct_predict_list.append(PIQA_test_data_item)
                    else:
                        PIQA_test_data_mispredict_list.append(PIQA_test_data_item)
                else:
                    PIQA_test_data_mispredict_list.append(PIQA_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(PIQA_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(PIQA_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio



def calc_accuracy_PIQA_simple(PIQA_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    PIQA_test_data_mispredict_list = []
    PIQA_test_data_correct_predict_list = []
    # for i in range(len(PIQA_test_data_list)):
    #     PIQA_test_data_list[i]['pred'] = predict_list[i]
    
    for i in range(min(len(PIQA_test_data_list), len(predict_list))):
        PIQA_test_data_list[i]['pred'] = predict_list[i]

    for i, answer in enumerate(predict_list):
        gold_label = str(PIQA_test_data_list[i]['gold_label'])
        gold_label = gold_label.lower()

        if answer:
            final_answer = extract_option(answer)
            PIQA_test_data_item = PIQA_test_data_list[i]
            PIQA_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                PIQA_test_data_correct_predict_list.append(PIQA_test_data_item)
            else:
                PIQA_test_data_mispredict_list.append(PIQA_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    return accuracy, cover_ratio




def calc_accuracy_AQuaRAT(test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    test_data_mispredict_list = []
    test_data_correct_predict_list = []
    for i in range(len(test_data_list)):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        # abcd_content = test_data_list[i][gold_label]
        # gold_label = gold_label.lower()
        # a_content = test_data_list[i]['A']
        # b_content = test_data_list[i]['B']
        # c_content = test_data_list[i]['C']
        # d_content = test_data_list[i]['D']

        # abcd_content = abcd_content.strip().lower().rstrip('.')
        # a_content = a_content.strip().lower().rstrip('.')
        # b_content = b_content.strip().lower().rstrip('.')
        # c_content = c_content.strip().lower().rstrip('.')
        # d_content = d_content.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_answer_aquarat(answer)
            test_data_item = test_data_list[i]
            test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                test_data_correct_predict_list.append(test_data_item)
            else:
                # final_answer = extract_context_after_answer(answer)
                # if final_answer == a_content.lower() or final_answer == b_content.lower() or final_answer == c_content.lower() or final_answer == d_content.lower():
                #     count += 1
                #     if final_answer == a_content.lower():
                #         test_data_item['extracted_answer'] = 'A'
                #     if final_answer == b_content.lower():
                #         test_data_item['extracted_answer'] = 'B'
                #     if final_answer == c_content.lower():
                #         test_data_item['extracted_answer'] = 'C'
                #     if final_answer == d_content.lower():
                #         test_data_item['extracted_answer'] = 'D'
                #     if final_answer == abcd_content:
                #         correct_count += 1
                #         test_data_correct_predict_list.append(test_data_item)
                #     else:
                #         test_data_mispredict_list.append(test_data_item)
                # else:
                test_data_mispredict_list.append(test_data_item)
                
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_WINOGRANDE(WINOGRANDE_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    WINOGRANDE_test_data_mispredict_list = []
    WINOGRANDE_test_data_correct_predict_list = []
    for i in range(len(WINOGRANDE_test_data_list)):
        WINOGRANDE_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(WINOGRANDE_test_data_list[i]['gold_label'])
        op12_content = WINOGRANDE_test_data_list[i]['option'+gold_label]
        gold_label = gold_label.lower()
        option1 = WINOGRANDE_test_data_list[i]['option1']
        option2 = WINOGRANDE_test_data_list[i]['option2']

        op12_content = op12_content.strip().lower().rstrip('.')
        option1 = option1.strip().lower().rstrip('.')
        option2 = option2.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_option(answer)
            WINOGRANDE_test_data_item = WINOGRANDE_test_data_list[i]
            WINOGRANDE_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                WINOGRANDE_test_data_correct_predict_list.append(WINOGRANDE_test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer == option1.lower() or final_answer == option2.lower():
                    count += 1
                    if final_answer == option1.lower():
                        WINOGRANDE_test_data_item['extracted_answer'] = '1'
                    if final_answer == option2.lower():
                        WINOGRANDE_test_data_item['extracted_answer'] = '2'
                    if final_answer == op12_content:
                        correct_count += 1
                        WINOGRANDE_test_data_correct_predict_list.append(WINOGRANDE_test_data_item)
                    else:
                        WINOGRANDE_test_data_mispredict_list.append(WINOGRANDE_test_data_item)
                else:
                    WINOGRANDE_test_data_mispredict_list.append(WINOGRANDE_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(WINOGRANDE_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(WINOGRANDE_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_WINOGRANDE_simple(WINOGRANDE_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    WINOGRANDE_test_data_mispredict_list = []
    WINOGRANDE_test_data_correct_predict_list = []
    for i in range(len(WINOGRANDE_test_data_list)):
        WINOGRANDE_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(WINOGRANDE_test_data_list[i]['gold_label'])
        gold_label = gold_label.lower()
        if answer:
            final_answer = extract_option(answer)
            WINOGRANDE_test_data_item = WINOGRANDE_test_data_list[i]
            WINOGRANDE_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                WINOGRANDE_test_data_correct_predict_list.append(WINOGRANDE_test_data_item)
            else:
                WINOGRANDE_test_data_mispredict_list.append(WINOGRANDE_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)
    return accuracy, cover_ratio




def calc_accuracy_ECQA(ECQA_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    ECQA_test_data_mispredict_list = []
    ECQA_test_data_correct_predict_list = []
    for i in range(len(ECQA_test_data_list)):
        ECQA_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(ECQA_test_data_list[i]['gold_label'])
        gold_label_content = ECQA_test_data_list[i][gold_label]
        gold_label_content = gold_label_content.lower()
        option1 = ECQA_test_data_list[i]['1']
        option2 = ECQA_test_data_list[i]['2']
        option3 = ECQA_test_data_list[i]['3']
        option4 = ECQA_test_data_list[i]['4']
        option5 = ECQA_test_data_list[i]['5']

        gold_label_content = gold_label_content.strip().lower().rstrip('.')
        option1 = option1.strip().lower().rstrip('.')
        option2 = option2.strip().lower().rstrip('.')
        option3 = option3.strip().lower().rstrip('.')
        option4 = option4.strip().lower().rstrip('.')
        option5 = option5.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_option_1_to_5(answer)
            ECQA_test_data_item = ECQA_test_data_list[i]
            ECQA_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                ECQA_test_data_correct_predict_list.append(ECQA_test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer == option1.lower() or final_answer == option2.lower() or final_answer == option3.lower() or final_answer == option4.lower() or final_answer == option5.lower():
                    count += 1
                    if final_answer == option1.lower():
                        ECQA_test_data_item['extracted_answer'] = '1'
                    if final_answer == option2.lower():
                        ECQA_test_data_item['extracted_answer'] = '2'
                    if final_answer == option1.lower():
                        ECQA_test_data_item['extracted_answer'] = '3'
                    if final_answer == option2.lower():
                        ECQA_test_data_item['extracted_answer'] = '4'
                    if final_answer == option1.lower():
                        ECQA_test_data_item['extracted_answer'] = '5'
                    if final_answer == gold_label_content:
                        correct_count += 1
                        ECQA_test_data_correct_predict_list.append(ECQA_test_data_item)
                    else:
                        ECQA_test_data_mispredict_list.append(ECQA_test_data_item)
                else:
                    ECQA_test_data_mispredict_list.append(ECQA_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(ECQA_test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(ECQA_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio




def calc_accuracy_ECQA_simple(ECQA_test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    count = 0
    ECQA_test_data_mispredict_list = []
    ECQA_test_data_correct_predict_list = []
    for i in range(len(ECQA_test_data_list)):
        ECQA_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(ECQA_test_data_list[i]['gold_label'])
        if answer:
            final_answer = extract_option_1_to_5(answer)
            ECQA_test_data_item = ECQA_test_data_list[i]
            ECQA_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                ECQA_test_data_correct_predict_list.append(ECQA_test_data_item)
            else:
                ECQA_test_data_mispredict_list.append(ECQA_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)
    return accuracy, cover_ratio

def calc_accuracy_MMLU_AGI(test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    test_data_mispredict_list = []
    test_data_correct_predict_list = []
    for i in range(min(len(predict_list), len(test_data_list))):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        abcd_content = test_data_list[i][gold_label]
        gold_label = gold_label.lower()
        a_content = test_data_list[i]['A']
        b_content = test_data_list[i]['B']
        c_content = test_data_list[i]['C']
        d_content = test_data_list[i]['D']

        abcd_content = abcd_content.strip().lower().rstrip('.')
        a_content = a_content.strip().lower().rstrip('.')
        b_content = b_content.strip().lower().rstrip('.')
        c_content = c_content.strip().lower().rstrip('.')
        d_content = d_content.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_option_mmlu_agi(answer)
            test_data_item = test_data_list[i]
            test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label == final_answer.lower():
                correct_count += 1
                test_data_correct_predict_list.append(test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer.lower() == a_content.lower() or final_answer.lower() == b_content.lower() or final_answer.lower() == c_content.lower() or final_answer.lower() == d_content.lower():
                    count += 1
                    if final_answer == a_content.lower():
                        test_data_item['extracted_answer'] = 'A'
                    if final_answer == b_content.lower():
                        test_data_item['extracted_answer'] = 'B'
                    if final_answer == c_content.lower():
                        test_data_item['extracted_answer'] = 'C'
                    if final_answer == d_content.lower():
                        test_data_item['extracted_answer'] = 'D'
                    if final_answer == abcd_content:
                        correct_count += 1
                        test_data_correct_predict_list.append(test_data_item)
                    else:
                        test_data_mispredict_list.append(test_data_item)
                else:
                    test_data_mispredict_list.append(test_data_item)
                
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(test_data_list)

    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_correct_predictions.json", 'w') as file:
    #     json.dump(test_data_correct_predict_list, file, indent=4)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{task_name}_mispredictions.json", 'w') as file:
    #     json.dump(test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_MMLU_AGI_simple(test_data_list, predict_list, output_folder_name, task_name=''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    test_data_mispredict_list = []
    test_data_correct_predict_list = []
    for i in range(min(len(predict_list), len(test_data_list))):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        gold_label = gold_label.lower()
        if answer:
            final_answer = extract_option_mmlu_agi(answer)
            test_data_item = test_data_list[i]
            test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                test_data_correct_predict_list.append(test_data_item)
            else:
                test_data_mispredict_list.append(test_data_item)
                
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(test_data_list)

    return accuracy, cover_ratio

def extract_nli_answer(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    neutral_count = 0
    entail_count = 0
    contradiction_count = 0
    for item in text.split():
        if 'neutr' in item:
            neutral_count += 1
        if 'entail' in item:
            entail_count += 1
        if 'contrad' in item:
            contradiction_count += 1
    max_num = max(neutral_count, entail_count, contradiction_count)

    final_answer = ''
    if max_num == 0:
        final_answer = 'null'
    else:
        if max_num == neutral_count:
            final_answer = 'neutral'
            if max_num == entail_count or max_num == contradiction_count:
                final_answer = 'null'
        elif max_num == entail_count:
            final_answer = 'entailment'
            if max_num == neutral_count or max_num == contradiction_count:
                final_answer = 'null'
        else:
            final_answer = 'contradiction'
            if max_num == neutral_count or max_num == entail_count:
                final_answer = 'null'
    return final_answer


def extract_option(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    if "1" in text or "2" in text:
        index_1 = text.find("1")
        index_2 = text.find("2")
        if index_1 != -1 and (index_2 == -1 or index_1 < index_2):
            return "1"
        elif index_2 != -1:
            return "2"
    else:
        return "null"
    

def find_first_number(text, numbers=[1, 2, 3, 4, 5]):
    # Initialize variables to store the first number and its index
    first_number = None
    first_index = len(text)  # Start with the highest possible index

    # Loop through the numbers to find which appears first
    for number in numbers:
        index = text.find(str(number))
        # Check if the number is found and appears before the current first number
        if index != -1 and index < first_index:
            first_number = str(number)
            first_index = index

    # Return the first number, or "null" if none of the numbers are found
    return first_number if first_number else "null"


def extract_option_1_to_5(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    
    return_text = find_first_number(text)
    return return_text
    
    
def extract_option_mmlu_agi(text):
    text = text.lower()
    abcd = extract_answer_mmlu_agi(text)
    if abcd != "null":
        return abcd
    else:
        return "null"

def extract_answer_mmlu_agi(text):
    pattern = r'^\s*(?:\(([A-Da-d])\)|([A-Da-d])\.?)\s*$'
    
    # Search for a match using the defined pattern
    match = re.search(pattern, text)

    # If a match is found, return it in uppercase
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    pattern_direct = r'(?:answer:|the\sanswer\sis)\s*\b([A-D])\b'
    match_direct = re.search(pattern_direct, text, re.IGNORECASE)
    
    if match_direct:
        return match_direct.group(1).upper()

    # If no direct mention is found, look for the first occurrence of A, B, C, or D after 'answer'
    pattern_fallback = r'\banswer\b[^A-D]*\b([A-D])\b'
    match_fallback = re.search(pattern_fallback, text, re.IGNORECASE)
    
    if match_fallback:
        return match_fallback.group(1).upper()

    # If no match is found by either pattern, return "null"
    return "null"











def calc_accuracy_MMLU_PRO(test_data_list, predict_list):
    correct_count = 0
    cover_ratio = 0
    count = 0
    for i in range(min(len(predict_list), len(test_data_list))):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        gold_label = gold_label.lower()

        if answer:
            final_answer = extract_option_mmlu_pro(answer)
            test_data_item = test_data_list[i]
            test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label == final_answer.lower():
                correct_count += 1
                
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(test_data_list)

    return accuracy, cover_ratio

def extract_option_mmlu_pro(text):
    text = text.lower()
    abcd = extract_answer_mmlu_pro(text)
    if abcd != "null":
        return abcd
    else:
        return "null"

def extract_answer_mmlu_pro(text):
    pattern = r'^\s*(?:\(([A-Ja-j])\)|([A-Ja-j])\.?)\s*$'

    # Search for a match using the defined pattern
    match = re.search(pattern, text)

    # If a match is found, return it in uppercase
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    pattern_direct = r'(?:answer:|the\sanswer\sis)\s*\b([A-J])\b'
    match_direct = re.search(pattern_direct, text, re.IGNORECASE)
    
    if match_direct:
        return match_direct.group(1).upper()

    # If no direct mention is found, look for the first occurrence of A, B, C, or D after 'answer'
    pattern_fallback = r'\banswer\b[^A-J]*\b([A-J])\b'
    match_fallback = re.search(pattern_fallback, text, re.IGNORECASE)
    
    if match_fallback:
        return match_fallback.group(1).upper()

    # If no match is found by either pattern, return "null"
    return "null"

def calc_accuracy_HELLASWAG(test_data_list, predict_list):
    correct_count = 0
    count = 0
    for i in range(len(test_data_list)):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        if answer:
            final_answer = extract_option_1_to_4(answer)
            ECQA_test_data_item = test_data_list[i]
            ECQA_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)
    return accuracy, cover_ratio

def extract_option_1_to_4(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    
    return_text = find_first_number(text, numbers=[1, 2, 3, 4])
    return return_text

def calc_accuracy_ARC_CHALLENGE(test_data_list, predict_list):
    correct_count = 0
    cover_ratio = 0
    count = 0
    for i in range(min(len(predict_list), len(test_data_list))):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        gold_label = gold_label.lower()

        if answer:
            final_answer = extract_option_ARC_CHALLENGE(answer)
            test_data_item = test_data_list[i]
            test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label == final_answer.lower():
                correct_count += 1
                
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(test_data_list)

    return accuracy, cover_ratio

def extract_option_ARC_CHALLENGE(text):
    text = text.lower()
    abcd = extract_answer_ARC_CHALLENGE(text)
    if abcd != "null":
        return abcd
    else:
        return "null"

def extract_answer_ARC_CHALLENGE(text):
    pattern = r'^\s*(?:\(([A-Da-d])\)|([A-Da-d])\.?)\s*$'

    # Search for a match using the defined pattern
    match = re.search(pattern, text)

    # If a match is found, return it in uppercase
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    pattern_direct = r'(?:answer:|the\sanswer\sis)\s*\b([A-D])\b'
    match_direct = re.search(pattern_direct, text, re.IGNORECASE)
    
    if match_direct:
        return match_direct.group(1).upper()

    # If no direct mention is found, look for the first occurrence of A, B, C, or D after 'answer'
    pattern_fallback = r'\banswer\b[^A-D]*\b([A-D])\b'
    match_fallback = re.search(pattern_fallback, text, re.IGNORECASE)
    
    if match_fallback:
        return match_fallback.group(1).upper()

    # If no match is found by either pattern, return "null"
    return "null"

def calc_accuracy_THEOREMQA(test_data_list, predict_list):
    correct_count = 0
    cover_ratio = 0
    count = 0
    for i in range(len(predict_list)):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gt = test_data_list[i]['gold_label']
        final_answer = extract_text_span(answer)

        if final_answer != 'null':
            count += 1

        if final_answer[0] == '"':  # remove the last period
            final_answer = final_answer[1:]

        if final_answer[-1] == '"':  # remove the last period
            final_answer = final_answer[:-1]
        
        if final_answer[-1] == '.':  # remove the last period
            final_answer = final_answer[:-1]


        if gt[0] == '"':  # remove the last period
            gt = gt[1:]

        if gt[-1] == '"':  # remove the last period
            gt = gt[:-1]

        if gt[-1] == '.':  # remove the last period
            gt = gt[:-1]


        if gt.lower() == final_answer.lower():
            correct_count += 1
        
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    return accuracy, cover_ratio


def run_dynamic_test_with_timeout(test_case, completion, timeout_seconds=50):
    def run_test(completion, test_case, namespace):
        local_namespace = {}  # Local namespace for the executed code
        global_namespace = {}  # Empty global namespace, or can use globals()

        # Clean up the completion code
        completion = completion.replace("\r", "").replace("\t", " ").strip()

        try:
            # Ensure function definitions are executed in global_namespace
            exec(completion, global_namespace)  # Executes in the global_namespace
            PASS = True
            for test_case_item in test_case:
                try:
                    # Execute each test case
                    exec(test_case_item, global_namespace, local_namespace)
                except Exception as e:
                    PASS = False
                    print(f"Error executing test case: {test_case_item}")
                    print(f"Exception: {e}")
                    break  # Stop at the first exception
            namespace['PASS'] = PASS
        except Exception as e:
            namespace['PASS'] = False
            print(f"Error during execution: {e}")

    # Use Manager from multiprocessing to create a shared namespace
    with Manager() as manager:
        namespace = manager.dict()
        # Define and start a new process for running the test with the provided code and test cases
        process = Process(target=run_test, args=(completion, test_case, namespace))
        process.start()

        # Wait for the process to complete or for the timeout
        process.join(timeout_seconds)

        # If the process is still alive after the timeout, it means it's likely stuck in an infinite loop
        if process.is_alive():
            process.terminate()  # Terminate the stuck process
            process.join()  # Ensure process resources are cleaned up
            return False  # Return False to indicate the test did not pass (due to timeout)

        # Fetch and return the result from the shared namespace
        return namespace.get('PASS', False)
    
# Example of using this with the provided test cases
def calc_accuracy_MBPP(MBPP_test_data_list, predict_list):
    cover_ratio = 0
    test_data_mispredict_list = []
    test_data_correct_predict_list = []
    correct_count = 0
    for i, item in enumerate(MBPP_test_data_list):
        completion = predict_list[i]
        original_completion = predict_list[i]
        completion = completion.lstrip()

        passed = run_dynamic_test_with_timeout(item['test_list'], completion)

        if passed:
            correct_count += 1
            item['prediction'] = original_completion
            test_data_correct_predict_list.append(item)
        else:
            item['prediction'] = original_completion
            test_data_mispredict_list.append(item)

    cover_ratio = int(1)
    accuracy = correct_count / len(MBPP_test_data_list)
    return accuracy, cover_ratio




def extract_answer_aquarat(text):
    pattern = r'^\s*(?:\(([A-Ea-e])\)|([A-Ea-e])\.?)\s*$'
    
    # Search for a match using the defined pattern
    match = re.search(pattern, text)

    # If a match is found, return it in uppercase
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    pattern_direct = r'(?:answer:|the\sanswer\sis)\s*\b([A-E])\b'
    match_direct = re.search(pattern_direct, text, re.IGNORECASE)
    
    if match_direct:
        return match_direct.group(1).upper()

    # If no direct mention is found, look for the first occurrence of A, B, C, or D after 'answer'
    pattern_fallback = r'\banswer\b[^A-E]*\b([A-E])\b'
    match_fallback = re.search(pattern_fallback, text, re.IGNORECASE)
    
    if match_fallback:
        return match_fallback.group(1).upper()

    # If no match is found by either pattern, return "null"
    return "null"


def extract_context_after_answer(text):
    
    # Convert the text to lowercase to ignore case
    text_lower = text.lower()

    # Regular expression to find all occurrences of 'answer:' followed by any content
    matches = re.findall(r'answer:\s*(.*)', text_lower)

    # Extract the last occurrence, if any
    if matches:
        last_answer = matches[-1]
        return_null_list = ['Neither Option 1', 'correct answer is not provided', 'There is no correct answer', 'The given options are incorrect.']
        for return_null in return_null_list:
            return_null = return_null.lower()
            if return_null in last_answer.lower():
                return 'null'
        # Remove a period at the end if present
        if last_answer.endswith('.'):
            last_answer = last_answer[:-1]
        last_answer = last_answer.lower().strip()
        return last_answer
    else:    
        return "null"

def find_smallest_index(index_1, index_2, index_3, index_4):
    # Assuming index_1, index_2, index_3, and index_4 are defined
    indices = [index_1, index_2, index_3, index_4]

    # Filter out any indices that are -1, indicating the string wasn't found
    filtered_indices = [index for index in indices if index != -1]

    # Find the smallest index if there are any indices left after filtering
    if filtered_indices:
        smallest_index = min(filtered_indices)
    else:
        smallest_index = None  # or -1, depending on how you want to handle no matches

    # smallest_index now holds the smallest index that isn't -1, or None if all were -1
    return smallest_index
    
def extract_bool(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    if "false" in text or "true" in text or "yes" in text or "no" in text:
        index_1 = text.find("true")
        index_2 = text.find("false")
        index_3 = text.find("yes")
        index_4 = text.find("no")

        smallest_index = find_smallest_index(index_1, index_2, index_3, index_4)
        if not smallest_index and smallest_index != 0:
            return 'null'
        if smallest_index == index_1 or smallest_index == index_3:
            return "true"
        else:
            return "false"
    else:
        return "null"
    
def extract_text_span(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    if found_answer_num != 0:
        if text != '':
            return text
        else:
            return 'null'
    else:
        return 'null'

def EVALUATION_LLAMA_FACTORY(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = '', task_name = '', merged_base_model_dir = '', data_name = '', LLAMA_FACTORY_DIRECTORY = ''):
    # if test_task_name.lower() == 'plan_bench':
    #     test_data_list_temp = []
    #     for item in test_data_list:
    #         test_data_list_temp += item
    #     test_data_list = test_data_list_temp
    predict_list = do_predict_llama_factory_unify(test_data_list, output_folder_name, test_config, file_name, check_point_folder_name = check_point_folder_name, merged_base_model_dir = merged_base_model_dir, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)

    accuracy, cover_ratio = Check_Correctness(predict_list, test_data_list, test_task_name, output_folder_name, task_name = task_name)
    return accuracy, cover_ratio

def do_predict_llama_factory_unify(data_list, output_folder_name, test_config, file_name, check_point_folder_name = '', merged_base_model_dir = '', data_name = '', LLAMA_FACTORY_DIRECTORY = ''):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    sys.path.append(f'{LLAMA_FACTORY_DIRECTORY}')
    # from src import train_bash

    file_name = file_name.replace('_log', '')
    put_json_list_to_data_info(data_list, data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)
    
    sys.path.append(parent_dir)
    output_folder_name = f'{HOME_DIRECTORY}/output/{file_name}'
    if merged_base_model_dir == '':
        model_path = f"{MODEL_DIRECTORY}/{test_config['model_name']}"
    else:
        model_path = merged_base_model_dir

    if not os.path.exists(f"{output_folder_name}"):
        os.makedirs(f"{output_folder_name}")
    
    if test_config['device_num'] > 1:
        start_line = 'accelerate launch'
    else:
        start_line = 'python'
    cmd = [
        # "accelerate launch",
        start_line,
        f"{LLAMA_FACTORY_DIRECTORY}/src/train_bash.py",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--do_predict", str(True),
        "--dataset", data_name,
        "--template", test_config['template'],
        "--finetuning_type", test_config['finetuning_type'],
        "--max_length", str(test_config['max_length']),
        "--cutoff_len", str(test_config['max_input_length']),
        "--output_dir", f'{output_folder_name}',
        "--per_device_eval_batch_size", str(test_config['per_device_eval_batch_size']),
        "--max_new_tokens", str(test_config['max_new_tokens']),
        "--predict_with_generate", str(True),
        "--overwrite_cache",
        "--fp16"
    ]

    if 'load_in_8bit' in test_config:
        cmd += ["--quantization_config", str(test_config['load_in_8bit'])]

    if check_point_folder_name != '':
        cmd += ['--adapter_name_or_path', check_point_folder_name]

    if 'seed_num' in test_config:
        cmd += ['--seed', str(test_config['seed_num'])]
    
    if 'do_sample' in test_config:
        cmd += ['--do_sample', str(test_config['do_sample'])]

    if 'temperature' in test_config:
        cmd += ['--temperature', str(test_config['temperature'])]
    
    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)

    source_file = f"{output_folder_name}/generated_predictions.jsonl"
    predict_list = []
    with open(source_file, 'r') as file:
        json_list = list(file)
        for line in json_list:
            data = json.loads(line)
            prediction = data.get("predict", "No label key found.")
            predict_list.append(prediction)

    return predict_list


def Check_Correctness(predict_list, test_data_list, test_task_name, output_folder_name, task_name = '', extract_gold_label_as_gt = False, simple_evaluation = False):
    question_list = []
    groundtruth_list = []
    for i in range(len(test_data_list)):
        question = test_data_list[i]['question']
        question_list.append(question)
        if not extract_gold_label_as_gt:
            if test_task_name.lower() == 'piqa' or test_task_name.lower() == 'boolq' or test_task_name.lower() == 'winogrande' or test_task_name.lower() == 'ecqa' or test_task_name.lower() == 'squad' or test_task_name.lower() == 'aquarat' or ('plan_bench' in test_task_name.lower()) or test_task_name.lower() == 'drop':
                groundtruth_list.append(test_data_list[i]['gold_label'])
            elif 'math' in test_task_name.lower() or 'gsm8k' in test_task_name.lower():
                groundtruth_list.append(test_data_list[i]['numerical_final_answer'])
            else:
                groundtruth_list.append(test_data_list[i]['answer'])
        else:
            groundtruth_list.append(test_data_list[i]['gold_label'])

    if 'gsm8k' in test_task_name.lower():
        accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, predict_list, groundtruth_list, output_folder_name, task_name = task_name)
    if 'math' in test_task_name.lower():
        accuracy, cover_ratio = calc_accuracy_MATH(question_list, predict_list, groundtruth_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'api_bank':
        if not simple_evaluation:
            accuracy, cover_ratio, lv12_accuracy, lv3_accuracy = calc_accuracy_API_BANK(test_data_list, predict_list, output_folder_name, task_name = task_name)
        else:
            accuracy, cover_ratio, lv12_accuracy, lv3_accuracy = calc_accuracy_API_BANK_simple(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'anli':
        accuracy, cover_ratio = calc_accuracy_ANLI(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'mnli':
        accuracy, cover_ratio = calc_accuracy_MNLI(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'esnli':
        accuracy, cover_ratio = calc_accuracy_ESNLI(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'scitail':
        accuracy, cover_ratio = calc_accuracy_SCITAIL(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'piqa':
        if not simple_evaluation:
            accuracy, cover_ratio = calc_accuracy_PIQA(test_data_list, predict_list, output_folder_name, task_name = task_name)
        else:
            accuracy, cover_ratio = calc_accuracy_PIQA_simple(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'boolq':
        accuracy, cover_ratio = calc_accuracy_BOOLQ(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'squad':
        if not simple_evaluation:
            accuracy, cover_ratio = calc_accuracy_SQUAD(test_data_list, predict_list, output_folder_name, task_name = task_name)
        else:
            accuracy, cover_ratio = calc_accuracy_SQUAD_simplified(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'drop':
        accuracy, cover_ratio = calc_accuracy_SQUAD_simplified(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'winogrande':
        if not simple_evaluation:
            accuracy, cover_ratio = calc_accuracy_WINOGRANDE(test_data_list, predict_list, output_folder_name, task_name = task_name)
        else:
            accuracy, cover_ratio = calc_accuracy_WINOGRANDE_simple(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if 'plan_bench' in test_task_name.lower():
        accuracy, cover_ratio = calc_accuracy_PLAN_BENCH(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'mmlu' or test_task_name.lower() == 'mmlu_moral_scenarios' or test_task_name.lower() == 'agieval':
        if not simple_evaluation:
            accuracy, cover_ratio = calc_accuracy_MMLU_AGI(test_data_list, predict_list, output_folder_name, task_name = task_name)
        else:
            accuracy, cover_ratio = calc_accuracy_MMLU_AGI_simple(test_data_list, predict_list, output_folder_name, task_name = task_name)
    if test_task_name.lower() == 'ecqa':
        if not simple_evaluation:
            accuracy, cover_ratio = calc_accuracy_ECQA(test_data_list, predict_list, output_folder_name, task_name = task_name)
        else:
            accuracy, cover_ratio = calc_accuracy_ECQA_simple(test_data_list, predict_list, output_folder_name, task_name = task_name)
    
    if test_task_name.lower() == 'mmlu_pro':
        accuracy, cover_ratio = calc_accuracy_MMLU_PRO(test_data_list, predict_list)
    if test_task_name.lower() == 'mmlu_pro_law':
        accuracy, cover_ratio = calc_accuracy_MMLU_PRO(test_data_list, predict_list)
    if test_task_name.lower() == 'hellaswag':
        accuracy, cover_ratio = calc_accuracy_HELLASWAG(test_data_list, predict_list)
    if test_task_name.lower() == 'arc_challenge':
        accuracy, cover_ratio = calc_accuracy_ARC_CHALLENGE(test_data_list, predict_list)
    if test_task_name.lower() == 'theoremqa':
        accuracy, cover_ratio = calc_accuracy_THEOREMQA(test_data_list, predict_list)
    if test_task_name.lower() == 'mbpp':
        accuracy, cover_ratio = calc_accuracy_MBPP(test_data_list, predict_list)

    if not simple_evaluation:
        file_name = 'record'
        log_line = f'{task_name} Evaluation for ' + test_task_name
        write_log(file_name, output_folder_name, log_line)
        log_line = f'{task_name} Accuracy: ' + str(accuracy) #+ ', Cover Ratio: ' + str(cover_ratio) 

        if test_task_name.lower() == 'api_bank':
            log_line += f'        lv12_accuracy: {lv12_accuracy}        lv3_accuracy: {lv3_accuracy}'
        write_log(file_name, output_folder_name, log_line)
    return accuracy, cover_ratio




def Find_correct_initial_prediction(task_name, predict_list, test_data_list, gold_label_list, output_folder_name, test_config, LLAMA_FACTORY_DIRECTORY = ''):
    predict_list = predict_list['initial_prediction']
    question_list = []
    groundtruth_list = gold_label_list
    def add_gold_label(item, groundtruth_item):
        item['gold_label'] = groundtruth_item
        return item
    import re
    def remove_after_final_answer(input_string):
        # Use regex to remove everything after '\n\nFinal Answer:'
        result = re.sub(r'\n\nFinal Answer:.*', '', input_string, flags=re.DOTALL)
        return result

    def extract_plan_content(input_string):
        # Regular expression to match content between [plan] and [plan end]
        pattern = r'\[PLAN\](.*?)\[PLAN END\]'
        
        # Find all occurrences of the pattern
        matches = re.findall(pattern, input_string, flags=re.DOTALL)
        
        # If there's exactly one match, return it
        if len(matches) == 1:
            return matches[0]
        else:
            input_string = remove_after_final_answer(input_string)
            if ':' in input_string:
                # Extract content after the last ':'
                last_colon_content = input_string.rsplit(':', 1)[-1]
                return last_colon_content.strip()
            else:
                # If no colon is found, return a default message or the whole string
                return input_string

        
    def extract_excution_result(input_string):
        # Regular expression to match content between [plan] and [plan end]
        pattern = r'\[PLAN\](.*?)\[PLAN END\]'
        
        # Find all occurrences of the pattern
        matches = re.findall(pattern, input_string, flags=re.DOTALL)
        
        # If there's exactly one match, return it
        if len(matches) == 1:
            return matches[0]
        else:
            input_string = remove_after_final_answer(input_string)
            if ':' in input_string:
                # Extract content after the last ':'
                last_colon_content = input_string.rsplit(':', 1)[-1]
                return last_colon_content.strip()
            else:
                # If no colon is found, return a default message or the whole string
                return input_string

    # 'not work for drop'
    for i in range(len(test_data_list)):
        question = test_data_list[i]['question']
        question_list.append(question)


    a = 1
    
    if 'mbpp' in task_name:
        initial_prediction_dict = {}
        correct_initial_prediction = []
        correct_index_list = []
        for current_index, item in enumerate(test_data_list):
            accuracy, cover_ratio = calc_accuracy_MBPP([test_data_list[current_index]], [predict_list[current_index]])
            if accuracy == 1:
                correct_index_list.append(current_index)
                correct_initial_prediction.append(predict_list[current_index])  
    else:
        data_list = []
        if 'arc_challenge' in task_name or 'mmlu' in task_name or 'agieval' in task_name:
            gold_label_type = 'A/B/C/D'
        elif 'piqa' in task_name or 'winogrande' in task_name:
            gold_label_type = '1/2'
        elif 'squad' in task_name:
            gold_label_type = 'text_span'
        elif 'gsm8k' in task_name or 'math' in task_name:
            gold_label_type = 'number'
        elif 'ecqa' in task_name:
            gold_label_type = '1/2/3/4/5'
        elif 'esnli' in task_name:
            gold_label_type = 'Entailment/Neutral/Contradiction'
        elif 'boolq' in task_name:
            gold_label_type = 'True/False'
        elif 'mmlu_pro' in task_name:
            gold_label_type = 'A/B/C/D/E/F/G/H/I/J'
        elif 'hellaswag' in task_name:
            gold_label_type = '1/2/3/4'
        elif 'drop' in task_name:
            gold_label_type = 'number_or_text_span'
        elif 'api_bank' in task_name:
            gold_label_type = 'API-request'
        elif 'plan_bench' in task_name:
            gold_label_type = 'plan'
        else:
            a = 1

    #     if 'plan_bench' in task_name:
    #         for i in range(len(question_list)):
    #             extracted_plan = extract_plan_content(predict_list[i])
    #             if extracted_plan:
    #                 question = \
    # f"""Given the groundtruth and the extracted plan that is written in natural language, does the extractdd plan describe the similar plan as the groundtruth? when I say similar, I mean at least 80% of the extracted plan should matches the groundtruth. 

    # The extracted plan is: "{extracted_plan}"

    # The groundtruth is: "{groundtruth_list[i]}"
    # Directly say Yes or No.
    # """
    #                 temp = {}
    #                 temp['question'] = question
    #                 temp['input'] = ''
    #                 temp['answer'] = ''
    #                 data_list.append(temp)
    #             else:
    #                 question = "say wrong"
    #     else:
        for i in range(len(question_list)):
            question = \
f"""Given the question and the prediction, what is the final answer predicted by the prediction? 

The question is "{question_list[i]}

The prediction is "{predict_list[i]}"

Directly output {gold_label_type} without saying anything else.
"""
            temp = {}
            temp['question'] = question
            temp['input'] = ''
            temp['answer'] = ''
            data_list.append(temp)

        initial_prediction_dict = {}
        extract_predict = ''
        # data_list = data_list[:6]
        if 'math' in task_name or 'gsm8k' in task_name:
            a = 1
        else:
            if 'plan_bench' not in task_name:
                extract_predict = do_predict_llama_factory_unify(data_list, output_folder_name, test_config, 'check_correctness', data_name = 'check_correctness', LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)

        correct_initial_prediction = []
        correct_index_list = []
        if not 'plan_bench' in task_name:
            data_list_temp = []
            if 'math' in task_name or 'gsm8k' in task_name:
                judge_list = []

                for i, predict_iiitem in enumerate(predict_list):
                    current_gt_item = add_gold_label(test_data_list[i], gold_label_list[i])
                    accuracy, cover_ratio = Check_Correctness([predict_iiitem], [current_gt_item], task_name, output_folder_name, task_name = 'error_correction', extract_gold_label_as_gt = True, simple_evaluation = True)
                    if accuracy > 0:
                        correct_initial_prediction.append(predict_list[i] + '\n\nFinal Answer: ' + groundtruth_list[i])
                        correct_index_list.append(i)
            else:
                for i in range(len(extract_predict)):
                    question = \
f"""You are a classifier. Given the question and the groundtruth, is the prediction correct?  

The prediction is: "{extract_predict[i]}"

The groundtruth is: "{groundtruth_list[i]}"

The question is: "{question_list[i]}"

Directly say Yes or No without any explaination.
"""
                    temp = {}
                    temp['question'] = question
                    temp['input'] = ''
                    temp['answer'] = ''
                    data_list_temp.append(temp)
                judge_list = do_predict_llama_factory_unify(data_list_temp, output_folder_name, test_config, 'check_correctness', data_name = 'check_correctness', LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)

            if 'math' in task_name or 'gsm8k' in task_name:
                a = 1
            else:
                for iii, judge_item in enumerate(judge_list):
                    if 'yes' in judge_item.lower():
                        # if 'drop' in task_name:
                        if 'api_bank' in task_name:
                            predict_item = predict_list[iii]
                            predict_item_list = predict_item.split()
                            correct_initial_prediction.append(predict_list[iii] + '\n\nFinal Answer: ' + groundtruth_list[iii])
                            correct_index_list.append(iii)
                        else:
                            correct_initial_prediction.append(predict_list[iii] + '\n\nFinal Answer: ' + groundtruth_list[iii])
                            correct_index_list.append(iii)
                        # else:
                        #     extracted_plan = '\n\nFinal Answer: ' + extract_plan_content(groundtruth_list[i])
                        #     accuracy, cover_ratio = Check_Correctness([extracted_plan], [test_data_list[i]], task_name, output_folder_name, task_name = 'error_correction', extract_gold_label_as_gt = True, simple_evaluation = True)
                        #     # if ' yes ' in item.lower() and ' no ' not in item.lower():
                        #     if accuracy > 0:
                        #         correct_initial_prediction.append(predict_list[i] + '\n\nFinal Answer: ' + groundtruth_list[i])
                        #         correct_index_list.append(i)
                # else:
                #     for i, item in enumerate(extract_predict):
                #         gt = groundtruth_list[i].lower()
                #         if item.lower() == gt:
                #             correct_initial_prediction.append(predict_list[i] + '\n\nFinal Answer: ' + groundtruth_list[i])
                #             correct_index_list.append(i)
                
        else:
            put_json_list_to_data_info(data_list, output_folder_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)
            parent = f'{HOME_DIRECTORY}/output/{output_folder_name}'
            pp_ = f'{parent}/intermediate_results'
            if not os.path.exists(f"{pp_}"):
                os.makedirs(f"{pp_}")
                
            for i, item in enumerate(predict_list):
                if  'plan_bench_verification' in task_name:
                    extracted_plan = '\n\nFinal Answer: ' + item
                elif 'plan_bench_execution' in task_name:
                    extracted_plan = '\n\nFinal Answer: ' + extract_excution_result(item)
                else:
                    extracted_plan = '\n\nFinal Answer: ' + extract_plan_content(item)
                accuracy, cover_ratio = Check_Correctness([extracted_plan], [test_data_list[i]], task_name, output_folder_name, task_name = 'error_correction', extract_gold_label_as_gt = True, simple_evaluation = True)
                # if ' yes ' in item.lower() and ' no ' not in item.lower():
                if accuracy > 0:
                    # correct_initial_prediction.append(predict_list[i])# + '\n\nFinal Answer: ' + groundtruth_list[i])
                    if groundtruth_list[i] == '':
                        groundtruth_list[i] = '()'
                    correct_initial_prediction.append(predict_list[i] + '\n\nFinal Answer: ' + groundtruth_list[i])
                    correct_index_list.append(i)

    initial_prediction_dict['initial_prediction'] = correct_initial_prediction
    initial_prediction_dict['correct_index'] = correct_index_list
    return initial_prediction_dict

def find_last_boxed_number_with_simple_format(text):
    # Regex pattern to match \boxed{17}, \boxed{-17}, \boxed{1.7}, \boxed{-5/4}
    pattern = r'\\boxed{(-?\d+(\.\d+)?|-\d+/\d+)}'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Check if there are any matches
    if matches:
        # Select the last match found
        last_match = matches[-1][0]  # [-1] gets the last match, [0] gets the full match ignoring capturing groups
        return last_match
    else:
        # Return None or an appropriate value if no matches are found
        return None

def evaluate_expression_(expr):
    if 'sqrt' in expr or '^' in expr or '(' in expr:
        return False
    try:
        # Handle LaTeX-style fractions and square roots
        expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        # expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
        expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

        # Evaluate the expression
        result = eval(expr)
        
        return float(result)
    except:
        return False

def extract_after_last_occurrence(text, keyword):
    # Find the last occurrence of the keyword
    last_index = text.rfind(keyword)
    if last_index == -1:
        return text # Return a message if the keyword is not found
    else:
        a = 1
    # Extract the context after the keyword
    extracted_text = text[last_index + len(keyword):]
    return extracted_text.strip()  # Strip any leading or trailing whitespace

def eval_MATH_correctness(predict_item, correct_number):


    def extract_boxed_content(s):
        start = s.rfind('\\boxed{')
        if start == -1:
            return None
        
        end = s.rfind('}')
            
        if end != 0:
            answer = s[start + 7 : end]
            return answer  # 7 is the length of '\\boxed{'
    
    def evaluate_expression_try_best(expr):
        try:
            # Handle LaTeX-style fractions and square roots
            expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
            expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
            expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
            expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

            expr = re.sub(r'(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)', r'(\1) / (\2)', expr)



            # Evaluate the expression
            result = eval(expr)
            result = float(result)
            return str(result)
        except:
            return False

    
    def extract_last_number(text):
        # New pattern to include LaTeX-style expressions
        # pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\}|\\dfrac\{.*?\}\{.*?\}|\\cfrac\{.*?\}\{.*?\}|\\sqrt\{.*?\})'
        pattern = r'(-?\d+\/-?\d+|-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\}|\\dfrac\{.*?\}\{.*?\}|\\cfrac\{.*?\}\{.*?\}|\\sqrt\{.*?\})'


        founded_text = extract_boxed_content(text)
        if founded_text:
            if '\\frac' in founded_text or '\\dfrac' in founded_text or '\\cfrac' in founded_text or '\\sqrt' in founded_text or '/' in founded_text:
                extracted_num = evaluate_expression_try_best(founded_text)
                if not extracted_num:
                    return -3333333333333 
                else:
                    return extracted_num
            else: 
                text = founded_text

        # Find all numbers and expressions in the string
        all_numbers = re.findall(pattern, text)

        # Process the last number or expression
        if all_numbers:
            number = all_numbers[-1]
            # Evaluate LaTeX-style expressions
            if '\\frac' in number or '\\dfrac' in number or '\\cfrac' in number or '\\sqrt' in number or '/' in number:
                extracted_num = evaluate_expression_try_best(number)
                if not extracted_num:
                    return -3333333333333 
                else:
                    return extracted_num
            
            # Handle percentages and remove commas from numbers
            is_percentage = '%' in number
            number = number.replace('%', '').replace(',', '')
            
            # Convert to float and adjust for percentage if needed
            number = float(number)
            if is_percentage:
                number /= 100

            return str(number)
        else:
            return -3333333333333 


    extracted_final_answer = extract_last_number(predict_item)
    correct_number = extract_last_number(correct_number)

    if extracted_final_answer == correct_number:
    
        return True
    else:
        return False
