import json
import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
from tool_manager import ToolManager
from api_call_extraction import parse_api_call, get_api_call
import logging
from rouge import Rouge

def split_by_uppercase(s):
    return ''.join([' ' + c if c.isupper() else c for c in s]).strip()

def calculate_rouge_l_score(reference, hypothesis):
    rouge = Rouge()
    if hypothesis == '':
        return 0
    scores = rouge.get_scores(hypothesis, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score

def eval_api_bank(API_BANK_test_data_list, home_directory):

    lv1_2_correctness = 0
    lv1_2_total = 0

    lv3_correctness = 0
    lv3_total = 0
    
    if_api = True
    # if_api = False
    gt_path = f'{home_directory}/dataset/API_BANK/test/test-data_level-3.json'
    # gt_path = f'{home_directory}/DAMO-ConvAI/api-bank/test/level-3.json'
    tool_manager = ToolManager(f'{home_directory}/DAMO_ConvAI/api_bank/lv3_apis')
    # with open(pred_path, 'r') as f:
    #     preds = [json.loads(line) for line in f.readlines()]
        # preds = json.load(f)
    preds = API_BANK_test_data_list
    with open(gt_path, 'r') as f:
        gts = json.load(f)

    # if_api = 'API-Request:' in preds[0]['output']

    if if_api:
        total_num = len(preds)
        correct_num = 0
        errored_sample_ids = []
        tool_search_error_num = 0
    else:
        rougel_scores = []
    for pred_id, pred in enumerate(preds):
        if if_api:
            sample_id = pred['sample_id']
            # if sample_id in errored_sample_ids:
            #     continue
            api_id = pred['api_id']
            gt = gts[sample_id]['apis'][api_id]
            gt_api_name = gt['api_name']
            # gt_result = gt['answer']
            gt_result = gt['output']

            if gt_result['api_name'] == 'ToolSearcher':
                lv3_total += 1
            else: 
                lv1_2_total += 1
            pred_api_call = get_api_call(pred['pred'])
            if not pred_api_call:
                logging.warning('No api call found in pred: {}'.format(pred_id))
                errored_sample_ids.append(sample_id)
                continue
            try:
                pred_api_name, pred_param_dict = parse_api_call(pred_api_call)
            except Exception as e:
                logging.warning('Parse api call error: {} {}'.format(str(e), pred_id))
                errored_sample_ids.append(sample_id)
                continue
            try:
                if pred_api_name == 'ToolSearcher':
                    pred_param_dict['keywords'] = split_by_uppercase(pred_param_dict['keywords'])
                pred_result = tool_manager.api_call(pred_api_name, **pred_param_dict)
            except TypeError as e:
                logging.warning('TypeError: {} {}'.format(str(e), pred_id))
                errored_sample_ids.append(sample_id)
                continue
            except AssertionError as e:
                logging.warning('AssertionError: {} {}'.format(str(e), pred_id))
                errored_sample_ids.append(sample_id)
                continue
            except Exception as e:
                if str(e) == 'invalid tool name.':
                    logging.warning('invalid tool name: {} {}'.format(str(e), pred_id))
                    errored_sample_ids.append(sample_id)
                    continue
                else:
                    logging.warning('I do not know what is happening here: {} {}'.format(str(e), pred_id))
                    errored_sample_ids.append(sample_id)
                    continue
                    # raise e
            
            gt_api = tool_manager.init_tool(gt_api_name)
            try:
                correct = gt_api.check_api_call_correctness(pred_result, gt_result)
            except KeyError:
                correct = False
                logging.warning('KeyError: {}'.format(pred_id))
            except AssertionError as e:
                correct = False
                logging.warning('AssertionError: {} {}'.format(str(e), pred_id))
            if correct:
                correct_num += 1
                if gt_result['api_name'] == 'ToolSearcher':
                    lv3_correctness += 1
                else: 
                    lv1_2_correctness += 1
            else:
                # for test toolsearcher
                errored_sample_ids.append(sample_id)
                if gt_api_name != 'ToolSearcher':
                    pass
                else:
                    tool_search_error_num += 1
                logging.warning('Incorrect: {}'.format(pred_id))
        else:
            gt_response = pred['answer']
            pred_response = pred['pred'].replace('User:', '').replace('AI:', '').strip()
            rouge_l_score = calculate_rouge_l_score(gt_response, pred_response)
            rougel_scores.append(rouge_l_score)

    if if_api:
        print('Accuracy: {}'.format(correct_num / total_num))
        print('Total: {}'.format(total_num))
        print('Correct: {}'.format(correct_num))

        print('Sample Accuracy: {}'.format((50 - len(set(errored_sample_ids))) / 50))
        print('Total: {}'.format(50))
        print('Correct: {}'.format(50 - len(set(errored_sample_ids))))

        print('ToolSearcher Error Num: {}'.format(tool_search_error_num))
    else:
        print('Rouge-L: {}'.format(sum(rougel_scores) / len(rougel_scores)))
    accuracy = float(correct_num / total_num)

    lv12_accuracy = float(lv1_2_correctness/lv1_2_total)
    lv3_accuracy = float(lv3_correctness/lv3_total)
    return accuracy, lv12_accuracy, lv3_accuracy

if __name__ == '__main__':
    # lv3_evaluator_new.py
    from pathlib import Path

    # 动态获取项目根目录（SPPL/）：当前文件所在目录 -> 上两级 -> 就是 SPPL
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ../../

    # 拼接配置文件路径
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.py"

    # 导入 config 模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    HOME_DIRECTORY = config.HOME_DIRECTORY  # 拿到变量

    with open(f"{HOME_DIRECTORY}/output/useless.json", 'r') as f:
        API_BANK_test_data_list = json.load(f)
    eval_api_bank(API_BANK_test_data_list, HOME_DIRECTORY)
