import sys
import os
import json

# 将上一级目录加入 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from utils.data_loader import add_gold_label, load_gold_label_and_question_list
from utils.function import load_experimental_result
from evaluation.eval import Check_Correctness
from config.config import HOME_DIRECTORY

# train_task_list = [
#     'gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval',
#     'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop',
#     'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank',
#     'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization',
#     'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification',
#     'plan_bench_replaning'
# ]



# we skip mbpp because it is hard to evaluate in zeroshot senarios. llm does not tend to directly output code without any explaination. therefore, the zeroshot prediction is likely to fail.
# we also skil planbench since it is hard to evaluate in zeroshot senarios.
train_task_list = [
    'gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval',
    'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop',
    'hellaswag', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank'
]

model_name_list = ['qwen', 'mistral', 'llama_3_instruct']


# log 文件路径
log_dir = f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/experiment_result/zero_shot_evaluation"
os.makedirs(log_dir, exist_ok=True)
txt_log_path = os.path.join(log_dir, "zero_shot_evaluation_log.txt")
json_log_path = os.path.join(log_dir, "zero_shot_evaluation_log.json")





def write_zero_shot_to_table(results_dict, win_ratio_dict, table_tex_name):
    output_file = os.path.join(
        HOME_DIRECTORY, "Mix_Score_Ranking_Calculation/experiment_result", f"{table_tex_name}.tex"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_name_list = list(results_dict.keys())
    task_list = list(next(iter(results_dict.values())).keys())
    n_tasks = len(task_list)

    with open(output_file, "w") as f:
        f.write("\\begin{table*}[t!]\n")
        f.write("  \\centering\n")
        f.write("  \\resizebox{1.0\\textwidth}{!}{%\n")

        # 每个任务占一列
        col_format = "l|" + "|".join(["c"] * len(task_list))
        f.write("  \\begin{{tabular}}{{{}}}\n".format(col_format))
        f.write("    \\hline\n")

        # 表头
        header = "Model " + " & " + " & ".join([task.replace("_", " ") for task in task_list])
        f.write("    {} \\\\ \\hline\n".format(header))

        # 每个模型一行
        for model_name in model_name_list:
            line = "    {}".format(model_name.replace("_", " "))
            for task in task_list:
                zero_acc = results_dict[model_name][task]["zero shot accuracy"]
                gpt4_acc = results_dict[model_name][task]["gpt4 accuracy"]

                zero_str = "{:.1f}\\%".format(zero_acc * 100)
                gpt4_str = "{:.1f}\\%".format(gpt4_acc * 100)

                if gpt4_acc > zero_acc:
                    gpt4_str = "\\textbf{" + gpt4_str + "}"

                cell = "{} / {}".format(zero_str, gpt4_str)
                line += " & {}".format(cell)

            line += " \\\\\n"
            f.write(line)

        f.write("    \\hline\n")
        f.write("  \\end{tabular}}\n")

        # caption 里写清楚说明 + 胜率统计
        ratio_parts = []
        for m, r in win_ratio_dict.items():
            win_count = int(round(r * n_tasks))
            ratio_parts.append("{}: {}/{} ({:.1f}\\%)".format(m.replace("_", " "), win_count, n_tasks, r*100))
        ratio_str = "; ".join(ratio_parts)

        f.write("  \\caption{Zero-shot (left) vs. trained on GPT-4o direct-answer data (right) accuracy across tasks for each model. "
                "Win ratios of GPT-4o direct-answer training over zero-shot: " + ratio_str + ".}\n")
        f.write("\\label{{tab:{}}}\n".format(table_tex_name))
        f.write("\\end{table*}\n")

# 存放所有任务结果的 dict
results_dict = {}
win_ratio_dict = {}
with open(txt_log_path, "a") as logfile:   # 追加模式
    for model_name in model_name_list:
        win_ratio_dict[model_name] = 0
        win_count = 0
        results_dict[model_name] = {}
        for task_name in train_task_list:
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/gpt4.json'
            predict_path = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/zero_shot_prediction/zero_shot_{task_name}_{model_name}_initial_prediction_1000.json'

            with open(data_path, 'r') as f:
                data_list = json.load(f)
            with open(predict_path, 'r') as f:
                prediction_list = json.load(f)

            total_correct = 0
            n_data_creation = 1000
            data_list = data_list[:n_data_creation]

            gold_label_list, groundtruth_list, question_list = load_gold_label_and_question_list(task_name, n_data_creation)

            for index, item in enumerate(data_list):
                pred_temp = []
                data_temp = []
                pred_temp.append(prediction_list['initial_prediction'][index])
                item = add_gold_label(task_name, item, gold_label_list[index])

                data_temp.append(item)
                accuracy, cover_ratio = Check_Correctness(
                    pred_temp,
                    data_temp,
                    task_name,
                    f'{HOME_DIRECTORY}/evaluation/intermediate_data',
                    task_name='error_correction',
                    extract_gold_label_as_gt=True,
                    simple_evaluation=True
                )

                if accuracy == 1:
                    total_correct += 1
            



            default_lr_experiment_result_dict = load_experimental_result([model_name], [task_name], 1000, '2e-05', 0, 20, load_none_as = None)
            gpt4_acc = default_lr_experiment_result_dict[model_name][task_name]['gpt4']
            gpt4_acc = float(gpt4_acc)
            
            final_acc = total_correct / len(data_list)
            final_acc = float(final_acc)
            print(f"Model Name={model_name}      {task_name} Zero Shot accuracy: {final_acc:.4f}, GPT4 Accuracy={gpt4_acc:.4f}")

            # 写 txt 日志
            logfile.write(
                f"Model Name={model_name}, Task={task_name}, Zero Shot Accuracy={final_acc:.4f}, GPT4 Accuracy={gpt4_acc:.4f}, N={len(data_list)}\n"
            )

            improvement_ratio = (gpt4_acc - final_acc) / (final_acc + 0.000000001)
            # 存结果到 dict
            results_dict[model_name][task_name] = {
                "zero shot accuracy": round(final_acc, 4),
                "gpt4 accuracy": round(gpt4_acc, 4),
                "improvement_ratio": round(improvement_ratio, 4),
            }

            if improvement_ratio > 0:
                win_count += 1
        win_ratio = win_count/len(train_task_list)
        # model_name = model_name.replace('_', '\_')
        win_ratio_dict[model_name] = win_ratio

    # 把 dict 存成 json（覆盖写）
    with open(json_log_path, "w") as jf:
        json.dump(results_dict, jf, indent=2)
    

    print(win_ratio_dict)


    table_tex_name = 'win_ratio_record'
    write_zero_shot_to_table(results_dict, win_ratio_dict, table_tex_name)
