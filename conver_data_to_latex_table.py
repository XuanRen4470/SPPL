import os
from config.config import HOME_DIRECTORY
from utils.function import load_experimental_result

# model_name_list = ['mistral', 'llama_3_instruct']
model_name_list = ['mistral', 'llama_3_instruct', 'qwen']
# task_name_list = ['gsm8k', 'math_algebra', 'ecqa', 'boolq', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'mmlu_pro', 'arc_challenge', 'drop', 'mbpp', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'math_intermediate_algebra', 'math_geometry', 'esnli', 'hellaswag', 'mmlu_pro_law', 'mmlu_moral_scenarios']
# task_name_list = ['gsm8k', 'math_algebra', 'ecqa', 'boolq', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'mmlu_pro', 'arc_challenge', 'drop', 'mbpp', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'math_intermediate_algebra', 'math_geometry', 'esnli', 'hellaswag', 'mmlu_pro_law', 'mmlu_moral_scenarios']

task_name_list = ['gsm8k', 'math_algebra', 'math_geometry', 'ecqa', 'boolq', 'winogrande', 'piqa', 'agieval', 'squad', 'arc_challenge', 'drop', 'mbpp', 'api_bank', 'hellaswag', 'mmlu_pro_law', 'mmlu_moral_scenarios']


# task_name_list = ['gsm8k', 'math_algebra', 'ecqa', 'boolq', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'mmlu_pro', 'arc_challenge', 'drop', 'mbpp', 'api_bank']#, 'plan_bench_generalization']# 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization']

# task_name_list = ['api_bank', 'drop', 'mmlu_pro', 'squad']


# task_name_list = ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse', 'plan_bench_verification']

task_name_list = ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_replaning', 'plan_bench_reuse', 'plan_bench_verification']



# train_task_list = ['plan_bench_generalization', 'plan_bench_execution']


# seed_num = 0
# seed_num = 1
# seed_num = 2
# seed_num = 'average_of_seed_0,1,2'
seed_num = 'average_of_seed_0,1'


sft_lr = '2e-05'#, '2e-04'
# sft_lr = '0.0002'

n_train = 1000
# n_train = 100

# epoch_num = 20
epoch_num = 40

output_file = f"{HOME_DIRECTORY}/log_total/experiment_data_recorder/latex_table/results_table_{n_train}_{sft_lr}_{epoch_num}_{seed_num}.tex"
if seed_num == 'average_of_seed_0,1,2':
    experiment_result_dict_0 = load_experimental_result(model_name_list, task_name_list, n_train, sft_lr, 0, epoch_num)
    experiment_result_dict_1 = load_experimental_result(model_name_list, task_name_list, n_train, sft_lr, 1, epoch_num)
    experiment_result_dict_2 = load_experimental_result(model_name_list, task_name_list, n_train, sft_lr, 1, epoch_num)

    experiment_result_dict = {}
    for model_name_item in experiment_result_dict_0:
        experiment_result_dict[model_name_item] = {}
        for task_name_item in experiment_result_dict_0[model_name_item]:
            experiment_result_dict[model_name_item][task_name_item] = {}
            for data_generation_method_item in experiment_result_dict_0[model_name_item][task_name_item]:
                experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] = 0
                count = 0
                if experiment_result_dict_0[model_name_item][task_name_item][data_generation_method_item]:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] += float(experiment_result_dict_0[model_name_item][task_name_item][data_generation_method_item])
                    count += 1
                if experiment_result_dict_1[model_name_item][task_name_item][data_generation_method_item]:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] += float(experiment_result_dict_1[model_name_item][task_name_item][data_generation_method_item])
                    count += 1
                if experiment_result_dict_2[model_name_item][task_name_item][data_generation_method_item]:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] += float(experiment_result_dict_2[model_name_item][task_name_item][data_generation_method_item])
                    count += 1
                
                if count > 0:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] = round(
    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] / count, 3
)
                else:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] = None
elif seed_num == 'average_of_seed_0,1':
    experiment_result_dict_0 = load_experimental_result(model_name_list, task_name_list, n_train, sft_lr, 0, epoch_num)
    experiment_result_dict_1 = load_experimental_result(model_name_list, task_name_list, n_train, sft_lr, 1, epoch_num)

    experiment_result_dict = {}
    for model_name_item in experiment_result_dict_0:
        experiment_result_dict[model_name_item] = {}
        for task_name_item in experiment_result_dict_0[model_name_item]:
            experiment_result_dict[model_name_item][task_name_item] = {}
            for data_generation_method_item in experiment_result_dict_0[model_name_item][task_name_item]:
                experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] = 0
                count = 0
                if experiment_result_dict_0[model_name_item][task_name_item][data_generation_method_item]:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] += float(experiment_result_dict_0[model_name_item][task_name_item][data_generation_method_item])
                    count += 1
                if experiment_result_dict_1[model_name_item][task_name_item][data_generation_method_item]:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] += float(experiment_result_dict_1[model_name_item][task_name_item][data_generation_method_item])
                    count += 1
                
                if count > 0:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] = round(
    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] / count, 3
)
                else:
                    experiment_result_dict[model_name_item][task_name_item][data_generation_method_item] = None
else:
    experiment_result_dict = load_experimental_result(model_name_list, task_name_list, n_train, sft_lr, seed_num, epoch_num)
    



a = 1
# task_name_list += task_name_list_hr
# experiment_result_dict_lr = load_experimental_result(model_name_list, ['gsm8k', 'math_algebra', 'ecqa', 'boolq', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'mmlu_pro', 'arc_challenge', 'drop', 'mbpp'], n_train, '2e-05', seed_num, 20)
# experiment_result_dict_hr = load_experimental_result(model_name_list, ['api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization'], n_train, '0.0002', seed_num, 40)
# output_file = f"{HOME_DIRECTORY}/log_total/experiment_data_recorder/latex_table/results_table_{n_train}_2e-05_0.0002_{epoch_num}_{seed_num}.tex"
# experiment_result_dict = {}
# for model_name in model_name_list:
#     result = experiment_result_dict_lr[model_name] | experiment_result_dict_hr[model_name]
#     experiment_result_dict[model_name] = result
#     a = 1


plan_bench_check = True
for item in task_name_list:
    if 'plan_bench' not in item:
        plan_bench_check = False

if plan_bench_check:
    output_file = output_file.replace('.tex', '_plan_bench.tex')

# Ensure the directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f:
    f.write("\\begin{table*}[t!]\n")
    f.write("\\centering\n")
    f.write("\\resizebox{1.0\\textwidth}{!}{\n")
    line_temp = "\\begin{tabular}{l|l|" + "|".join(["c"]*len(task_name_list)) + "}\n"
    line_temp = line_temp.replace('_', ' ')
    f.write(line_temp)
    f.write("\\hline\n")

    # 表头
    line_temp = "Data Generation Strategy & Model Type & " + " & ".join(task_name_list) + " \\\\ \\hline\n"
    line_temp = line_temp.replace('_', ' ')
    f.write(line_temp)

for cc, model_name in enumerate(model_name_list):
    # Sort the keys of the dictionary by their length
    # sorted_keys = sorted(experiment_result_dict[model_name]['ecqa'].keys(), key=len)

    keys = experiment_result_dict[model_name][task_name_list[0]].keys()
    # keys = experiment_result_dict[model_name]['ecqa'].keys()
    priority_keys = ['gold_label', 'groundtruth']
    other_keys = [key for key in keys if key not in priority_keys]

    # Sort other keys by length
    sorted_other_keys = sorted(other_keys, key=len)

    # Combine priority keys and sorted other keys
    sorted_keys = priority_keys + sorted_other_keys

    for i, method in enumerate(sorted_keys):
        method_temp = method.replace('_', ' ')
        model_name_temp = model_name.replace('_', ' ')
        if i == 0:
            line = f"{method_temp} & {model_name_temp}"
        else:
            line = f"{method_temp} & "
        for task_name in experiment_result_dict[model_name].keys():
            try:
                accuracy = experiment_result_dict[model_name][task_name][method]
                line += f' & {accuracy}'
            except:
                line += f' & '
        with open(output_file, "a") as f:
            f.write(f"{line}" + "\\\\\n")
        a = 1
    if cc == len(model_name_list) - 1:
        continue
    else:
        with open(output_file, "a") as f:
            f.write(f"\hline\hline\n")
    

with open(output_file, "a") as f:
    f.write("\\hline\n\\end{tabular}}\n")
    label_content = f'tab:ntrain_{n_train}_lr_{sft_lr}_seed{seed_num}'
    if plan_bench_check:
        label_content += '_pan_bench'
    line = "\\caption{seed " + str(seed_num) + ' train datasize ' + str(n_train) + ' lr ' + str(sft_lr) + ' epoch num ' + str(epoch_num) + '}\n'
    # line = "\\caption{seed " + str(seed_num) + ' train datasize ' + str(n_train) + ' lr ' + str(2e-05) + ' hr ' + str(2e-04) + ' epoch num ' + str(20) + ' epoch num ' + str(40) + '}\n'
    f.write(line)
    f.write("\\label{"  + label_content + "}\n")
    f.write("\\end{table*}\n")
