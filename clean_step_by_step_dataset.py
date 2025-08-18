import json
from config.config import *



train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank' , 'plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality', 'plan_bench_reuse', 'plan_bench_verification', 'plan_bench_replaning', 'plan_bench_execution']


for name in train_task_list:
    name = name.upper()
    with open(f'/gpfs/users/a1796450/ACL_2024/SPPL/dataset/{name}/varient/gpt4_generated_step_by_step_1000.json', 'r') as f:
        data_list_temp = json.load(f)

    # 要去除的前缀列表
    prefixes_to_remove = [
        '### Step by Step Inference:\n\n',
        'Step by Step INFERENCE HERE\n\n',
        '### Step by Step INFERENCE\n\n',
        'Step by Step INFERENCE\n\n'
        'Step by Step INFERENCE HERE:\n\n'
    ]

    # 处理数据
    data_list = []
    for item in data_list_temp:
        answer = item['answer']
        for prefix in prefixes_to_remove:
            answer = answer.replace(prefix, '')
        item['answer'] = answer
        data_list.append(item)

    # 保存处理后的 JSON 文件
    with open(f'/gpfs/users/a1796450/ACL_2024/SPPL/dataset/{name}/varient/gpt4_generated_step_by_step_1000.json', 'w') as f:
        json.dump(data_list, f, indent=2)
