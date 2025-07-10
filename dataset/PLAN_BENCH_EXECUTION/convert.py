import json

file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_EXECUTION/task_7_plan_execution.json"
file_path_another = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_EXECUTION/task_7_plan_execution_another.json'
with open(file_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)

with open(file_path_another, "r", encoding="utf-8") as f:
    data_list_another = json.load(f)


n_data = 3000

# 定义块到颜色的映射，支持任意新加块
colors = {
    'a': 'red',
    'b': 'blue',
    'c': 'orange',
    'd': 'yellow',  # 示例：新加黄色块
    'e': 'red',     # 示例：e 也映射为红色块
}

def translate_predicate(pred, colors):
    """
    将 ground_truth_plan 中的符号 predicate 转换为自然语言描述。
    支持 clear, holding, ontable, on 以及 handempty。
    """
    if pred == 'handempty':
        return 'the hand is empty'

    parts = pred.split('_')
    action = parts[0]

    if action == 'clear' and len(parts) == 2:
        blk = parts[1]
        return f'the {colors.get(blk, blk)} block is clear'

    if action == 'holding' and len(parts) == 2:
        blk = parts[1]
        return f'the hand is currently holding {colors.get(blk, blk)} block'

    if action == 'ontable' and len(parts) == 2:
        blk = parts[1]
        return f'the {colors.get(blk, blk)} block is on the table'

    if action == 'on' and len(parts) == 3:
        top, bottom = parts[1], parts[2]
        return f'the {colors.get(top, top)} block is on top of the {colors.get(bottom, bottom)} block'

    # 未匹配则原样返回
    return pred

# 遍历两个域，分别处理数据
data_list_temp = []
for domain_name in ['blocksworld', 'blocksworld_3']:
    # 根据域名选择不同的数据源
    if domain_name == 'blocksworld':
        source = data_list
    else:
        source = data_list_another

    instances = source['instances']
    domain = source['domain']

    for item in instances:
        # 翻译每个 predicate
        gt_preds = [translate_predicate(p, colors) for p in item['ground_truth_plan']]
        # 拼接为字符串，中间用逗号+空格分隔
        temp = ', '.join(gt_preds)

        # 更新字段
        item['task'] = 'task_7_plan_execution'
        item['prompt_type'] = 'oneshot'
        item['domain'] = domain
        item['question'] = item.pop('query')  # 重命名字段
        item['answer'] = 'Final Answer: ' + temp
        item['gold_label'] = temp

        data_list_temp.append(item)

    # 如果需要，可以将处理结果写回：
    # source['instances'] = data_list_temp

    # 这里示例仅打印前3个
    print(f"Domain {domain_name}: processed {len(data_list_temp)} items, sample:")
    for sample in data_list_temp[:3]:
        print(sample)


with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_EXECUTION/groundtruth.json", "w") as json_file:
    json.dump(data_list_temp[:400], json_file, indent=4)
with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_EXECUTION/validation.json", "w") as json_file:
    json.dump(data_list_temp[400:], json_file, indent=4)
with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_EXECUTION/test.json", "w") as json_file:
    json.dump(data_list_temp[400:], json_file, indent=4)

