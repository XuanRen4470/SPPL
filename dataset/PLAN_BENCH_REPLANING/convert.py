import json

file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_REPLANING/task_6_replanning.json"
file_path_another = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_REPLANING/task_6_replanning_another.json'
with open(file_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)

with open(file_path_another, "r", encoding="utf-8") as f:
    data_list_another = json.load(f)


n_data = 3000


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

        # 更新字段
        item['task'] = 'task_6_replanning'
        item['prompt_type'] = 'oneshot'
        item['domain'] = domain
        item['question'] = item.pop('query')  # 重命名字段
        item['answer'] = 'Final Answer: ' + item['ground_truth_plan']
        item['gold_label'] = item['ground_truth_plan']

        data_list_temp.append(item)

    # 如果需要，可以将处理结果写回：
    # source['instances'] = data_list_temp

    # 这里示例仅打印前3个
    print(f"Domain {domain_name}: processed {len(data_list_temp)} items, sample:")
    for sample in data_list_temp[:3]:
        print(sample)


with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_REPLANING/groundtruth.json", "w") as json_file:
    json.dump(data_list_temp[:400], json_file, indent=4)
with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_REPLANING/validation.json", "w") as json_file:
    json.dump(data_list_temp[400:], json_file, indent=4)
with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_REPLANING/test.json", "w") as json_file:
    json.dump(data_list_temp[400:], json_file, indent=4)

