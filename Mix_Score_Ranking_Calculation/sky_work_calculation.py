from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import sys
import os
import os
import pickle

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import HOME_DIRECTORY
from utils.in_context_data_loader import perplexity_calculation_in_context_data_loader


import os, sys, json, pickle, time
from transformers import AutoTokenizer, AutoModelForSequenceClassification


os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_reuse', 'plan_bench_execution', 'plan_bench_verification', 'plan_bench_replaning']

n_train = 300

# ----------- 载入一次奖励模型，循环里重复用 -----------
rm_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModelForSequenceClassification.from_pretrained(
    rm_name, torch_dtype=torch.bfloat16, device_map=device, num_labels=1
)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)

def skywork_reward_calculation(question, response):
    conv = [{"role": "user", "content": question},
            {"role": "assistant", "content": response}]
    tokens = rm_tokenizer.apply_chat_template(conv, tokenize=True,
                                              return_tensors="pt").to(device)
    with torch.no_grad():
        score = rm(tokens).logits[0, 0].item()
    return score

# ------------ 主循环 ------------
for train_task_name in train_task_list:

    dataset_list, _, _, test_task_name, _ = perplexity_calculation_in_context_data_loader(
        train_task_name, n_train, False, -1, ''
    )

    temp        = {}   # 存数据集 → 分数列表
    time_cost   = {}   # 存数据集 → 处理秒数

    for data_name, data_list, *_ in dataset_list:
        print(f'---------------- {train_task_name}: {data_name} ----------------')

        # ==== 计时开始 ====
        torch.cuda.synchronize() if device.startswith("cuda") else None  # 等待异步 GPU 结束
        t0 = time.perf_counter()

        scores = []              # ← 每轮单独的列表
        for item in data_list:
            q, r = item['question'], item['answer']
            scores.append(skywork_reward_calculation(q, r))

        temp[data_name]  = scores

        torch.cuda.synchronize() if device.startswith("cuda") else None
        elapsed = time.perf_counter() - t0
        time_cost[data_name] = elapsed
        print(f"{data_name} done in {elapsed:.2f}s, {len(scores)} items")

    # ===== 保存结果 =====
    save_dir   = f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/skywork_reward_record"
    os.makedirs(save_dir, exist_ok=True)
    file_path  = f"{save_dir}/{train_task_name}_{n_train}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(temp, f)

    print(f"Saved to {file_path}\n")
