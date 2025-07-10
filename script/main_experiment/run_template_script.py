import subprocess
import time
import os
import json
email_address = os.environ.get("EMAIL_ADDRESS")

def generate_script(job_name, train_task_name, file_suffix, n_train, sft_epoch, sft_lr, num_of_sft_checkpoints, disable_final_eval, seed_num, variation_suffix, model_type, train_method):
    debug_mode = False
    partition = "a100"
    num_cpus = 8
    account = "strategic"
    time_alloc = "2-00:00:00"  # D-HH:MM format
    gres = "gpu:1,tmpfs:100G"
    memory = "80GB"
    email = email_address
    # Create a template for the shell script
    sh_template = \
f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH -n {num_cpus} # SBATCH -N 2
#SBATCH -A {account}
#SBATCH --time={time_alloc} # time allocation
#SBATCH --gres={gres} # generic resource required
#SBATCH --mem={memory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={email}
#SBATCH --signal B:USR2

conda init
conda activate acl_2025

# Variables
train_task_name={train_task_name}
file_suffix={file_suffix}
n_train={n_train}
sft_epoch="{sft_epoch}"
sft_lr={sft_lr}
num_of_sft_checkpoints={num_of_sft_checkpoints}
disable_final_eval={str(disable_final_eval).lower()}
debug_mode={str(debug_mode).lower()}
seed_num={seed_num}
variation_suffix={variation_suffix}
model_type={model_type}
train_method={train_method}

python /gpfs/users/a1796450/ACL_2024/Minimum_Change/main.py --file_suffix $file_suffix --train_task_name $train_task_name --n_train $n_train --n_eval 1000 --n_validation 300 --seed_num $seed_num --sft_epoch $sft_epoch --sft_lr $sft_lr --num_of_sft_checkpoints $num_of_sft_checkpoints --disable_final_eval $disable_final_eval --train_method $train_method --model_type $model_type --debug_mode $debug_mode --variation_suffix $variation_suffix"""

    # Write the script to a file
    path = os. getcwd()
    script_path = f"{path}/script/main_experiment/submit_job_template.sh"
    with open(script_path, "w") as f:
        f.write(sh_template)

    time.sleep(10)

    # Optionally, you can print out the script to double-check
    print("Generated script:")
    print(sh_template)

    # 直接调用 sbatch
    try:
        subprocess.run(["sbatch", script_path])
        print(f"Simulated sbatch submission for {script_path}")
    except FileNotFoundError:
        print('----------------------------------------------------------------')
        print("sbatch command not found. Make sure Slurm is installed and sbatch is in your PATH.")
        print('----------------------------------------------------------------')
    except Exception as e:
        print('----------------------------------------------------------------')
        print(f"An error occurred: {e}")
        print('----------------------------------------------------------------')

    time.sleep(3)



# task_name_list = ["gsm8k", 'math_algebra', 'ecqa', 'esnli', 'boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'agieval', 'mmlu_pro', 'hellaswag', 'arc_challenge', 'drop', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'math_intermediate_algebra', 'math_geometry']

# task_name_list = ["gsm8k", 'math_algebra']#, 'ecqa', 'boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'agieval', 'mmlu_pro', 'hellaswag', 'arc_challenge', 'drop', 'mbpp'] # 'esnli',    # mistral
# task_name_list = ['math_algebra', 'winogrande', 'piqa', 'boolq', "squad", 'arc_challenge', 'mmlu_pro', 'drop', 'mbpp'] # llama_3
# task_name_list = ['mmlu_pro']#, 'drop', 'squad'] # qwen

# task_name_list = ['plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality']
# task_name_list = ['plan_bench_generalization']

# task_name_list = ['plan_bench_verification', 'plan_bench_execution']#, 'plan_bench_reuse', 'plan_bench_replaning']
# task_name_list = ['plan_bench_reuse', 'plan_bench_execution']
# task_name_list = ['math_intermediate_algebra']#, 'math_geometry']

# task_name_list = ['ecqa', 'squad', 'drop', 'winogrande'] # cross domain full
# task_name_list = ['drop', 'winogrande'] # cross-domain left

# task_name_list = ['math_algebra', 'boolq']

# task_name_list = ['plan_bench_optimality']

# task_name_list = ['plan_bench_verification', 'plan_bench_reuse']

task_name_list = ['plan_bench_reuse']


# task_name_list = ['math_algebra']

# task_name_list = ['math_geometry']
# task_name_list = ['mmlu_moral_scenarios']

file_suffix = "dec_8"
model_type_list = ["mistral", 'qwen', "llama_3_instruct"]
# model_type_list = ['qwen', 'llama_3_instruct']

train_method_list = ["groundtruth", "gpt4", "mini_gpt4", "claude"]
# train_method_list = ["mini_gpt4", 'claude']
# train_method_list = ["gpt4", "mini_gpt4"]
# train_method_list = ["gpt4", 'groundtruth']
# train_method_list = ["mini_gpt4", 'gpt4']
# train_method_list = ["claude"]
# train_method_list = ["gpt4", "groundtruth", "mini_gpt4", "claude"]
# train_method_list = ["gpt4", "mini_gpt4", "claude"]
# train_method_list = ["groundtruth"]


# sft_epoch = "20"
sft_epoch = "40"

seed_num_list = ['0']
# seed_num_list = ['1']
# seed_num_list = ['1', '2']
# seed_num_list = ['2']
# seed_num_list = ['1']
# seed_num_list = ['0', '1', '2']

# sft_lr_list = ['2e-5', '2e-4']
# sft_lr_list = ['2e-5']
sft_lr_list = ['2e-4']

variation_suffix_list_non_gpt4 = ["none"]
variation_suffix_list_gpt4_gt_non_cot = ['none', "variation_gpt4_style_in_context_examples", "variation_openai_human_written_examples", 'variation_step_by_step']#, 'variation_simple_response']
variation_suffix_list_gpt4_gt_cot = ['none', "variation_gpt4_style_in_context_examples", "variation_openai_human_written_examples", 'variation_step_by_step', 'variation_rewrite_groundtruth_in_own_words']#, 'variation_simple_response']


# variation_suffix_list_gpt4_gt_cot = ['none']
# variation_suffix_list_gpt4_gt_non_cot = ['none']


# variation_suffix_list_gpt4_gt_cot = ['variation_gpt4_style_in_context_examples']
# variation_suffix_list_gpt4_gt_non_cot = ['variation_gpt4_style_in_context_examples']

# variation_suffix_list_gpt4_gt_non_cot = ["variation_gpt4_style_in_context_examples", "variation_openai_human_written_examples", 'variation_step_by_step']#, 'variation_simple_response']
# variation_suffix_list_gpt4_gt_cot = ["variation_gpt4_style_in_context_examples", "variation_openai_human_written_examples", 'variation_step_by_step', 'variation_rewrite_groundtruth_in_own_words']#, 'variation_simple_response']

# variation_suffix_list_gpt4_gt_non_cot = ["variation_openai_human_written_examples", 'variation_gpt4_style_in_context_examples']
# variation_suffix_list_gpt4_gt_cot = ["variation_openai_human_written_examples", 'variation_gpt4_style_in_context_examples']
# variation_suffix_list_gpt4_gt_cot = ['variation_mistral_self_generated', 'variation_qwen_self_generated', 'variation_llama_3_instruct_self_generated']
# variation_suffix_list_gpt4_gt_non_cot = ['variation_mistral_self_generated', 'variation_qwen_self_generated', 'variation_llama_3_instruct_self_generated']

# variation_suffix_list_gpt4_gt_cot = ['variation_mistral_self_generated', 'variation_qwen_self_generated']
# variation_suffix_list_gpt4_gt_non_cot = ['variation_mistral_self_generated', 'variation_qwen_self_generated']


# variation_suffix_list_gpt4_gt_non_cot = ['variation_api_bank_total_combine_1000']#, 'variation_api_bank_total_combine_good', 'variation_api_bank_total_combine']
# variation_suffix_list_gpt4_gt_cot = ['variation_api_bank_total_combine_1000']#,'variation_api_bank_total_combine_good', 'variation_api_bank_total_combine']

# variation_suffix_list_gpt4_gt_non_cot = ['variation_api_bank_total_combine_good_1000']#, 'variation_api_bank_total_combine_1000']#, 'variation_api_bank_total_combine_good', 'variation_api_bank_total_combine']
# variation_suffix_list_gpt4_gt_cot = ['variation_api_bank_total_combine_good_1000']#, 'variation_api_bank_total_combine_1000']#,'variation_api_bank_total_combine_good', 'variation_api_bank_total_combine']

# variation_suffix_list_gpt4_gt_non_cot = ['variation_drop_total_combine_1000']#, 'variation_drop_total_combine_good', 'variation_drop_total_combine']
# variation_suffix_list_gpt4_gt_cot = ['variation_drop_total_combine_1000']#,'variation_drop_total_combine_good', 'variation_drop_total_combine']



# variation_suffix_list_gpt4_gt_non_cot = ['mmlu_pro_total_combine_good_1000']#, 'mmlu_pro_total_combine', 'mmlu_pro_total_combine_good']
# variation_suffix_list_gpt4_gt_cot = ['mmlu_pro_total_combine_good_1000']#,'mmlu_pro_total_combine', 'mmlu_pro_total_combine_good']


# # variation_suffix_list_gpt4_gt_non_cot = ['variation_hellaswag_total_combine_good_1000', 'variation_hellaswag_total_combine_1000']
# # variation_suffix_list_gpt4_gt_non_cot = ['variation_hellaswag_total_combine_good']
# variation_suffix_list_gpt4_gt_non_cot = ['variation_hellaswag_total_combine']

# # variation_suffix_list_gpt4_gt_cot = ['variation_hellaswag_total_combine_good_1000', 'variation_hellaswag_total_combine_1000']

# # variation_suffix_list_gpt4_gt_cot = ['variation_hellaswag_total_combine_good']
# variation_suffix_list_gpt4_gt_cot = ['variation_hellaswag_total_combine']



# variation_suffix_list_gpt4_gt_non_cot = ['variation_openai_human_written_examples', 'variation_gpt4_style_in_context_examples']
# variation_suffix_list_gpt4_gt_cot = ['variation_openai_human_written_examples', 'variation_gpt4_style_in_context_examples']

# variation_suffix_list_gpt4_gt_non_cot = ['variation_gpt4_style_in_context_examples']
# variation_suffix_list_gpt4_gt_cot = ['variation_gpt4_style_in_context_examples']


# variation_suffix_list_gpt4_gt_non_cot = ['variation_api_bank_total_combine_best_worst_2000']
# variation_suffix_list_gpt4_gt_cot = ['variation_api_bank_total_combine_best_worst_2000']

# variation_suffix_list_gpt4_gt_non_cot = ['variation_api_bank_total_combine_1000', 'variation_api_bank_total_combine_good_1000']
# variation_suffix_list_gpt4_gt_cot = ['variation_api_bank_total_combine_1000', 'variation_api_bank_total_combine_good_1000']


# variation_suffix_list_gpt4_gt_non_cot = ['variation_api_bank_total_combine_good']
# variation_suffix_list_gpt4_gt_cot = ['variation_api_bank_total_combine_good']

# variation_suffix_list_gpt4_gt_non_cot = ['variation_api_bank_total_combine']
# variation_suffix_list_gpt4_gt_cot = ['variation_api_bank_total_combine']


# variation_suffix_list_gpt4_gt_non_cot = ['none', 'variation_step_by_step']
# variation_suffix_list_gpt4_gt_cot = ['none', 'variation_step_by_step']


disable_final_eval = False

variation_suffix_list_only_gold_label = ["variation_gold_label"]
variation_suffix_list_only_groundtruth = ["none"]
variation_suffix_list_both_gold_label_groundtruth = ["none", "variation_gold_label"]

non_cot_name_list = ['boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'mmlu_pro', 'mmlu_pro_law',  'mmlu_moral_scenarios', 'agieval', 'api_bank', 'mmlu_pro', 'hellaswag', 'theoremqa', 'arc_challenge', 'drop']
cot_name_list = ['mbpp', "gsm8k", 'math_algebra', 'ecqa', 'esnli', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_reuse', 'plan_bench_replaning', 'math_intermediate_algebra', 'math_geometry']

only_gold_label_name_list = ['boolq', "squad", 'winogrande', 'piqa', 'mmlu', 'mmlu_pro', 'mmlu_pro_law', 'mmlu_moral_scenarios', 'agieval', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_reuse', 'plan_bench_replaning', 'mmlu_pro', 'hellaswag', 'theoremqa', 'arc_challenge', 'drop']
only_groundtruth_name_list = ['mbpp', "gsm8k", 'math_algebra', 'math_intermediate_algebra', 'math_geometry']
both_gold_label_groundtruth_name_list = ['ecqa', 'esnli']

# for n_train in ['1000']:
# for n_train in ['2000']:
# for n_train in ['1000', '100']:
# for n_train in ['3000']:
# for n_train in ['6000']:
for n_train in ['100']:
    for task_name in task_name_list:
        if 'api_bank' == task_name:
            num_of_sft_checkpoints = '15'
        else:
            num_of_sft_checkpoints = '20'
        for train_method in train_method_list:
            if train_method == "gpt4":
                if task_name in cot_name_list:
                    variation_suffix_list = variation_suffix_list_gpt4_gt_cot
                if task_name in non_cot_name_list:
                    variation_suffix_list = variation_suffix_list_gpt4_gt_non_cot
            elif train_method == "groundtruth":
                if task_name in only_gold_label_name_list:
                    variation_suffix_list = variation_suffix_list_only_gold_label
                if task_name in only_groundtruth_name_list:
                    variation_suffix_list = variation_suffix_list_only_groundtruth
                if task_name in both_gold_label_groundtruth_name_list:
                    variation_suffix_list = variation_suffix_list_both_gold_label_groundtruth
            else:
                variation_suffix_list = variation_suffix_list_non_gpt4
            for seed_num in seed_num_list:
                for sft_lr in sft_lr_list:
                    for variation_suffix in variation_suffix_list:
                        for model_type in model_type_list:
                            job_name = task_name    # e.g. "my_experiment"
                            train_task_name = task_name
                            # variation_suffix = 'cross_domain_' + variation_suffix
                            if 'total' in variation_suffix:
                                num_of_sft_checkpoints = '15'
                            if n_train == '6000' or n_train == '3000':
                                num_of_sft_checkpoints = '10'
                            # if task_name == 'mmlu_pro_law' and 'gpt4_style_in_context_examples' in variation_suffix:
                            #     num_of_sft_checkpoints = '15'
                            if task_name == 'plan_bench_verification':
                                num_of_sft_checkpoints = '15'
                            generate_script(job_name, train_task_name, file_suffix, n_train, sft_epoch, sft_lr, num_of_sft_checkpoints, disable_final_eval, seed_num, variation_suffix, model_type, train_method) 
