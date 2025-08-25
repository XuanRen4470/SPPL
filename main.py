import os
import torch
import shutil
from utils.__init__ import *
from config.config import *
from config.modify_config_on_current_job import set_config
import argparse
import gc
import time
from evaluation.eval import EVALUATION_LLAMA_FACTORY
from utils.log_writter import *
from utils.train import train_llama_factory

print(torch.__version__)

parser = argparse.ArgumentParser(description='train and evaluate')

# Add arguments
parser.add_argument('--file_suffix', type=str, required=True, help='Training method')
parser.add_argument('--train_task_name', type=str, required=True, help='Training task name')
parser.add_argument('--n_train', type=int, required=True, help='Number of training examples')
parser.add_argument('--n_eval', type=int, required=True, help='Number of evaluation examples')
parser.add_argument('--n_validation', type=int, required=True, help='Number of validation examples')
parser.add_argument('--seed_num', type=int, required=True, help='Seed number')
parser.add_argument('--train_method', type=str, required=False, default = '', help='')
parser.add_argument('--sft_epoch', nargs='+', type=int, default=[10], help='')
parser.add_argument('--sft_lr', type=float, default=5e-5, help='')
parser.add_argument('--num_of_sft_checkpoints', type=int, default=50, help='')
parser.add_argument('--disable_final_eval', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--variation_suffix', type=str, required=False, default = '', help='')
parser.add_argument('--lora_rank', type=int, required=False, default = 8, help='')
parser.add_argument('--model_type', type=str, required=False, default='llama2-13b')
parser.add_argument('--debug_mode', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--merged_base_model_dir', type=str, required=False, default='', help='Training method')
parser.add_argument('--load_in_8bit', type=lambda x: (str(x).lower() == 'true'), default=False, help='')



# Parse arguments
args = parser.parse_args()

file_suffix = args.file_suffix
train_task_name = args.train_task_name
n_train = args.n_train
n_eval = args.n_eval
n_validation = args.n_validation
seed_num = args.seed_num
sft_epoch_list = args.sft_epoch
sft_lr = args.sft_lr
num_of_sft_checkpoints = args.num_of_sft_checkpoints
disable_final_eval = args.disable_final_eval
variation_suffix = args.variation_suffix
load_in_8bit = args.load_in_8bit

train_method = args.train_method
lora_rank = args.lora_rank
model_type = args.model_type
debug_mode = args.debug_mode
merged_base_model_dir = args.merged_base_model_dir

model_type_name = 'llama'
if 'mistral' in model_type:
    model_type = '_mistral'
    model_type_name = 'mistral'
elif 'llama_3_instruct' in model_type:
    model_type = '_llama_3_instruct'
    model_type_name = 'llama_3_instruct'
elif 'llama_3' in model_type:
    model_type = '_llama_3'
    model_type_name = 'llama_3'
elif 'phi_4' in model_type or 'phi-4' in model_type:
    model_type = '_phi_4'
    model_type_name = 'phi_4'
    load_in_8bit = True
elif 'qwen' in model_type:
    model_type = '_qwen'
    model_type_name = 'qwen'
else:
    model_type = ''
    model_type_name = 'llama'


if 'none' == variation_suffix:
    variation_suffix = ''
if merged_base_model_dir != '':
    enable_merged_base_model_dir = '_merge'
else:
    enable_merged_base_model_dir = ''


output_folder_name = f'{train_task_name}_{train_method}_{model_type_name}{enable_merged_base_model_dir}_{variation_suffix}_{lora_rank}_{seed_num}_{file_suffix}_{n_train}_{n_validation}_{sft_epoch_list[0]}_{sft_lr}_{num_of_sft_checkpoints}'

file_name = f'{train_task_name}_{model_type_name}_{train_method}_{enable_merged_base_model_dir}_{variation_suffix}_{lora_rank}_{file_suffix}_{seed_num}_{n_train}_{n_validation}_{sft_epoch_list[0]}_{sft_lr}_{num_of_sft_checkpoints}_log'

# Construct the new directory name
if debug_mode:
    LLAMA_FACTORY_DIRECTORY_new = f"{LLAMA_FACTORY_DIRECTORY}-debug"
else:
    LLAMA_FACTORY_DIRECTORY_new = f"/gpfs/users/a1796450/llama_factory_temp/delete_later/{model_type_name}_{train_method}_{variation_suffix}_{train_task_name}_{file_suffix}_{seed_num}_{n_train}_{sft_epoch_list[0]}_{sft_lr}"

    # Check if the destination directory exists, and if so, remove it
    if os.path.exists(LLAMA_FACTORY_DIRECTORY_new):
        shutil.rmtree(LLAMA_FACTORY_DIRECTORY_new)
        print(f"Existing directory {LLAMA_FACTORY_DIRECTORY_new} removed")
        time.sleep(10)
    # Copy the directory
    try:
        shutil.copytree(LLAMA_FACTORY_DIRECTORY, LLAMA_FACTORY_DIRECTORY_new)
    except:
        time.sleep(10)
        shutil.copytree(LLAMA_FACTORY_DIRECTORY, LLAMA_FACTORY_DIRECTORY_new)
    print(f"Directory copied successfully to {LLAMA_FACTORY_DIRECTORY_new}")
    time.sleep(5)

test_task_name_list = [train_task_name.lower()]
if 'cross_domain' in variation_suffix:
    if train_task_name.upper() == 'BOOLQ':
        test_task_name_list = ['boolq', 'gsm8k', 'ecqa', 'squad', 'math_algebra', 'boolq']
    if train_task_name.upper() == 'GSM8K':
        test_task_name_list = ['gsm8k', 'math_algebra', 'ecqa', 'esnli', 'boolq']#, 'winogrande']#, 'agieval']
    elif train_task_name.upper() == 'MATH_ALGEBRA':
        test_task_name_list = ['math_algebra', 'gsm8k', 'ecqa', 'esnli', 'boolq']#, 'winogrande', 'agieval']
    elif train_task_name.upper() == 'MATH_GEOMETRY':
        test_task_name_list = ['math_geometry', 'gsm8k', 'math_algebra', 'ecqa', 'esnli']
    elif train_task_name.upper() == 'MATH_COUNTING_AND_PROBABILITY':
        test_task_name_list = ['boolq', 'piqa', 'mmlu', 'agieval', 'math_algebra', 'gsm8k', 'ecqa']
    elif train_task_name.upper() == 'MATH_NUMBER_THEORY':
        test_task_name_list = ['math_algebra', 'math_counting_and_probability', 'gsm8k', 'ecqa']#, 'mbpp', 'code', 'anli', 'snli', 'scitail']
    elif train_task_name.upper() == 'ESNLI':
        test_task_name_list = ['esnli', 'gsm8k', 'math_algebra', 'ecqa', 'boolq']
    elif train_task_name.upper() == 'PIQA':
        test_task_name_list = ['piqa', 'gsm8k', 'math_algebra', 'ecqa', 'esnli', 'boolq', 'winogrande'] 
    elif train_task_name.upper() == 'MMLU':
        test_task_name_list = ['mmlu', 'gsm8k', 'math_algebra', 'ecqa', 'esnli']#, 'math_counting_and_probability',  'mbpp', 'code', 'snli', 'anli', 'scitail']
    elif train_task_name.upper() == 'AGIEVAL':
        test_task_name_list = ['agieval', 'gsm8k', 'math_algebra', 'ecqa', 'esnli', 'boolq', 'winogrande']
    elif train_task_name.upper() == 'AQUARAT':
        test_task_name_list = ['aquarat', 'gsm8k', 'math_algebra', 'ecqa', 'esnli', 'boolq', 'winogrande']
    elif 'PLAN_BENCH' in train_task_name.upper():
        test_task_name_list = [train_task_name.lower(), 'gsm8k', 'ecqa', 'esnli', 'boolq', 'math_algebra']#, 'winogrande']
    elif train_task_name.upper() == 'API_BANK':
        test_task_name_list = ['api_bank', 'ecqa', 'gsm8k', 'math_algebra', 'squad', 'drop', 'winogrande']#, 'winogrande']
    elif train_task_name.upper() == 'ECQA':
        test_task_name_list = ['ecqa', 'gsm8k', 'math_algebra', 'squad', 'drop', 'winogrande']#, 'winogrande', 'agieval']
    elif train_task_name.upper() == 'SQUAD':
        test_task_name_list = ['squad', 'ecqa', 'gsm8k', 'math_algebra', 'drop', 'winogrande']
    elif train_task_name.upper() == 'DROP':
        test_task_name_list = ['drop', 'ecqa', 'gsm8k', 'math_algebra', 'squad', 'winogrande']    
    elif train_task_name.upper() == 'WINOGRANDE':
        test_task_name_list = ['winogrande', 'drop', 'ecqa', 'gsm8k', 'math_algebra', 'squad']

print('------------------------------------------------')

print('file_name', file_name)

print('variation_suffix', variation_suffix)

print('file_suffix', file_suffix)

print('train_task_name', train_task_name)

print('n_train', n_train)

print('seed_num', seed_num)

print('sft_lr', sft_lr)

print('train_method', train_method)

print('model_type', model_type)

print('------------------------------------------------')


initial_output_folder(output_folder_name, seed_num)

with open(f"{HOME_DIRECTORY}/log/{output_folder_name}/log.txt", 'w') as f:
    pass
with open(f"{HOME_DIRECTORY}/log/{output_folder_name}/{file_name}.txt", 'w') as f:
    pass


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

data_container_path = f"{LLAMA_FACTORY_DIRECTORY_new}/data"
intermediate_sft_file_path = set_up_training_dataset(train_method, HOME_DIRECTORY, train_task_name, n_train, variation_suffix, data_container_path, model_name = model_type)


zeroshot = False
Best_lora_dir = ''

current_stage = 'SFT'
epoch_list = sft_epoch_list
max_accuracy = 0
best_train_num = 0
max_learning_rate = 0
test_task_name = train_task_name
task_name = 'validation'
write_log(file_name, output_folder_name, f"""--------------------------{current_stage} Stage: {train_method} {task_name}--------------------------""")

for epoch_num in epoch_list:
    enable_full_set = False
    # calc here
    log_file_item_path = f"{HOME_DIRECTORY}/log/{output_folder_name}/{train_method}_{current_stage}_Stage_{file_name}.txt"
    if os.path.exists(log_file_item_path):
        with open(log_file_item_path, 'w') as f:
            pass
    if num_of_sft_checkpoints == 0:
        enable_full_set = True
    learning_rate = sft_lr
    save_chekpoints_num = num_of_sft_checkpoints
    intermediate_train_file_path = intermediate_sft_file_path

    with open(intermediate_train_file_path, 'r') as file:
        full_data_set = json.load(file)
    full_data_set_length = len(full_data_set)            
    
    current_task_name = test_task_name.lower()                
    train_config, test_config = set_config(current_task_name, 1, seed_num, model_name = model_type, data_n_train = full_data_set_length, load_in_8bit = load_in_8bit)
    
    if 'step' in variation_suffix:
        train_config['max_new_tokens'] = 1024
        xxxxx = train_config['model_name']
        if 'llama' in xxxxx.lower():
            train_config['per_device_train_batch_size'] = 2

    if not enable_full_set:
        save_steps = int(full_data_set_length * epoch_num / save_chekpoints_num/ (train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']))
        train_config['save_steps'] = save_steps
    train_config['num_train_epochs'] = epoch_num
    train_config['learning_rate'] = learning_rate
    train_config['r'] = lora_rank

    train_config_curriculum_learning = train_config.copy()

    warmup_steps = int(full_data_set_length * epoch_num * 0.1/ (train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']))
    train_config['warmup_steps'] = warmup_steps
    batch_size = 1 * train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']
    if full_data_set_length < batch_size:
        batch_size = full_data_set_length
    
    data_name = train_method + '_' + str(n_train) + '_' + train_task_name + variation_suffix + '_train'

    check_point_folder = train_llama_factory(intermediate_train_file_path, output_folder_name, train_config, file_name, dpo_enable = False, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)

    checkpoints = extract_checkpoint_names(check_point_folder)
    if enable_full_set:
        checkpoints = ['full_set']
    else:
        checkpoints.append('full_set')
    
    checkpoints_temp = []
    for checkpoint in checkpoints:
        if checkpoint != 'full_set':
            numbers = re.findall(r'\d+', checkpoint)
            checkpoint_num = int(numbers[0])
            checkpoints_temp.append(checkpoint)
        else:
            checkpoints_temp.append(checkpoint)
    checkpoints = checkpoints_temp

    checkpoint_iteration = 0
    for checkpoint in checkpoints:
        checkpoint_iteration += 1
        if checkpoint != 'full_set':
            numbers = re.findall(r'\d+', checkpoint)
            checkpoint_num = int(numbers[0])
            train_num = checkpoint_num * batch_size
        else:
            train_num = epoch_num * full_data_set_length
            checkpoint_num = 99999

        check_point_folder_temp = ''
        if checkpoint != 'full_set':
            check_point_folder_temp = check_point_folder + '/' + checkpoint
        else:
            check_point_folder_temp = check_point_folder
    
        torch.cuda.empty_cache()
        gc.collect()

        test_data_list = load_evaluation_dataset('validation', n_validation, test_task_name, train_task_name, train_method, HOME_DIRECTORY, variation_suffix)

        data_name = train_method + '_' + str(n_validation) + '_' + test_task_name.lower() +'_validation'

        accuracy, cover_ratio = EVALUATION_LLAMA_FACTORY(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = check_point_folder_temp, task_name = task_name, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)
        
        num_train_epochs, learning_rate = train_config['num_train_epochs'], train_config['learning_rate']
        if checkpoint_num == 99999:
            log_line = f'{task_name} train_num: {train_num} total_epoch_num: {num_train_epochs}, {task_name} learning_rate: {learning_rate}'
        else:
            log_line = f'{task_name} train_num: {train_num} checkpoint_iteration: {checkpoint_iteration} total_epoch_num: {num_train_epochs}, {task_name} learning_rate: {learning_rate}'
        write_log(file_name, output_folder_name, log_line)                      

        log_line = f'{accuracy}'
        if accuracy > max_accuracy or max_accuracy == 0:
            max_accuracy = accuracy
            best_train_num = train_num
            max_learning_rate = learning_rate
            best_model_dir = f"{MODEL_DIRECTORY}/output/{output_folder_name}/bestmodel"
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
                time.sleep(1)
            shutil.copytree(check_point_folder_temp, best_model_dir)

        torch.cuda.empty_cache()
        gc.collect()
        

Best_lora_dir = f"{MODEL_DIRECTORY}/output/{output_folder_name}/bestmodel"
write_log(file_name, output_folder_name, f'{current_stage} Stage: Best validation best_train_num: {best_train_num} Best validation learning_rate: {max_learning_rate} Best validation accuracy: {max_accuracy}')

if not disable_final_eval:
    write_log(file_name, output_folder_name, f"""

# --------------------------{current_stage} Stage: {train_method} Final Evaluation--------------------------""")
    zeroshot = False
    task_name = f'best_model_evaluation_{current_stage}'
    for test_task_name in test_task_name_list:
        test_data_list = load_evaluation_dataset('test', n_eval, test_task_name, train_task_name, train_method, HOME_DIRECTORY, variation_suffix)
        
        current_task_name = test_task_name.lower()
        train_config, test_config = set_config(current_task_name, 1, seed_num, model_name = model_type)
        data_name = current_task_name + '_full_' + train_method 

        if 'step' in variation_suffix:
            test_config['max_new_tokens'] = 1024
            xxxxx = test_config['model_name']
            if 'llama' in xxxxx.lower():
                test_config['per_device_train_batch_size'] = 2

        accuracy, cover_ratio = EVALUATION_LLAMA_FACTORY(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = Best_lora_dir, task_name = task_name, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)

        write_log(file_name, output_folder_name, f"""





""", accuracy = accuracy, lr = sft_lr, n_train = n_train, seed_num = seed_num, model_type = model_type)



recorder_folder_path = f"{HOME_DIRECTORY}/log_total/experiment_data_recorder/{model_type_name}/{n_train}/{sft_lr}/{sft_epoch_list[0]}"
record_file_name = f"{train_task_name}_{train_method}_{variation_suffix}_{seed_num}.txt"

record_accuracy(recorder_folder_path, record_file_name, accuracy)


# Check if the destination directory exists, and if so, remove it
if os.path.exists(LLAMA_FACTORY_DIRECTORY_new):
    shutil.rmtree(LLAMA_FACTORY_DIRECTORY_new)
    print(f"Existing directory {LLAMA_FACTORY_DIRECTORY_new} removed")
    time.sleep(10)

model_output_folder = f"{MODEL_DIRECTORY}/output/{output_folder_name}"
if os.path.exists(model_output_folder):
    shutil.rmtree(model_output_folder)
    print(f"Existing directory {model_output_folder} removed")
    time.sleep(10)

time.sleep(5)

a = 1
