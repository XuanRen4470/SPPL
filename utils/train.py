import sys
import os
import shutil
import subprocess

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.llama_factory_data_file_processor import put_file_path_to_data_info
from config.config import MODEL_DIRECTORY, HOME_DIRECTORY, IGNORE_INDEX, LLAMA_FACTORY_ALPACA, tokenizer, train_max_length

def train_llama_factory(train_data_path, output_folder_name, train_config, file_name, dpo_enable = False, merged_base_model_dir = '', data_name = '', LLAMA_FACTORY_DIRECTORY = '', check_point_folder_name = '', enable_perplexity_curriculum_learning_initialization = False):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    file_name = file_name.replace('_log', '')
    put_file_path_to_data_info(data_name, train_data_path, dpo_enable = dpo_enable, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)
    
    seed = train_config['seed_num']

    output_folder_name = f'{MODEL_DIRECTORY}/output/{output_folder_name}/{seed}'
    model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    
    # Construct the command
    if merged_base_model_dir == '':
        model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    else:
        model_path = f"{merged_base_model_dir}"
    stage = "sft"
    per_device_train_batch_size = train_config['per_device_train_batch_size']
    gradient_accumulation_steps = train_config['gradient_accumulation_steps'] 
    # Check if the folder exists
    if os.path.exists(output_folder_name):
        # Remove all files in the folder
        for filename in os.listdir(output_folder_name):
            file_path = os.path.join(output_folder_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # Create the folder if it does not exist
        os.makedirs(output_folder_name)

    if 'save_steps' in train_config: 
        save_steps = train_config['save_steps']
    else:
        save_steps = 50
    
    if train_config['device_num'] > 1:
        start_line = 'accelerate launch'
    else:
        start_line = 'python'

    cmd = [
        start_line,
        f"{LLAMA_FACTORY_DIRECTORY}/src/train_bash.py",
        "--do_train",
        "--stage", stage,
        "--model_name_or_path", model_path,
        "--dataset", data_name,
        "--template", train_config['template'],
        # "--r", train_config['r'],
        "--finetuning_type", train_config['finetune_type'],
        "--lora_target", "q_proj,v_proj",
        "--output_dir", output_folder_name,
        # "--overwrite_cache",
        "--max_length", str(train_config['max_length']),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--lr_scheduler_type", train_config['lr_scheduler_type'],
        "--logging_steps",  str(100),
        "--save_steps", str(save_steps),
        "--learning_rate", str(train_config['learning_rate']),
        "--num_train_epochs", str(train_config['num_train_epochs']),
        "--lora_rank", str(train_config['r']),
        "--overwrite_output_dir", str(True),
        "--plot_loss",
        "--fp16"
    ]

    if 'load_in_8bit' in train_config:
        cmd += ["--quantization_config", str(train_config['load_in_8bit'])]
    
    if 'checkpoint_dir' in train_config:
        cmd += ["--checkpoint_dir", f"{LLAMA_FACTORY_DIRECTORY}/{train_config['checkpoint_dir']}"]
    if dpo_enable:
        if 'DPO_BETA' in train_config:
            cmd += ["--dpo_beta", str(train_config['DPO_BETA'])]
    
    if 'seed' in train_config:
        cmd += ["--seed", str(train_config['seed_num'])]
    if dpo_enable:
        cmd += ['--create_new_adapter']
    # else: 
    #     cmd += ['--overwrite_cache']
    cmd += ['--overwrite_cache']
    if check_point_folder_name != '':
        cmd += ['--adapter_name_or_path', check_point_folder_name]

    if enable_perplexity_curriculum_learning_initialization:
        cmd += ['--streaming']
        cmd += ['--buffer_size', str(1)]

    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)
    check_point_folder_name = output_folder_name
    return check_point_folder_name


def merge_lora_llama_factory(adapter_name_or_path, export_dir, train_config, LLAMA_FACTORY_DIRECTORY = '', merged_base_model_dir = ''):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    if merged_base_model_dir == '':
        model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    else:
        model_path = f"{merged_base_model_dir}"
    
    cmd = [
        "python",
        f"{LLAMA_FACTORY_DIRECTORY}/src/export_model.py",
        "--model_name_or_path", model_path,
        "--template", train_config['template'],
        "--finetuning_type", train_config['finetune_type'],
        "--adapter_name_or_path", adapter_name_or_path,
        "--export_dir", export_dir,
        "--overwrite_cache",
        "--export_size", str(2),
        "--export_legacy_format", str(False)
    ]

    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)
    return export_dir
