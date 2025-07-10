import sys
import os
import hashlib
import json
import shutil
import time
import random
import portalocker
    
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import os


def put_file_path_to_data_info(intermediate_file_name, intermediate_file_path, dpo_enable=False, LLAMA_FACTORY_DIRECTORY = '', pre_train_stage = False):
    intermediate_file_name = intermediate_file_name.replace('_log', '')
    with open(intermediate_file_path, 'r') as f:
        data = json.load(f)

    data_new = []
    if not pre_train_stage:
        key_map = {'question': 'question', 'instruction': 'question', 'answer': 'answer', 'output': 'answer'}
        for item in data:
            temp = {
                'question': item.get(key_map['question'], item.get('instruction', '')),
                'input': '',
                'answer': item.get(key_map['answer'], item.get('output', ''))
            }
            data_new.append(temp)
    else:
        data_new = data

    intermediate_llama_factory_file_path = intermediate_file_path.replace('.json', '_llama_facotry.json')
    with open(intermediate_llama_factory_file_path, 'w') as json_file:
        json.dump(data_new, json_file, indent=4)
    time.sleep(10)
    destination_file = f"{LLAMA_FACTORY_DIRECTORY}/data/{intermediate_file_name}.json"
    shutil.copy(intermediate_llama_factory_file_path, destination_file)

    with open(intermediate_llama_factory_file_path, 'rb') as f:
        bytes = f.read()
        sha1_hash = hashlib.sha1(bytes).hexdigest()

    # Use portalocker to lock 'dataset_info.json' for updating
    dataset_info_path = f"{LLAMA_FACTORY_DIRECTORY}/data/dataset_info.json"
    with open(dataset_info_path, 'r+') as f:
        portalocker.lock(f, portalocker.LOCK_EX)  # Exclusive lock to prevent other processes from writing
        dataset_info = json.load(f)
        if not pre_train_stage:
            new_data = {
                "file_name": f"{intermediate_file_name}.json",
                "file_sha1": sha1_hash,
                "columns": {
                    "prompt": "question",
                    "query": "input",
                    "response": "answer"
                }
            }
        else:
            new_data = {
                "file_name": f"{intermediate_file_name}.json",
                "file_sha1": sha1_hash,
                "columns": {
                    "prompt": "text"
                }
            }

        if dpo_enable:
            new_data["ranking"] = True

        dataset_info[intermediate_file_name] = new_data
        f.seek(0)  # Move to the beginning of the file before writing
        json.dump(dataset_info, f, indent=4)
        f.truncate()  # Truncate the file size in case the new data is smaller
        portalocker.unlock(f)  # Release the lock
    
    time.sleep(10)

# Remember to replace `LLAMA_FACTORY_DIRECTORY` with your actual directory path.


def put_json_list_to_data_info(data, intermediate_file_name, LLAMA_FACTORY_DIRECTORY = '', pre_train_stage = False):
    intermediate_file_path = f"{LLAMA_FACTORY_DIRECTORY}/data/{intermediate_file_name}.json"
    data_new = []
    for item in data:
        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = item['answer']
        data_new.append(temp)

    with open(intermediate_file_path, 'w') as json_file:
        json.dump(data_new, json_file, indent=4)
    time.sleep(3)
    put_file_path_to_data_info(intermediate_file_name, intermediate_file_path, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY, pre_train_stage = pre_train_stage)
    time.sleep(3)

def check_intermediate_file_available(intermediate_file_name, LLAMA_FACTORY_DIRECTORY = ''):
    with open(f"{LLAMA_FACTORY_DIRECTORY}/data/dataset_info.json", 'r') as f:
        dataset_info = json.load(f)
    if intermediate_file_name in dataset_info:
        return True
    else:
        return False


