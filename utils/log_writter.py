import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import HOME_DIRECTORY


def write_log(file_name, folder_name, log_line, accuracy = -1, lr = '', n_train = '', seed_num = '', model_type = ''):
    print(log_line)
    with open(f"{HOME_DIRECTORY}/log/{folder_name}/{file_name}.txt", 'a') as f:
        f.write(f'{log_line}.\n')

    

        
    if accuracy != -1:
        if lr != '':
            lr = '_lr_' + str(lr)

        if n_train != '':
            n_train = '_' + str(n_train)
        
        if seed_num != '':
            seed_num = '_seed_' + str(seed_num)
        
        # Path to the log file
        log_file_path = f"{HOME_DIRECTORY}/log_total/accuracy{model_type}{n_train}{lr}{seed_num}{model_type}.txt"

        # Check if the file exists and create it if it doesn't
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                # Optionally, you can write a header or leave it empty
                f.write("Log file created.\n")  # You can remove this line if no initial content is needed

        with open(log_file_path, 'a') as f:
            f.write(f"""



{folder_name}
Accuracy: {accuracy}




""")

def write_log_dpo_accuracy_record(file_name, folder_name, log_line):
    with open(f"{HOME_DIRECTORY}/log/{folder_name}/{file_name}.txt", 'a') as f:
        f.write(f'{log_line}.\n')
