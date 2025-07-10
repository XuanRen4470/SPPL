import os
import json

parent_dir = os.path.dirname(os.path.abspath(__file__))
HOME_DIRECTORY = parent_dir
n_data_creation = 100

file_path = f'{HOME_DIRECTORY}/winogrande_minimum_change_100.json'
# Open the file and read line by line
with open(file_path, 'r') as file:
    data = json.load(file)

for i in range(len(data)):
    data[i]['question'] = data[i]['question'].replace('____', '____?')

# Write the list of JSON objects to a new JSON file
with open(file_path, 'w') as file:
    # Dump the list as JSON into the file, set `indent` for pretty printing
    json.dump(data[:n_data_creation], file, indent=4)

print(f"Converted JSON saved to {file_path}")

