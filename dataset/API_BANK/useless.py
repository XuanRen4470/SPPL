# import os
# import json

# # path_1 = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/API_BANK/"
# # path_1 = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/API_BANK/varient/"

# # path_1 = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_GENERATION/'
# path_1 = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH_GENERATION/varient/'
# # 获取路径下所有文件名
# json_files = [f for f in os.listdir(path_1) if f.endswith('.json')]

# all_data = []

# # 读取每个 JSON 文件
# for json_file in json_files:
#     if 'test' in json_file or 'validation' in json_file or 'groundtruth' in json_file:
#         a = 1
#     else:
#         file_path = os.path.join(path_1, json_file)
#         data_temp = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             for item in data:
#                 question = item['question']
#                 question = question.replace("\n\nPlease infer first, then place the plan at the end, after 'Final Answer:'. The plan you place after 'Final Answer:' should be written in triplet format, which contains (action, object_1, object_2). For example, (unstack red blue) means that you unstack the red object from the blue object.", '')
#                 question = question.replace("Please inference first then provide the final plan at the end after the word 'Final Answer:'", '')
#                 question = question + "\n\nPlease infer first, then place the plan at the end, after 'Final Answer:'. The plan you place after 'Final Answer:' should be written in triplet format, which contains (action, object_1, object_2). For example, (unstack red blue) means that you unstack the red object from the blue object."
#                 item['question'] = question
#                 data_temp.append(item)
#         with open(file_path, 'w') as f:
#             json.dump(data_temp, f, indent=4)
        

# # 打印所有数据
# print(all_data)



import os
import json

path_1 = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/API_BANK/"
# path_1 = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/API_BANK/varient/"
# 获取路径下所有文件名
json_files = [f for f in os.listdir(path_1) if f.endswith('.json')]

all_data = []

def modify_answer(s, gold_label):
    # Find the position of "Final Answer:" in the string
    start_index = s.find("Final Answer:")
    
    # If "Final Answer:" is found, extract the substring after it
    if start_index != -1:
        # Extract the substring after "Final Answer:"
        answer = s[start_index + len("Final Answer:"):].strip()

        result_temp = answer.replace("'", '')
        gold_label = gold_label.replace("'", '')

        if result_temp == gold_label:
            answer = s[:start_index]
            answer = answer + "Final Answer: " + gold_label
            return answer
        else: 
            return None
    else:
        return None  # If "Final Answer:" is not found


# 读取每个 JSON 文件
for json_file in json_files:
    file_path = os.path.join(path_1, json_file)
    data_temp = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            correctness = item['correct']
            if not correctness:
                answer_replace = item['answer']
                gold_label = item['gold_label']
                gold_label = gold_label.replace('Final Answer: ', '')
                answer_replace = modify_answer(answer_replace, gold_label)
                if answer_replace:
                    item['answer'] = answer_replace
                    item['gold_label'] = gold_label
                    item['correct'] = True
            data_temp.append(item)
    with open(file_path, 'w') as f:
        json.dump(data_temp, f, indent=4)
        
a = 1
# # 打印所有数据
# print(all_data)


# import transformers
# import torch

# # Path to the locally saved model
# model_path = "/gpfs/users/a1796450/ACL_2024/model/Meta-Llama-3-8B-Instruct-temp/"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_path,  # Use local model path here
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# )

# messages = [
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "Suppose $f$ and $g$ are polynomials, and that $h(x)=f(g(x))+g(x)$.  Find the degree of $g(x)$ given that the degree of $h(x)$ is $6$ and the degree of $f(x)$ is $2$."},
# ]

# prompt = pipeline.tokenizer.apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=True
# )

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipeline(
#     prompt,
#     max_new_tokens=1024,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )

# print(outputs[0]["generated_text"][len(prompt):])

# a = 1



# import os
# import json
# import ast

# # Specify the directory
# directory_1 = '/gpfs/users/a1796450/gorilla/berkeley-function-call-leaderboard/result/gpt-4o-mini-2024-07-18/'
# directory_2 = '/gpfs/users/a1796450/gorilla/berkeley-function-call-leaderboard/data/possible_answer/'
# directory_3 = '/gpfs/users/a1796450/gorilla/berkeley-function-call-leaderboard/data/'

# # List all files in the directory
# files_1 = [f for f in os.listdir(directory_1) if f.endswith('.json')]
# files_2 = [f for f in os.listdir(directory_2) if f.endswith('.json')]
# files_3 = [f for f in os.listdir(directory_3) if f.endswith('.json')]


# files_1.remove('BFCL_v3_irrelevance_result.json')

# source_code = "SQLCompletionAnalyzer.makeProposalsFromObject(object=Customers, useShortName=True, params={\"limit\": [50], \"schemaFilter\": [\"public\"]})"

# # Convert the curly braces in the params field to escaped single quotes
# temp = source_code.replace("{", "\'{")
# temp = temp.replace("}", "}\'")

# # Check the result
# print(temp)

# # source_code = "[SQLCompletionAnalyzer.makeProposalsFromObject(object='Customers', useShortName='true', params='{\"limit\": \"50\", \"schema\": \"public\"}')]"

# def generate_dynamic_api_call(input_str):
#     parsed_data = input_str
#     def process_data(data):
#         for key, value in data:
#             content = ''
#             for pair_key in value:
#                 content += pair_key
#                 content += '='
#                 temp = str(value[pair_key][0])
#                 # temp = temp.replace("'", '\\"')
#                 # temp = temp.replace("\\", '')
#                 # temp = temp.replace("{", "\'{")
#                 # temp = temp.replace("}", "}\'")
#                 content += temp + ', '
#             content = content[:-2]
#             tt = f"{key}({content})"
#             # tt = '[SQLCompletionAnalyzer.makeProposalsFromObject(object=Customers, useShortName=True, params=\'{"limit": ["50"], "schemaFilter": ["public"]}\')]'
#             return tt
#     api_call_str = "[" + process_data(parsed_data) + "]"
#     return api_call_str

# # Read all JSON files
# json_data = []
# response_answer_list = []
# response_answer_list_2 = []
# original_data_list = []
# for file in files_1:
#     task_name = file.replace('BFCL_v3_', '')
#     task_name = task_name.replace('_result.json', '')
#     json_data_1 = []
#     json_data_2 = []
#     json_data_3 = []
    
#     file_path_1 = os.path.join(directory_1, file)
#     with open(file_path_1, 'r') as f:
#         for line in f:
#             json_data_1.append(json.loads(line))

#     file_2 = file.replace('_result', '')
#     file_path_2 = os.path.join(directory_2, file_2)
#     with open(file_path_2, 'r') as f:
#         for line in f:
#             json_data_2.append(json.loads(line))

#     file_2 = file.replace('_result', '')
#     file_path_3 = os.path.join(directory_3, file_2)
#     with open(file_path_3, 'r') as f:
#         for line in f:
#             ttemp = json.loads(line)
#             ttemp['task_name'] = task_name
#             json_data_3.append(ttemp)
#     original_data_list.extend(json_data_3)

#     for item_1, item_2 in zip(json_data_1, json_data_2):
#         temp = {}
#         question_part_1 = item_1['question'][0]['content']
#         question_part_2 = item_1['question'][1]['content']
#         question = \
# f"""{question_part_1}

# {question_part_2}"""
#         for key, value in item_2['ground_truth'][0].items():
#             answer = f'{key}: {value}'
#         temp['question'] = question
#         temp['id'] = item_2['id']
#         temp['input'] = ''
#         answer_ = item_2['ground_truth'][0].items()
#         answer_temp = generate_dynamic_api_call(answer_)
#         print(answer_temp)
#         temp['answer'] = answer_temp
#         temp['gold_label'] = answer_temp
#         json_data.append(temp)

#         temp_1 = {}
#         temp_1['id'] = item_2['id']
#         temp_1['result'] = answer_temp
#         response_answer_list.append(temp_1)

#         # temp_2 = {}
#         # temp_2['id'] = item_2['id']
#         # answer_te = str(answer_)
#         # answer_te = answer_te.replace('dict_items(', '')
#         # answer_te = answer_te[:-1]
#         # temp_2['ground_truth'] = str(answer_te)
#         # response_answer_list_2.append(temp_2)

#         response_answer_list_2.append(item_2)

#         temp_3 = {}
#         temp_3['id'] = item_2['id']
#         temp_3['question'] = item_1['question']

# path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/BFCL/groundtruth.json'
# with open(path, "w") as f:
#     json.dump(json_data, f, indent=4)

# file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/BFCL/prediction.json'
# with open(file_path, 'w') as json_file:
#     for entry in response_answer_list:
#         json.dump(entry, json_file)
#         json_file.write("\n")

# file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/BFCL/original_data.json'
# with open(file_path, 'w') as json_file:
#     for entry in original_data_list:
#         json.dump(entry, json_file)
#         json_file.write("\n")

# file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/BFCL/response_answer.json'
# with open(file_path, 'w') as json_file:
#     for entry in response_answer_list_2:
#         json.dump(entry, json_file)
#         json_file.write("\n")

# # Print or process the loaded JSON data
# print(json_data)





# # from multiprocessing import Process, Manager
# # import os

# # def run_dynamic_test_with_timeout(test_case, completion, timeout_seconds=50):
# #     def run_test(completion, test_case, namespace):
# #         local_namespace = {}  # Local namespace for the executed code
# #         global_namespace = {}  # Empty global namespace, or can use globals()

# #         # Clean up the completion code
# #         completion = completion.replace("\r", "").replace("\t", " ").strip()

# #         try:
# #             # Ensure function definitions are executed in global_namespace
# #             exec(completion, global_namespace)  # Executes in the global_namespace
# #             PASS = True
# #             for test_case_item in test_case:
# #                 try:
# #                     # Execute each test case
# #                     exec(test_case_item, global_namespace, local_namespace)
# #                 except Exception as e:
# #                     PASS = False
# #                     print(f"Error executing test case: {test_case_item}")
# #                     print(f"Exception: {e}")
# #                     break  # Stop at the first exception
# #             namespace['PASS'] = PASS
# #         except Exception as e:
# #             namespace['PASS'] = False
# #             print(f"Error during execution: {e}")

# #     # Use Manager from multiprocessing to create a shared namespace
# #     with Manager() as manager:
# #         namespace = manager.dict()
# #         # Define and start a new process for running the test with the provided code and test cases
# #         process = Process(target=run_test, args=(completion, test_case, namespace))
# #         process.start()

# #         # Wait for the process to complete or for the timeout
# #         process.join(timeout_seconds)

# #         # If the process is still alive after the timeout, it means it's likely stuck in an infinite loop
# #         if process.is_alive():
# #             process.terminate()  # Terminate the stuck process
# #             process.join()  # Ensure process resources are cleaned up
# #             return False  # Return False to indicate the test did not pass (due to timeout)

# #         # Fetch and return the result from the shared namespace
# #         return namespace.get('PASS', False)

# # # Test case provided
# # test_case_2 = {
# #     "question": "You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. \n\n\nTask: Write a python function to find the first repeated character in a given string.\nTest Example: assert first_repeated_char(\"abcabc\") == \"a\"\n\n1. We wish you to answer the question.\n2. You must directly provide the code answer without say anything else. Please not saying anything 'like sure I can help you with'.\n3. The code should be runnable code which means You do not need to add ``` ``` or add python in front of the code.\n4. The test is only used to show you the input structure. Please do not run the test in your answer.\n",
# #     "task_id": 627,
# #     "test_list": [
# #         "assert first_repeated_char(\"abcabc\") == \"a\"",
# #         "assert first_repeated_char(\"abc\") == \"None\"",
# #         "assert first_repeated_char(\"123123\") == \"1\""
# #     ],
# #     "test_setup_code": "",
# #     "challenge_test_list": [],
# #     "input": "",
# #     "answer": \
# # """def first_repeated_char(str1):\n    # Function to find the first repeated character in a given string.\n\n    # Parameters:\n    # str1 (str): Input string to check for repeated characters.\n\n    # Returns:\n    # str: The first repeated character, or \"None\" if no character repeats.\n\n    # Iterate through each character in the string along with its index\n    for index, c in enumerate(str1):\n        # Check if the character has appeared more than once in the substring up to the current index\n        if str1[:index + 1].count(c) > 1:\n            return c  # Return the first repeated character\n    \n    return \"None\"  # Return \"None\" if no repeated character is found"""
# # }

# # # Test the second case
# # result = run_dynamic_test_with_timeout(test_case_2['test_list'], test_case_2['answer'])
# # print(f"Test passed: {result}")



# # import os
# # a = os.getenv('GPT_API')
# # print(a)
# # a = 1
# # from datasets import load_dataset
# # import json
# # # Load the full MBPP dataset
# # dataset_full = load_dataset("mbpp")

# # # Print dataset structure
# # print(dataset_full)

# # train = dataset_full['train']
# # test = dataset_full['test']
# # validation = dataset_full['validation']
# # prompt = dataset_full['prompt']


# # task_id_total = train['task_id']
# # text_total = train['text']
# # code_total = train['code']
# # test_list_total = train['test_list']
# # test_setup_code_total = train['test_setup_code']
# # challenge_test_list_total = train['challenge_test_list']


# # task_id = validation['task_id']
# # text = validation['text']
# # code = validation['code']
# # test_list = validation['test_list']
# # test_setup_code = validation['test_setup_code']
# # challenge_test_list = validation['challenge_test_list']


# # task_id_total += task_id
# # text_total += text
# # code_total += code
# # test_list_total += test_list
# # test_setup_code_total += test_setup_code
# # challenge_test_list_total += challenge_test_list


# # task_id = prompt['task_id']
# # text = prompt['text']
# # code = prompt['code']
# # test_list = prompt['test_list']
# # test_setup_code = prompt['test_setup_code']
# # challenge_test_list = prompt['challenge_test_list']


# # task_id_total += task_id
# # text_total += text
# # code_total += code
# # test_list_total += test_list
# # test_setup_code_total += test_setup_code
# # challenge_test_list_total += challenge_test_list


# # dataset_full = []

# # for i in range(len(task_id_total)):
# #     temp = {}
# #     temp['question'] = text_total[i]
# #     temp['task_id'] = task_id_total[i]
# #     temp['test_list'] = test_list_total[i]
# #     temp['test_setup_code'] = test_setup_code_total[i]
# #     temp['challenge_test_list'] = challenge_test_list_total[i]
# #     temp['input'] = ''
# #     temp['answer'] = code_total[i]
# #     dataset_full.append(temp)

# # path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MBPP/train.json'
# # with open(path, "w") as f:
# #     json.dump(dataset_full, f, indent=4)






# # task_id_total = test['task_id']
# # text_total = test['text']
# # code_total = test['code']
# # test_list_total = test['test_list']
# # test_setup_code_total = test['test_setup_code']
# # challenge_test_list_total = test['challenge_test_list']

# # dataset_full = []
# # for i in range(len(task_id_total)):
# #     temp = {}
# #     temp['question'] = text_total[i]
# #     temp['task_id'] = task_id_total[i]
# #     temp['test_list'] = test_list_total[i]
# #     temp['test_setup_code'] = test_setup_code_total[i]
# #     temp['challenge_test_list'] = challenge_test_list_total[i]
# #     temp['input'] = ''
# #     temp['answer'] = code_total[i]
# #     dataset_full.append(temp)

# # path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MBPP/test.json'
# # with open(path, "w") as f:
# #     json.dump(dataset_full[:300], f, indent=4)

# # path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MBPP/validation.json'
# # with open(path, "w") as f:
# #     json.dump(dataset_full[300:], f, indent=4)





# # import json

# # # Define the file path
# # # file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/NATRAL_PLAN/trip_planning.json"
# # file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/NATRAL_PLAN/calendar_scheduling.json"

# # # Load the JSON file
# # with open(file_path, "r") as file:
# #     data = json.load(file)


# # a = 1

# # from datasets import load_dataset
# # import json


# # # Load the dataset
# # dataset = load_dataset("ucinlp/drop")

# # # Inspect the dataset
# # print(dataset)

# # # train_data = dataset["train"]
# # validation_data = dataset["validation"]



# # # Convert the dataset to a list of dictionaries
# # # validation_data = validation_data["validation"].to_dict()
# # answers_spans = validation_data['answers_spans']
# # passage = validation_data['passage']
# # question = validation_data['question']

# # answer_list = []
# # question_list = []
# # for answer_item, passage_item, question_item in zip(answers_spans, passage, question):
# #     k = len(answer_item['types'])
# #     previous_item = answer_item['spans'][0]
# #     all_same = True
# #     for item in answer_item['spans']:
# #         if item != previous_item:
# #             all_same = False
# #     if all_same:
# #         answer_list.append(answer_item['spans'][0])
# #         q = \
# # f"""Given the passage: {passage_item}
# # Please answer the question: {question_item}
# # """
# #         question_list.append(q)
# # data_list = []
# # for answer_item, question_item in zip(answer_list, question_list):
# #     temp = {}
# #     temp['question'] = question_item
# #     temp['input'] = ''
# #     temp['answer'] = answer_item
# #     temp['gold_label'] = answer_item
# #     data_list.append(temp)
# # # Save as a JSON file
# # path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/DROP/validation.json'
# # with open(path, "w") as f:
# #     json.dump(data_list[:300], f, indent=4)

# # path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/DROP/test.json'
# # with open(path, "w") as f:
# #     json.dump(data_list[300:800], f, indent=4)

# # print("Train dataset saved as a JSON array.")



# # # import glob
# # # import json
# # # import os
# # # from evaluation.eval import find_last_boxed_number_with_simple_format, extract_boxed_content

# # # # Define the path to JSON files
# # # # input_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH_ALGEBRA/test/intermediate_algebra/*.json"
# # # input_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH_ALGEBRA/test/geometry/*.json"

# # # # Initialize an empty list to store the combined JSON data
# # # combined_data = []

# # # # Iterate through all JSON files in the specified directory
# # # for file_path in glob.glob(input_path):
# # #     with open(file_path, 'r') as f:
# # #         data = json.load(f)
# # #         answer = data['solution']
# # #         if find_last_boxed_number_with_simple_format(answer):
# # #             num = extract_boxed_content(answer)
# # #             print(num)
# # #             if num:
# # #                 answer = answer + f'\nFinal Answer: {num}'
# # #                 temp = {}
# # #                 temp['question'] = data['problem']
# # #                 temp['input'] = ''
# # #                 temp['level'] = data['level']
# # #                 temp['gold_label'] = num
# # #                 temp['numerical_final_answer'] = num
# # #                 temp['groundtruth'] = data['solution']
# # #                 temp['answer'] = answer
# # #                 combined_data.append(temp)
# # #             else:
# # #                 # print(num)
# # #                 a = 1
# # #         else:
# # #             # num = extract_boxed_content(answer)
# # #             # print(num)
# # #             a = 1

# # # # input_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH_GEOMETRY/groundtruth.json"

# # # # with open(input_path, 'w') as f:
# # # #     json.dump(combined_data, f, indent=4)

# # # # Define the output path for the combined JSON file
# # # output_path = os.path.expanduser("/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH_GEOMETRY/validation.json")
# # # # Save the combined data to a single JSON file
# # # with open(output_path, 'w') as f:
# # #     json.dump(combined_data[:], f, indent=4)

# # # output_path = os.path.expanduser("/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH_GEOMETRY/test.json")
# # # with open(output_path, 'w') as f:
# # #     json.dump(combined_data[:], f, indent=4)
# # # print(f"Combined JSON file saved at: {output_path}")




# # # output_file = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/test.json"
# # # with open(output_file, 'w') as f:
# # #     json.dump(data_list_temp[1000:], f, indent=4)  # Use `indent=4` for pretty-printed JSON




# # # output_file = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/test.json"
# # # with open(output_file, 'w') as f:
# # #     json.dump(data_list_temp[1000:], f, indent=4)  # Use `indent=4` for pretty-printed JSON




# # # import pandas as pd
# # # import json

# # # # Define file paths
# # # input_file = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/test-00000-of-00001.parquet"
# # # output_file = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/groundtruth.json"

# # # # Read the Parquet file into a DataFrame
# # # df = pd.read_parquet(input_file)

# # # # Convert the DataFrame to a list of dictionaries
# # # data_list = df.to_dict(orient="records")
# # # # data_list = data_list[:1000]

# # # data_list_temp = []

# # # for item in data_list:
# # #     category = item['category']
# # #     if 'physics' in category.lower():
# # #         temp = {}
# # #         question = item['question']
# # #         choices = item['options']
# # #         answer = item['answer']
# # #         option = ''
# # #         cc = ''
# # #         for kk, item_temp in enumerate(choices):
# # #             if kk == 0:
# # #                 cc = 'A'
# # #             elif kk == 1:
# # #                 cc = 'B'
# # #             elif kk == 2:
# # #                 cc = 'C'
# # #             elif kk == 3:
# # #                 cc = 'D'
# # #             elif kk == 4:
# # #                 cc = 'E'
# # #             elif kk == 5:
# # #                 cc = 'F'
# # #             elif kk == 6:
# # #                 cc = 'G'
# # #             elif kk == 7:
# # #                 cc = 'H'
# # #             elif kk == 8:
# # #                 cc = 'I'
# # #             elif kk == 9:
# # #                 cc = 'J'
# # #             option += f'\n({cc}): "{item_temp}"'

# # #         question = 'We have the question.\n"' + question + f'"\n Here is the options{option}' + '\n\nWhat is the correct option?'
# # #         if cc:
# # #             temp['question'] = question
# # #             temp['input'] = ''
# # #             temp['answer'] = answer
# # #             temp['gold_label'] = answer
# # #             data_list_temp.append(temp)
# # # # Write the list to a JSON file
# # # with open(output_file, 'w') as f:
# # #     json.dump(data_list_temp[:1000], f, indent=4)  # Use `indent=4` for pretty-printed JSON

# # # output_file = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/validation.json"
# # # with open(output_file, 'w') as f:
# # #     json.dump(data_list_temp[1000:], f, indent=4)  # Use `indent=4` for pretty-printed JSON

# # # output_file = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/test.json"
# # # with open(output_file, 'w') as f:
# # #     json.dump(data_list_temp[1000:], f, indent=4)  # Use `indent=4` for pretty-printed JSON


# # # print(f"JSON file saved to {output_file}")





# # from datasets import load_dataset

# # # Load the Rowan/hellaswag dataset
# # dataset = load_dataset("Rowan/hellaswag")

# # # Access specific splits
# # train_data = dataset["train"]
# # validation_data = dataset["validation"]
# # test_data = dataset["test"]

# # # Display some examples
# # ind_list = train_data['ind']
# # question_list = train_data['ctx']
# # options_list = train_data['endings']
# # answer_list = train_data['label']

# # data_list = []
# # for i in range(1000):
# #     temp = {}
# #     option = ''
# #     for k, item in enumerate(options_list[i]):
# #         option += f'\n({k+1}): "{item}"'
# #     question = 'Please complete the paragraph.\n"' + question_list[i] + f'"\n Here is the options{option}' + '\n\nWhat is the correct option?'
# #     temp['question'] = question
# #     temp['option'] = options_list[i]
# #     temp['answer'] = str(int(answer_list[i]) + 1)
# #     temp['gold_label'] = str(int(answer_list[i]) + 1)
# #     temp['input'] = ''
# #     data_list.append(temp)

# # import json
# # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/HELLASWAG/groundtruth.json'
# # with open(file_path, 'w') as f:
# #     json.dump(data_list, f, indent=4)





# # ind_list = validation_data['ind']
# # question_list = validation_data['ctx']
# # options_list = validation_data['endings']
# # answer_list = validation_data['label']

# # data_list = []
# # for i in range(300):
# #     temp = {}
# #     option = ''
# #     for k, item in enumerate(options_list[i]):
# #         option += f'\n({k+1}): "{item}"'
# #     question = 'Please complete the paragraph.\n"' + question_list[i] + f'"\n Here is the options{option}' + '\n\nWhat is the correct option?'
# #     temp['question'] = question
# #     temp['option'] = options_list[i]
# #     temp['answer'] = str(int(answer_list[i]) + 1)
# #     temp['gold_label'] = str(int(answer_list[i]) + 1)
# #     temp['input'] = ''
# #     data_list.append(temp)
# # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/HELLASWAG/validation.json'
# # with open(file_path, 'w') as f:
# #     json.dump(data_list, f, indent=4)






# # ind_list = validation_data['ind']
# # question_list = validation_data['ctx']
# # options_list = validation_data['endings']
# # answer_list = validation_data['label']

# # data_list = []
# # for i in range(300,1300):
# #     temp = {}
# #     option = ''
# #     for k, item in enumerate(options_list[i]):
# #         option += f'\n({k+1}): "{item}"'
# #     question = 'Please complete the paragraph.\n"' + question_list[i] + f'"\n Here is the options{option}' + '\n\nWhat is the correct option?'
# #     temp['question'] = question
# #     temp['option'] = options_list[i]
# #     temp['answer'] = str(int(answer_list[i]) + 1)
# #     temp['gold_label'] = str(int(answer_list[i]) + 1)
# #     temp['input'] = ''
# #     data_list.append(temp)

# # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/HELLASWAG/test.json'
# # with open(file_path, 'w') as f:
# #     json.dump(data_list, f, indent=4)
# # a = 1

# # # import pyarrow.parquet as pq

# # # file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/test-00000-of-00001.parquet"
# # # table = pq.read_table(file_path)

# # # df = table.to_pandas()  # Convert to pandas DataFrame
# # # print(df.head())  # Display the first few rows

# # # answer_choice_list = []
# # # question_list = df['question'].to_list()
# # # category_list = df['category'].to_list()
# # # option_list = df['options'].to_list()
# # # answer_list = df['answer']
# # # answer_index_list = df['answer_index'].to_list()
# # # dataset_list = []
# # # for q, c, o, a, ai in zip(question_list, category_list, option_list, answer_list, answer_index_list):
# # #     temp = {}
# # #     options = '\nWhat is the correct option?'
# # #     for i, o_item in enumerate(o):
# # #         options += '\n({i}) : ' + str(o_item) + '\n'
# # #     if 'physics' in c:
# # #         temp['question'] = q
# # #         temp['category'] = c
# # #         temp['options'] = options
# # #         # temp['answer'] = a
# # #         temp['input'] = ''
# # #         temp['answer'] = f'({ai})'
# # #         dataset_list.append(temp)

# # # a = 1

# # # import json
# # # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU_PRO/groundtruth_physics.json'
# # # with open(file_path, 'w') as f:
# # #     json.dump(dataset_list, f, indent=4)



# # import pyarrow.parquet as pq

# # dataset_list = []



# # file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU/test-00000-of-00001.parquet"

# # table = pq.read_table(file_path)

# # df = table.to_pandas()  # Convert to pandas DataFrame
# # print(df.head())  # Display the first few rows

# # answer_choice_list = []
# # question_list = df['question'].to_list()
# # category_list = df['subject'].to_list()
# # option_list = df['choices'].to_list()
# # answer_list = df['answer'].to_list()
# # for q, c, o, a in zip(question_list, category_list, option_list, answer_list):
# #     temp = {}
# #     temp['question'] = q
# #     temp['subject'] = c
# #     options = []
# #     for item in o:
# #         options.append(item)
# #     temp['choices'] = options
# #     if a == 0:
# #         a = 'A'
# #     elif a == 1:
# #         a = 'B'
# #     elif a == 2:
# #         a = 'C'
# #     elif a == 3:
# #         a = 'D'
# #     temp['answer'] = a
# #     temp['input'] = ''
# #     dataset_list.append(temp)








# # file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU/validation-00000-of-00001.parquet"


# # table = pq.read_table(file_path)

# # df = table.to_pandas()  # Convert to pandas DataFrame
# # print(df.head())  # Display the first few rows

# # answer_choice_list = []
# # question_list = df['question'].to_list()
# # category_list = df['subject'].to_list()
# # option_list = df['choices'].to_list()
# # answer_list = df['answer'].to_list()
# # for q, c, o, a in zip(question_list, category_list, option_list, answer_list):
# #     temp = {}
# #     temp['question'] = q
# #     temp['subject'] = c
# #     options = []
# #     for item in o:
# #         options.append(item)
# #     temp['choices'] = options
# #     if a == 0:
# #         a = 'A'
# #     elif a == 1:
# #         a = 'B'
# #     elif a == 2:
# #         a = 'C'
# #     elif a == 3:
# #         a = 'D'
# #     temp['answer'] = a
# #     temp['input'] = ''
# #     dataset_list.append(temp)

# # a = 1


# # file_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU/dev-00000-of-00001.parquet"


# # table = pq.read_table(file_path)

# # df = table.to_pandas()  # Convert to pandas DataFrame
# # print(df.head())  # Display the first few rows

# # answer_choice_list = []
# # question_list = df['question'].to_list()
# # category_list = df['subject'].to_list()
# # option_list = df['choices'].to_list()
# # answer_list = df['answer'].to_list()
# # for q, c, o, a in zip(question_list, category_list, option_list, answer_list):
# #     temp = {}
# #     temp['question'] = q
# #     temp['subject'] = c
# #     options = []
# #     for item in o:
# #         options.append(item)
# #     temp['choices'] = options
# #     if a == 0:
# #         a = 'A'
# #     elif a == 1:
# #         a = 'B'
# #     elif a == 2:
# #         a = 'C'
# #     elif a == 3:
# #         a = 'D'
# #     temp['answer'] = a
# #     temp['input'] = ''
# #     dataset_list.append(temp)





# # import json
# # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU/groundtruth.json'
# # with open(file_path, 'w') as f:
# #     json.dump(dataset_list[:1000], f, indent=4)

# # dataset_list = dataset_list[1000:]

# # import json
# # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU/validation.json'
# # with open(file_path, 'w') as f:
# #     json.dump(dataset_list[-200:], f, indent=4)

# # dataset_list = dataset_list[:-200]
# # import json
# # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MMLU/test.json'
# # with open(file_path, 'w') as f:
# #     json.dump(dataset_list, f, indent=4)










# # # import json
# # # import random

# # # # Define the file path
# # # # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/train_plan_generation_total.json'
# # # # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/anthropic_generated_plan_bench_False_800_total.json'
# # # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/gpt4_generated_plan_bench_False_800_total.json'
# # # # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/varient/openai_mini_gpt4_total.json'

# # # # Read the JSON data
# # # with open(file_path, 'r') as f:
# # #     data_list = json.load(f)

# # # # Ensure that the data is a list before shuffling
# # # if isinstance(data_list, list):
# # #     # Shuffle the list in place
# # #     random.shuffle(data_list)
# # #     print("Data has been shuffled successfully.")
# # # else:
# # #     print("The JSON data is not a list. Shuffling is not applicable.")

# # # # Write the shuffled data back to the JSON file
# # # with open(file_path, "w") as json_file:
# # #     json.dump(data_list, json_file, indent=4)


# # # a = 1




# # # from datasets import load_dataset
# # # import json

# # # # Specify the desired configuration
# # # task_name_list = ['task_1_plan_generation', 'task_2_plan_optimality', 'task_3_plan_verification', 'task_5_plan_generalization', 'task_7_plan_execution', 'task_8_1_goal_shuffling', 'task_8_2_full_to_partial']
# # # name_list = ['plan_generation', 'plan_optimality', 'plan_verification', 'plan_generalization', 'plan_execution', 'goal_shuffling', 'full_to_partial']

# # # for task_name, name in zip(task_name_list, name_list):
# # #     dataset = load_dataset("tasksource/planbench", task_name)
# # #     dataset = dataset['train']

# # #     data_list = [dict(record) for record in dataset]  # Convert to list of dicts for the selected config

# # #     with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/train_{name}.json", "w") as json_file:
# # #         json.dump(data_list, json_file, indent=4)

# # # a = 1




# # # import json

# # # name_list = ['plan_generation', 'plan_optimality', 'plan_verification', 'plan_generalization', 'plan_execution', 'goal_shuffling', 'full_to_partial']

# # # for name in name_list:
# # #     # Define the file path
# # #     file_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/train_{name}.json"

# # #     # Load the JSON file
# # #     with open(file_path, 'r') as f:
# # #         data = json.load(f)

# # #     cleaned_data = []
# # #     # Rename column names (keys) in each dictionary
# # #     for i, entry in enumerate(data):
# # #         try:
# # #             answer = entry['ground_truth_plan']
# # #             entry['question'] = entry.pop('query')
# # #             entry['input'] = ""
# # #             entry['answer'] = 'Final Answer: ' + answer
# # #             entry['gold_label'] = answer
            
# # #             cleaned_data.append(entry)
# # #         except:
# # #             a = 1

# # #     # Save the modified data back to the JSON file
# # #     with open(file_path, 'w') as f:
# # #         json.dump(cleaned_data, f, indent=4)

# # #     print("Column names have been updated.")

















# # # import json
# # # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/LLMs-Planning-main/plan-bench/prompts/blocksworld/task_1_plan_generation.json'
# # # with open(file_path, 'r') as f:
# # #     data_list = json.load(f)
# # # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/blocksworld_task_1_plan_generation.json", "w") as json_file:
# # #     json.dump(data_list, json_file, indent=4)

# # # cleaned_data = []
# # # # Rename column names (keys) in each dictionary
# # # for i, entry in enumerate(data_list['instances']):
# # #     try:
# # #         answer = entry['ground_truth_plan']
# # #         entry['question'] = entry.pop('query')
# # #         entry['input'] = ""
# # #         entry['answer'] = 'Final Answer: ' + answer
# # #         entry['gold_label'] = answer
# # #         cleaned_data.append(entry)
# # #     except:
# # #         a = 1

# # # # Save the modified data back to the JSON file
# # # with open(file_path, 'w') as f:
# # #     json.dump(cleaned_data, f, indent=4)

# # # print("Column names have been updated.")


# # # import json
# # # file_path = '/gpfs/users/a1796450/ACL_2024/Minimum_Change/LLMs-Planning-main/plan-bench/prompts/blocksworld/task_1_plan_generation.json'
# # # with open(file_path, 'r') as f:
# # #     data_list = json.load(f)

# # # train_data = data_list[:300]
# # # test_data = data_list[300:]
# # # validation_data = data_list[300:]
# # # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/validation_plan_generalization.json", "w") as json_file:
# # #     json.dump(validation_data, json_file, indent=4)
# # # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/train_plan_generalization.json", "w") as json_file:
# # #     json.dump(train_data, json_file, indent=4)
# # # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/test_plan_generation.json", "w") as json_file:
# # #     json.dump(test_data, json_file, indent=4)


# # import json
# # file_name_list = ['anthropic_generated_plan_bench_False_1000', 'gpt4_generated_plan_bench_False_1000', 'openai_mini_gpt4']
# # # /gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/anthropic_generated_plan_bench_False_1000.json
# # # /gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/gpt4_generated_plan_bench_False_1000.json
# # # /gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/varient/openai_mini_gpt4.json


# # # file_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/anthropic_generated_plan_bench_False_1000.json'
# # # with open(file_path, 'r') as f:
# # #     data_list = json.load(f)

# # # data_list_temp = []
# # # for item in data_list:
# # #     domain = item['domain']
# # #     task = item['task']
# # #     if domain == 'blocksworld' and task == 'task_1_plan_generation':
# # #         item["domain"] = "blocksworld"
# # #         data_list_temp.append(item)
# # # data_list_temp = data_list_temp[:300]
# # # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/anthropic_generated_plan_bench_False_1000.json", "w") as json_file:
# # #     json.dump(data_list_temp, json_file, indent=4)



# # # file_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/gpt4_generated_plan_bench_False_1000.json'
# # # with open(file_path, 'r') as f:
# # #     data_list = json.load(f)

# # # data_list_temp = []
# # # for item in data_list:
# # #     domain = item['domain']
# # #     task = item['task']
# # #     if domain == 'blocksworld' and task == 'task_1_plan_generation':
# # #         item["domain"] = "blocksworld"
# # #         data_list_temp.append(item)
# # # data_list_temp = data_list_temp[:300]
# # # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/gpt4_generated_plan_bench_False_1000.json", "w") as json_file:
# # #     json.dump(data_list_temp, json_file, indent=4)



# # # file_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/varient/openai_mini_gpt4.json'
# # # with open(file_path, 'r') as f:
# # #     data_list = json.load(f)

# # # data_list_temp = []
# # # for item in data_list:
# # #     domain = item['domain']
# # #     task = item['task']
# # #     if domain == 'blocksworld' and task == 'task_1_plan_generation':
# # #         item["domain"] = "blocksworld"
# # #         data_list_temp.append(item)
# # # data_list_temp = data_list_temp[:300]
# # # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/varient/openai_mini_gpt4.json", "w") as json_file:
# # #     json.dump(data_list_temp, json_file, indent=4)







# # # domain_name = ''
# # domain_name = 'blocksworld'
# # # domain_name = 'blocksworld_3'
# # # domain_name = 'logistics'
# # # domain_name = 'mystery_blocksworld'

# # # domain_name = 'depots'

# # n_data = 3000

# # for domain_name in ['blocksworld', 'blocksworld_3', 'logistics', 'depots']:#, 'mystery_blocksworld']:
# #     # if domain_name == 'logistics':
# #     #     n_data = 100
# #     file_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/old/train_plan_generation.json'
# #     with open(file_path, 'r') as f:
# #         data_list = json.load(f)

# #     data_list_temp = []
# #     for item in data_list:
# #         domain = item['domain']
# #         task = item['task']
# #         if domain_name == domain and task == 'task_1_plan_generation':
# #             item["domain"] = domain
# #             data_list_temp.append(item)

# #         # if domain != domain_name:
# #         #     # print('-----------------------------')
# #         #     domain_name = domain
# #         #     print(domain_name)
# #         #     # print(item['answer'])
# #         #     # print('-----------------------------')
        
# #     data_list_train = data_list_temp[:n_data]
# #     # data_list_test = data_list_temp[n_data:]
# #     with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/train_plan_generation_{domain_name}.json", "w") as json_file:
# #         json.dump(data_list_train, json_file, indent=4)
# #     # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/test_plan_generation_{domain_name}.json", "w") as json_file:
# #     #     json.dump(data_list_test, json_file, indent=4)
# #     # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/validation_plan_generation_{domain_name}.json", "w") as json_file:
# #     #     json.dump(data_list_test, json_file, indent=4)


# # # # a = 1



# # data_list_train = []
# # data_list_test = []
# # for domain_name in ['blocksworld', 'blocksworld_3', 'logistics', 'depots']:#, 'mystery_blocksworld']:
# #     file_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/train_plan_generation_{domain_name}.json'
# #     with open(file_path, 'r') as f:
# #         data_list = json.load(f)
# #     print(len(data_list))
# #     print(data_list[0]['answer'])
# #     print()
# #     print()
# #     if domain_name == 'blocksworld':
# #         n_data = 361
# #     if domain_name == 'blocksworld_3':
# #         n_data = 72
# #     if domain_name == 'logistics':
# #         n_data = 206
# #     if domain_name == 'depots':
# #         n_data = 361
# #     temp = data_list[:n_data]
# #     if domain_name == 'blocksworld':
# #         temp = temp[300:]
# #         data_list_train += temp
# #     else:
# #         data_list_train += data_list[:n_data]
# #     data_list_test.append(data_list[n_data:])

# # a = 1
# # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/train_plan_generation_total.json", "w") as json_file:
# #     json.dump(data_list_train, json_file, indent=4)
# # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/test_plan_generation_total.json", "w") as json_file:
# #     json.dump(data_list_test, json_file, indent=4)
# # with open(f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/PLAN_BENCH/validation_plan_generation_total.json", "w") as json_file:
# #     json.dump(data_list_test, json_file, indent=4)

# # a = 1







# # # /gpfs/users/a1796450/ACL_2024/Minimum_Change/LLMs-Planning-main
# # # python3 prompt_generation.py --task t5 --config CONFIG [--ignore_existing] [--specific_instances SPECIFIC-INSTANCES] [--random_example RANDOM-EXAMPLE] [--verbose VERBOSE] [--seed SEED]

# # # python3 response_evaluation.py --task t5 --config CONFIG --engine ENGINE [--ignore_existing] [--verbose VERBOSE]
# # # python3 response_generation.py --task t5 --config CONFIG --engine ENGINE [--ignore_existing] [--run_till_completion RUN-TILL-COMPLETION] 



# # # python3 prompt_generation.py --task t5 --config CONFIG

# # # python3 prompt_generation.py --task t5 --config blocksworld 