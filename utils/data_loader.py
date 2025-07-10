import json
import sys
import os
import re
import time
from config.config import HOME_DIRECTORY


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from evaluation.eval import *


def load_MATH(path, n_row, zeroshot = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            num = line['numerical_final_answer']
            original_question = line['question']
            
            if num:
                line['question'] = line['question'] + """

Please provide the final answer (a number) at the end, after 'Final Answer:'
"""
                if zeroshot:
                    line['question'] += f"""

Please put the final digital answer at the end after you finish inference in this format FINAL ANSWER: final neumerical answer

Format:
SOME_INFERENCE

FINAL ANSWER: """
                
                if load_original_question:
                    line['question'] = original_question
                
                line['numerical_final_answer'] = str(evaluate_expression_(num))
                line['original_question'] = original_question
                data_list.append(line)
    
    data_list = data_list[:n_row]
    return data_list

def load_GSM8K(path, n_row, zeroshot = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            question = line['question'] 
            original_question = question
            if zeroshot:
                question += f"""

Please put the final digital answer at the end after you finish inference in this format Final Answer: final neumerical answer

Format:
SOME_INFERENCE

Final Answer: """
            else:
                question += """

Please provide the final answer (a number) at the end, after 'Final Answer:'
"""

            answer = line['answer']
            if zeroshot:
                answer = answer.replace('####', 'Final Answer: ')
            line['answer'] = answer
            if load_original_question:
                line['question'] = original_question
            else:
                line['question'] = question
            line['original_question'] = original_question
            data_list.append(line)
        
    data_list = data_list[:n_row]
    return data_list

def load_AQuaRAT(path, n_row, zeroshot = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            question = line['question'] 
            A = '(' + line['options'][0] 
            B = '(' + line['options'][1] 
            C = '(' + line['options'][2] 
            D = '(' + line['options'][3] 
            E = '(' + line['options'][4] 
            if zeroshot:
                question += f"""

Please put the final digital answer at the end after you finish inference in this format Final Answer: final neumerical answer

Format:
SOME_INFERENCE

Final Answer: """
            if load_original_question:
                question += f"""

Options:
{A}
{B}
{C}
{D} 
{E}
Please choose the correct answer (A)/(B)/(C)/(D)/(E) and place it at the end, after '\n\nFinal Answer: '
"""
            else:
                question += f"""

Options:
{A}
{B}
{C}
{D} 
{E}
Please choose the correct answer (A)/(B)/(C)/(D)/(E) and place it at the end, after '\n\nFinal Answer: '
"""
         
            answer = line['answer']
            if zeroshot:
                answer = answer.replace('####', 'Final Answer: ')
            line['answer'] = answer
            line['question'] = question
            data_list.append(line)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK(path, n_row, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            prompt  = line['instruction'] + line['input']
            temp['question'] = prompt
            temp['input'] = ''
            answer_temp = line['output']
            temp['gold_label'] = line['output']
            temp['answer'] = f'Final Answer: {answer_temp}'
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK_optimized(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                for i in range(1, api_history.count('API-Request:') + 1):
                    if cc > 1:
                        api_history = api_history.replace('API-Request:', f"API-Request{i}:", i)
                for i in range(1, cc+ 1):
                    api_history = api_history.replace(f'API-Request{i}:', f"""{i}. 
API-Request:""")

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


Previous API-Request History:

{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
            if '->' in line['input']:
                end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            else:
                end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            temp['original_question'] = prompt
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_ESNLI(path, n_row, use_gold_label = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            if use_gold_label:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer (Entailment/Neutral/Contradiction) directly.
"""
            else:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. 
"""
            if use_gold_label:
                prompt += \
f"""
Please choose the option directly. Final Answer: (entailment/contradiction/neutral)
"""
            elif load_original_question:
                a = 1
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'
"""
            if not use_gold_label:
                answer_item = line['answer']
                explaination = line['explanation_1']
                explaination = explaination.lower()
                explaination = explaination.replace('.', ',')
                answer = f"""Because {explaination} the answer is {answer_item}.
Final Answer: {answer_item}"""
            else:
                answer_temp = line['answer']
                answer = f'Final Answer: {answer_temp}'
            temp['question'] = prompt
            original_question = f"""We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context.'
"""
            temp['original_question'] = original_question
            
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_PIQA(path, n_row, finetune = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            goal = line['goal']
            sol1 = line['sol1']
            sol2 = line['sol2']
            label = line['gold_label']
            prompt = \
f"""Given the question: {goal}

What option is correct?
Option 1: {sol1}
Option 2: {sol2}
"""
            if load_original_question:
                a = 1
            elif finetune:
                prompt += \
f"""
Please choose the option directly. Answer: (1/2)
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['sol1'] = line['sol1']
            temp['sol2'] = line['sol2']
            temp['gold_label'] = str(label)
            temp['answer'] = str(label)
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_BOOLQ(path, n_row, finetune = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            passage = line['passage']
            label = line['gold_label']
            prompt = \
f"""Given the context: {passage}

{question}?
"""
            temp['original_question'] = prompt
            if not load_original_question:
                if finetune:
                    prompt += \
f"""
Please answer True or False directly. Final Answer: (True/False)
"""
                else:
                    prompt += \
f"""
Please inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['passage'] = passage
            temp['gold_label'] = str(label)
            temp['answer'] = f'Final Answer: {label}'
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_WINOGRANDE(path, n_row, finetune = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        data = data[:n_row]
        for line in data:
            temp = {}
            sentence = line['sentence']
            option1 = line['option1']
            option2 = line['option2']
            label = line['gold_label']
            prompt = \
f"""Given the question: {sentence}

What option is correct?
Option 1: {option1}
Option 2: {option2}
"""
            temp['original_question'] = prompt
            if not load_original_question:
                if finetune:
                    prompt += \
f"""
Please choose the option directly (1/2) and place it after 'Final Answer:'.
"""
                else:
                    prompt += \
f"""
Please inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['option1'] = line['option1']
            temp['option2'] = line['option2']
            temp['gold_label'] = str(label)
            temp['answer'] = f'Final Answer: {label}'
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_TRIVIAQA(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            gold_label = line['gold_label']
            evidence = line['evidence']

            if finetune:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may provide the final answer directly.
"""
            else:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may inference first, then provide the final answer.
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = gold_label
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = gold_label
            temp['answer'] = answer_item
            temp['evidence'] = evidence
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_SQUAD(path, n_row, finetune = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for i, line in enumerate(data):
            temp = {}
            question = line['question']
            context = line['context']
            answer = line['gold_label']['text']
            if answer == []:
                gold_label = 'No answer'
            else:
                gold_label = answer[0]
                prompt = \
f"""Given the question: {question} and the context: {context}

What is the answer?
"""
                if not load_original_question:
                    if finetune:
                        prompt += f"""
Please directly provide the final answer (text span) at the end, after 'Final Answer:'
"""
                    else:
                        prompt += \
f"""
Please inference first, then provide the final answer (text span) at the end, after 'Final Answer:'
"""

                if finetune:
                    gold_label = 'Final Answer: ' + gold_label
                temp['question'] = prompt
                temp['input'] = ''
                temp['context'] = context
                temp['gold_label'] = gold_label
                temp['answer'] = gold_label
                temp['answer_list'] = answer
                data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_DROP(path, n_row, finetune_with_gt = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for i, line in enumerate(data):
            temp = {}
            prompt = line['question']
            gold_label = line['gold_label']

            if not load_original_question:
                if finetune_with_gt:
                    prompt += f"""
Please directly provide the final answer at the end, after 'Final Answer:'
"""
                else:
                    prompt += \
f"""
Please inference first, then provide the final answer at the end, after 'Final Answer:'
"""                

            answer = 'Final Answer: ' + gold_label
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = gold_label
            temp['answer'] = answer
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_NATURAL_QUESTIONS(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            gold_label = line['gold_label']
            evidence = line['evidence']

            if finetune:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may provide the final answer directly.
"""
            else:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may inference first, then provide the final answer.
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = gold_label
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = gold_label
            temp['answer'] = answer_item
            temp['evidence'] = evidence
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_MMLU(path, n_row, finetune = False, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            subject = line['subject']
            A = line['choices'][0]
            B = line['choices'][1]
            C = line['choices'][2]
            D = line['choices'][3]
            gold_label = line['answer']
            prompt = \
f"""Given the question: {question}

and the options:
A: {A}
B: {B}
C: {C}
D: {D}

What is the answer?
"""
            if not load_original_question:
                if finetune:
                    prompt += \
f"""
Please answer directly (A/B/C/D).
"""
                else:
                    prompt += \
f"""
Please inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['subject'] = subject
            temp['A'] = A
            temp['B'] = B
            temp['C'] = C
            temp['D'] = D
            temp['gold_label'] = gold_label
            temp['answer'] = f'Final Answer: {gold_label}'
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_AGIEVAL(path, n_row, finetune = False, category = 'logiqa', load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            passage = line['passage']
            question = line['question']
            options = line['options']
            option_item = \
f"""{options[0]}
{options[1]}
{options[2]}
{options[3]}
"""
            gold_label = line['label']
            prompt = \
f"""Given the statement: {passage}

and the question: {question}

and the options:
{option_item}

What is the answer?

"""
            if category == 'sat':
                prompt = \
f"""Given the context: {passage}

and the question: {question}

and the options:
{option_item}

What is the answer?

"""
            if not load_original_question:
                if finetune:
                    prompt += \
f"""
Please answer directly. Answer: (A/B/C/D)
"""
                else:
                    prompt += \
f"""
Please inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['passage'] = passage
            temp['input'] = ''
            temp['A'] = options[0]
            temp['B'] = options[1]
            temp['C'] = options[2]
            temp['D'] = options[3]
            temp['gold_label'] = gold_label
            temp['answer'] = gold_label
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_ECQA(path, n_row, finetune = False, use_gt_rationale = True, load_original_question = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            question = line['question']
            o1 = line['1']
            o2 = line['2']
            o3 = line['3']
            o4 = line['4']
            o5 = line['5']
            gold_label = line['gold_label']
            answer = line['answer']
            pos_explaination = line['pos_explaination']
            neg_explaination = line['neg_explaination']
            combined_explaination = line['combined_explaination']
            prompt = \
f"""We have the question: {question}
and the options:
(1): {o1}
(2): {o2}
(3): {o3}
(4): {o4}
(5): {o5}

what is the correct option?
"""
            line['original_question'] = prompt
            if not load_original_question:
                if not use_gt_rationale:
                    prompt += \
f"""
Please answer 1/2/3/4/5 directly. after Final Answer: 
"""
                else:
                    prompt += \
f"""
Please inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'
"""
            if use_gt_rationale:
                answer = f"""Inference: {combined_explaination}

Final Answer: """ + answer
            else:
                answer = f"""Final Answer: """ + gold_label

            line['answer'] = answer
            line['question'] = prompt
            data_list.append(line)
        
    data_list = data_list[:n_row]
    return data_list

def load_plan_bench_with_proportion(train_data_list_total, n_train):
    train_data_list = []
    train_data_list_item = []
    domain_ = ''
    for item in train_data_list_total:
        current_domain = item['domain']
        if domain_ == current_domain:
            train_data_list_item.append(item)
        if domain_ != current_domain:
            if train_data_list_item != []:
                train_data_list.append(train_data_list_item)
            domain_ = current_domain
            train_data_list_item = []
            train_data_list_item.append(item)
    train_data_list.append(train_data_list_item)

    train_data_list_temp = []
    for train_data_item in train_data_list:
        n_train_item = int(len(train_data_item) * (n_train/len(train_data_list_total)))
        # print(n_train_item)
        train_data_list_temp+=train_data_item[:n_train_item]
    train_data_list = train_data_list_temp
    return train_data_list

def load_MBPP(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            prompt = line['question']
            test_example = line['test_list'][0]
            
            modified_prompt = \
f"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. 


Task: {prompt}
Test Example: {test_example}

1. We wish you to answer the question.
2. You must directly provide the code answer without say anything else. Please not saying anything 'like sure I can help you with'.
3. The code should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
4. The test is only used to show you the input structure. Please do not run the test in your answer.
"""

            line['question'] = modified_prompt
            data_list.append(line)
    return data_list[:n_row]

def load_groundtruth(train_task_name, n_train, variation_suffix, root_path, HOME_DIRECTORY):
    if 'plan_bench' in train_task_name.lower():
        full_path = f'{root_path}/groundtruth.json'
        with open(full_path, 'r') as f:
            train_data_list = json.load(f)
        train_data_list = train_data_list[:n_train]
        for kkk, test_item in enumerate(train_data_list):
            qqq = test_item['question'] + "\nPlease answer directly and place the final answer at the end after 'Final Answer:'"
            train_data_list[kkk]['question'] = qqq
    
    if train_task_name.upper() =='GSM8K':
        full_path = f'{HOME_DIRECTORY}/dataset/GSM8K/train_filtered.json'
        train_data_list = load_GSM8K(full_path, n_train)
        
    elif train_task_name.lower() == 'math_algebra':
        full_path = f'{HOME_DIRECTORY}/dataset/MATH_ALGEBRA/train_algebra_total_filtered.json'
        train_data_list = load_MATH(full_path, 999999)
        
    elif train_task_name.lower() == 'math_geometry':
        full_path = f'{HOME_DIRECTORY}/dataset/MATH_GEOMETRY/groundtruth.json'
        train_data_list = load_MATH(full_path, n_train)
    
    elif train_task_name.lower() == 'math_intermediate_algebra':
        full_path = f'{HOME_DIRECTORY}/dataset/MATH_INTERMEDIATE_ALGEBRA/groundtruth.json'
        train_data_list = load_MATH(full_path, n_train)
    
    elif train_task_name.upper() =='ESNLI': 
        full_path = f'{HOME_DIRECTORY}/dataset/ESNLI/train.json'
        if 'gold_label' in variation_suffix:
            train_data_list = load_ESNLI(full_path, n_train, use_gold_label = True)
        else:
            train_data_list = load_ESNLI(full_path, n_train, use_gold_label = False) 

    elif train_task_name.upper() =='AQUARAT': 
        full_path = f'{HOME_DIRECTORY}/dataset/AQuaRAT/train.json'
        train_data_list = load_AQuaRAT(full_path, n_train)
    
    elif train_task_name.upper() =='WINOGRANDE':
        full_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
        train_data_list = load_WINOGRANDE(full_path, n_train, finetune = True)
        
    elif train_task_name.upper() =='PIQA': 
        full_path = f'{HOME_DIRECTORY}/dataset/PIQA/train.json'
        train_data_list = load_PIQA(full_path, n_train, finetune = True)
        
    elif train_task_name.upper() =='BOOLQ': 
        full_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
        train_data_list = load_BOOLQ(full_path, n_train, finetune = True)
        
    elif train_task_name.upper() =='SQUAD': 
        full_path = f'{HOME_DIRECTORY}/dataset/SQUAD/train.json'
        train_data_list = load_SQUAD(full_path, n_train, finetune = True)
    
    elif train_task_name.upper() =='DROP':
        full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        train_data_list = load_DROP(full_path, n_train, finetune_with_gt=True)
        
    elif train_task_name.upper() =='MMLU': 
        # train_path = f'{HOME_DIRECTORY}/dataset/MMLU/mmlu_train.json'
        full_path = f'{HOME_DIRECTORY}/dataset/MMLU/groundtruth.json'
        train_data_list = load_MMLU(full_path, n_train, finetune = True)

    elif train_task_name.upper() =='MMLU_MORAL_SCENARIOS': 
        full_path = f'{HOME_DIRECTORY}/dataset/MMLU_MORAL_SCENARIOS/groundtruth.json'
        train_data_list = load_MMLU(full_path, n_train, finetune = True)
        
    elif train_task_name.upper() =='API_BANK': 
        full_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-api-train.json'
        train_data_list = load_API_BANK(full_path, n_train)

    elif train_task_name.upper() =='AGIEVAL': 
        full_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/train.json'
        train_data_list = load_AGIEVAL(full_path, n_train, finetune = True)
        
    elif train_task_name.upper() =='ECQA': 
        full_path = f'{HOME_DIRECTORY}/dataset/ECQA/train.json'
        if 'gold_label' in variation_suffix:
            train_data_list = load_ECQA(full_path, n_train, finetune = True, use_gt_rationale = False)
        else:            
            train_data_list = load_ECQA(full_path, n_train, finetune = True, use_gt_rationale = True)

    elif train_task_name.upper() == 'HELLASWAG':
        full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(full_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_train]

        for ii, item in enumerate(train_data_list):
            aaa = item['answer']
            qqq = item['question']
            qqq += "\nPlease directly provide the final answer (1 or 2 or 3 or 4) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            train_data_list[ii]['answer'] = 'Final Answer: ' + aaa
    
    elif train_task_name.upper() == 'MMLU_PRO':
        full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(full_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_train]

        for ii, item in enumerate(train_data_list):
            aaa = item['answer']
            qqq = item['question']
            qqq += "\nPlease directly provide the final answer (A or B or C or D or E or F or G or H or I or J) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            train_data_list[ii]['answer'] = 'Final Answer: ' + aaa
    
    elif train_task_name.upper() == 'MMLU_PRO_LAW':
        full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(full_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_train]

        for ii, item in enumerate(train_data_list):
            aaa = item['answer']
            qqq = item['question']
            qqq += "\nPlease directly provide the final answer (A or B or C or D or E or F or G or H or I or J) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            train_data_list[ii]['answer'] = 'Final Answer: ' + aaa

    elif train_task_name.upper() == 'THEOREMQA':
        full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(full_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_train]

        for ii, item in enumerate(train_data_list):
            aaa = item['answer']
            qqq = item['question']
            qqq += "\nPlease directly provide the final answer at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            train_data_list[ii]['answer'] = 'Final Answer: ' + aaa
    
    elif train_task_name.upper() == 'ARC_CHALLENGE':
        full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        with open(full_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_train]

        for ii, item in enumerate(train_data_list):
            aaa = item['answer']
            qqq = item['question']
            qqq += "\nPlease directly provide the final answer (A or B or C or D) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            train_data_list[ii]['answer'] = 'Final Answer: ' + aaa
    
    elif train_task_name.upper() == 'MBPP':
        full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
        train_data_list = load_MBPP(full_path, n_train)

    data_list = train_data_list[:n_train]
    return data_list, full_path

def set_up_training_dataset(train_method, HOME_DIRECTORY, train_task_name, n_train, variation_suffix, data_container_path, model_name = ''):
    data_list, _ = load_training_dataset(train_method, HOME_DIRECTORY, train_task_name, n_train, variation_suffix, model_name = model_name)
    if variation_suffix:
        intermediate_gpt4_generated_data_train_file_name = f'{train_method}_{train_task_name}_{variation_suffix}_{model_name}_{n_train}'
    else:
        intermediate_gpt4_generated_data_train_file_name = f'{train_method}_{train_task_name}_{model_name}_{n_train}'
        
    train_data = []
    for item in data_list:
        temp = {}
        try:
            temp['instruction'] = item['question']
        except: 
            temp['instruction'] = item['instruction']
        try:
            temp['output'] = item['answer']
        except: 
            temp['output'] = item['output']
        temp['input'] = ''
        train_data.append(temp)
    intermediate_sft_file_path = f"{data_container_path}/{intermediate_gpt4_generated_data_train_file_name}.json"
    with open(intermediate_sft_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)
    time.sleep(3)
    return intermediate_sft_file_path

def load_variation_path(variation_suffix, root_path, api_type = 'gpt4', model_name = 'mistral'):
    full_path = ''
    if 'mini_gpt4' in api_type:
        if 'rewirte_groundtruth_in_own_words' in variation_suffix:
            full_path = f'{root_path}/varient/mini_gpt4_rewirte_groundtruth_in_own_words.json'
        if 'step_by_step' in variation_suffix:
            full_path = f'{root_path}/varient/mini_gpt4_step_by_step.json'
        if 'openai_human_written_examples' in variation_suffix:
            full_path = f'{root_path}/varient/mini_gpt4_human_written_examples.json'
        if 'gpt4_style_in_context_examples' in variation_suffix:
            full_path = f'{root_path}/varient/provide_mini_gpt4_example.json'
        if 'simple_response' in variation_suffix:
            full_path = f'{root_path}/varient/mini_gpt4_simple_response.json'
        if 'rewrite_in_natural_language' in variation_suffix:
            full_path = f'{root_path}/varient/mini_gpt4_in_natural_language.json'
    elif 'gpt4' in api_type:
        if 'in_own_words' in variation_suffix:
            full_path = f'{root_path}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if 'step_by_step' in variation_suffix:
            full_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
        if 'openai_human_written_examples' in variation_suffix:
            full_path = f'{root_path}/varient/openai_human_written_examples.json'
        if 'gpt4_style_in_context_examples' in variation_suffix:
            full_path = f'{root_path}/varient/openai_gpt4_provide_gpt4_example_1000.json'
        if 'simple_response' in variation_suffix:
            full_path = f'{root_path}/varient/simple_response.json'
        if 'mistral_self_generated' in variation_suffix:
            full_path = f'{root_path}/varient/mistral_self_generated.json'
        if 'qwen_self_generated' in variation_suffix:
            full_path = f'{root_path}/varient/qwen_self_generated.json'
        if 'llama_3_instruct_self_generated' in variation_suffix:
            full_path = f'{root_path}/varient/llama_3_instruct_self_generated.json'
        if 'rewrite_in_natural_language' in variation_suffix:
            full_path = f'{root_path}/varient/openai_gpt4_generated_in_natural_language_1000.json'
        if 'paraphrase' in variation_suffix:
            full_path = f'{root_path}/varient/paraphrase.json'
        if 'redundant' in variation_suffix:
            full_path = f'{root_path}/varient/redundant.json'
    elif 'claude' in api_type:
        if 'rewirte_groundtruth_in_own_words' in variation_suffix:
            full_path = f'{root_path}/varient/claude_rewirte_groundtruth_in_own_words.json'
        if 'step_by_step' in variation_suffix:
            full_path = f'{root_path}/varient/claude_step_by_step.json'
        if 'openai_human_written_examples' in variation_suffix:
            full_path = f'{root_path}/varient/claude_human_written_examples.json'
        if 'gpt4_style_in_context_examples' in variation_suffix:
            full_path = f'{root_path}/varient/provide_calude_example.json'
        if 'simple_response' in variation_suffix:
            full_path = f'{root_path}/varient/claude_simple_response.json'
        if 'rewrite_in_natural_language' in variation_suffix:
            full_path = f'{root_path}/varient/claude_in_natural_language.json'
    elif 'anthropic_thinking' in api_type:
        if 'rewirte_groundtruth_in_own_words' in variation_suffix:
            full_path = f'{root_path}/varient/claude_thinking_rewirte_groundtruth_in_own_words.json'
        if 'step_by_step' in variation_suffix:
            full_path = f'{root_path}/varient/claude_thinking_step_by_step.json'
        if 'openai_human_written_examples' in variation_suffix:
            full_path = f'{root_path}/varient/claude_thinking_human_written_examples.json'
        if 'gpt4_style_in_context_examples' in variation_suffix:
            full_path = f'{root_path}/varient/provide_calude_thinking_example.json'
        if 'simple_response' in variation_suffix:
            full_path = f'{root_path}/varient/claude_thinking_simple_response.json'
        if 'rewrite_in_natural_language' in variation_suffix:
            full_path = f'{root_path}/varient/claude_thinking_in_natural_language.json'
    return full_path

def load_training_dataset(train_method, HOME_DIRECTORY, train_task_name, n_train, variation_suffix, model_name = ''):
    root_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}'
    if train_method == 'groundtruth':
        data_list, full_path = load_groundtruth(train_task_name, n_train, variation_suffix, root_path, HOME_DIRECTORY)

    if train_method == 'minimum_change':
        if 'mistral' in model_name:
            full_path = f'{root_path}/mistral_minimum_change.json'
        if '3' in model_name and 'llama' in model_name:
            full_path = f'{root_path}/llama_3_minimum_change.json'

        with open(full_path, 'r') as file:
            minimum_change_train_data_list = json.load(file)

        data_list = minimum_change_train_data_list[:n_train]

    if train_method == 'gpt4' or train_method == 'claude' or ('mini_gpt4' in train_method) or train_method == 'anthropic_thinking':
        if 'gpt4' == train_method:
            full_path = f'{root_path}/gpt4.json'
        elif 'claude' == train_method:
            full_path = f'{root_path}/claude.json'
        elif 'mini_gpt4' in train_method:
            full_path = f'{root_path}/varient/mini_gpt4.json'
        elif 'anthropic_thinking' in train_method:
            full_path = f'{root_path}/varient/claude_thinking.json'
        
        with open(full_path, 'r') as file:
            gpt4_generated_train_data_list = json.load(file)
        data_list = gpt4_generated_train_data_list[:n_train]
        
        if 'variation' in variation_suffix and (train_method == 'gpt4' or train_method == 'claude' or ('mini_gpt4' in train_method)) or train_method == 'anthropic_thinking':
            full_path = load_variation_path(variation_suffix, root_path)

# ------------------
            # if 'claude' == train_method and 'varient/gpt4' in full_path:
            #     full_path = full_path.replace('varient/', 'varient/claude_')

            # if 'mini_gpt' == train_method and 'varient/gpt4' in full_path:
            #     full_path = full_path.replace('varient/', 'varient/mini_gpt4_')
# ------------------

            if full_path:
                with open(full_path, 'r') as file:
                    data_list = json.load(file)
                data_list = data_list[:n_train]
            elif 'ablation_on_correctness' in variation_suffix:
                full_path = ''
                if 'mini_gpt' in variation_suffix:
                    mini_full_path = f'{root_path}/varient/openai_mini_gpt4.json'
                    with open(mini_full_path, 'r') as file:
                        gpt4_generated_train_data_list = json.load(file)
                    gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train] 
                if 'ablation_on_correctness_predicting_on_correct_data_only'in variation_suffix:
                    gpt4_temp = []
                    for iiiii, item in enumerate(gpt4_generated_train_data_list):
                        temp = {}
                        answer_temp = item['answer']
                        answer_filtered = extract_after_last_occurrence(answer_temp, "Groundtruth:")
                        answer_filtered = answer_filtered.strip('\'"')
                        correct = eval_MATH_correctness(answer_filtered, train_data_list[iiiii]['numerical_final_answer'])
                        if correct:
                            gpt4_temp.append(item)
                if 'ablation_on_correctness_mix_correct_data_and_in_own_words'in variation_suffix:
                    gpt4_temp = []
                    in_own_words_full_path = f'{root_path}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
                    with open(in_own_words_full_path, 'r') as file:
                        gpt4_in_own_words_generated_train_data_list = json.load(file)
                    gpt4_in_own_words_generated_train_data_list = gpt4_in_own_words_generated_train_data_list[:n_train]
                    for iiiii, item in enumerate(gpt4_generated_train_data_list):
                        temp = {}
                        answer_temp = item['answer']
                        answer_filtered = extract_after_last_occurrence(answer_temp, "Groundtruth:")
                        answer_filtered = answer_filtered.strip('\'"')
                        correct = eval_MATH_correctness(answer_filtered, train_data_list[iiiii]['numerical_final_answer'])
                        if correct:
                            gpt4_temp.append(item)
                        else:
                            if 'in_own_words' in variation_suffix:
                                gpt4_temp.append(gpt4_in_own_words_generated_train_data_list[iiiii])
                data_list = gpt4_temp[:n_train]
            else:
                full_path = ''
                if 'api_bank_total_combine' in variation_suffix:
                    train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-api-train.json'
                    train_data_list = load_API_BANK(train_path, n_train)

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_gpt4 = gpt4_generated_train_data_list_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/claude.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_calude = json.load(file)
                    gpt4_generated_train_data_list_calude = gpt4_generated_train_data_list_calude[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/mini_gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_mini_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_mini_gpt4 = gpt4_generated_train_data_list_mini_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_step_by_step = json.load(file)
                    gpt4_generated_train_data_list_step_by_step = gpt4_generated_train_data_list_step_by_step[:n_train]
                    
                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_gpt4_provide_gpt4_example_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4_example = json.load(file)
                    gpt4_generated_train_data_list_gpt4_example = gpt4_generated_train_data_list_gpt4_example[:n_train]

                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_human_written_examples.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_human_written_example = json.load(file)
                    gpt4_generated_train_data_list_human_written_example = gpt4_generated_train_data_list_human_written_example[:n_train]


                    gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4 + gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_mini_gpt4 + gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_human_written_example

                    if 'api_bank_total_combine_good' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_gpt4 + gpt4_generated_train_data_list_mini_gpt4
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_gpt4
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_gpt4 + gpt4_generated_train_data_list_human_written_example

                    if 'api_bank_total_combine_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4[:166] + gpt4_generated_train_data_list_calude[166:333] + gpt4_generated_train_data_list_mini_gpt4[333:498] + gpt4_generated_train_data_list_step_by_step[498:665] + gpt4_generated_train_data_list_gpt4_example[665:831] + gpt4_generated_train_data_list_human_written_example[831:]
                    
                    if 'api_bank_total_combine_good_1000' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude[:333] + gpt4_generated_train_data_list_gpt4[333:666] + gpt4_generated_train_data_list_mini_gpt4[666:]
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4_example[:333] + gpt4_generated_train_data_list_calude[333:666] + gpt4_generated_train_data_list_gpt4[666:]
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude[:333] + gpt4_generated_train_data_list_gpt4[333:666] + gpt4_generated_train_data_list_human_written_example[666:]
                    if 'api_bank_total_combine_best_worst_2000' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_step_by_step
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_calude


                if 'hellaswag_total_combine' in variation_suffix:
                    train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
                    with open(train_path, 'r') as file:
                        train_data_list = json.load(file)
                    train_data_list = train_data_list[:n_train]

                    for ii, item in enumerate(train_data_list):
                        aaa = item['answer']
                        qqq = item['question']
                        qqq += "\nPlease directly provide the final answer (1 or 2 or 3 or 4) at the end, after 'Final Answer:'"
                        train_data_list[ii]['question'] = qqq
                        train_data_list[ii]['answer'] = 'Final Answer: ' + aaa

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/HELLASWAG/gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_gpt4 = gpt4_generated_train_data_list_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/HELLASWAG/claude.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_calude = json.load(file)
                    gpt4_generated_train_data_list_calude = gpt4_generated_train_data_list_calude[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/mini_gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_mini_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_mini_gpt4 = gpt4_generated_train_data_list_mini_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_step_by_step = json.load(file)
                    gpt4_generated_train_data_list_step_by_step = gpt4_generated_train_data_list_step_by_step[:n_train]
                    
                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_gpt4_provide_gpt4_example_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4_example = json.load(file)
                    gpt4_generated_train_data_list_gpt4_example = gpt4_generated_train_data_list_gpt4_example[:n_train]

                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_human_written_examples.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_human_written_example = json.load(file)
                    gpt4_generated_train_data_list_human_written_example = gpt4_generated_train_data_list_human_written_example[:n_train]


                    gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4 + gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_mini_gpt4 + gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_human_written_example

                    if 'hellaswag_total_combine_good' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_mini_gpt4 + gpt4_generated_train_data_list_human_written_example + gpt4_generated_train_data_list_gpt4_example
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_human_written_example
                    
                    if 'hellaswag_total_combine_best_worst_2000' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_gpt4_example
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example + gpt4_generated_train_data_list_gpt4
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_mini_gpt4

                    if 'hellaswag_total_combine_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4[:166] + gpt4_generated_train_data_list_calude[166:333] + gpt4_generated_train_data_list_mini_gpt4[333:498] + gpt4_generated_train_data_list_step_by_step[498:665] + gpt4_generated_train_data_list_gpt4_example[665:831] + gpt4_generated_train_data_list_human_written_example[831:]
                    
                    if 'hellaswag_total_combine_good_1000' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4_example[:333] + gpt4_generated_train_data_list_human_written_example[333:666] + gpt4_generated_train_data_list_mini_gpt4[666:]
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_step_by_step[:333] + gpt4_generated_train_data_list_gpt4_example[333:666] + gpt4_generated_train_data_list_gpt4[666:]
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_step_by_step[:333] + gpt4_generated_train_data_list_gpt4_example[333:666] + gpt4_generated_train_data_list_human_written_example[666:]

                    a = 1
                
                if 'drop_total_combine' in variation_suffix:
                    full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
                    train_data_list = load_DROP(full_path, n_train, finetune_with_gt=True)

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_gpt4 = gpt4_generated_train_data_list_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/claude.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_calude = json.load(file)
                    gpt4_generated_train_data_list_calude = gpt4_generated_train_data_list_calude[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/mini_gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_mini_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_mini_gpt4 = gpt4_generated_train_data_list_mini_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_step_by_step = json.load(file)
                    gpt4_generated_train_data_list_step_by_step = gpt4_generated_train_data_list_step_by_step[:n_train]
                    
                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_gpt4_provide_gpt4_example_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4_example = json.load(file)
                    gpt4_generated_train_data_list_gpt4_example = gpt4_generated_train_data_list_gpt4_example[:n_train]

                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_human_written_examples.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_human_written_example = json.load(file)
                    gpt4_generated_train_data_list_human_written_example = gpt4_generated_train_data_list_human_written_example[:n_train]


                    gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4 + gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_mini_gpt4 + gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_human_written_example

                    if 'drop_total_combine_good' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_human_written_example + gpt4_generated_train_data_list_gpt4_example
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_step_by_step
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_step_by_step

                    if 'drop_total_combine_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4[:166] + gpt4_generated_train_data_list_calude[166:333] + gpt4_generated_train_data_list_mini_gpt4[333:498] + gpt4_generated_train_data_list_step_by_step[498:665] + gpt4_generated_train_data_list_gpt4_example[665:831] + gpt4_generated_train_data_list_human_written_example[831:]
                    
                    if 'drop_total_combine_good_1000' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude[:333] + gpt4_generated_train_data_list_human_written_example[333:666] + gpt4_generated_train_data_list_gpt4_example[666:]
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example[:333] + gpt4_generated_train_data_list_gpt4_example[333:666] + gpt4_generated_train_data_list_step_by_step[666:]
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example[:333] + gpt4_generated_train_data_list_gpt4_example[333:666] + gpt4_generated_train_data_list_step_by_step[666:]

                    if 'drop_total_combine_best_worst_2000' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example + gpt4_generated_train_data_list_mini_gpt4
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_mini_gpt4

                    a = 1
                if 'mmlu_pro_total_combine' in variation_suffix:
                    full_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/groundtruth.json'
                    train_data_list = load_DROP(full_path, n_train, finetune_with_gt=True)

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_gpt4 = gpt4_generated_train_data_list_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/claude.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_calude = json.load(file)
                    gpt4_generated_train_data_list_calude = gpt4_generated_train_data_list_calude[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/mini_gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_mini_gpt4 = json.load(file)
                    gpt4_generated_train_data_list_mini_gpt4 = gpt4_generated_train_data_list_mini_gpt4[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_step_by_step = json.load(file)
                    gpt4_generated_train_data_list_step_by_step = gpt4_generated_train_data_list_step_by_step[:n_train]
                    
                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_gpt4_provide_gpt4_example_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_gpt4_example = json.load(file)
                    gpt4_generated_train_data_list_gpt4_example = gpt4_generated_train_data_list_gpt4_example[:n_train]

                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_human_written_examples.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list_human_written_example = json.load(file)
                    gpt4_generated_train_data_list_human_written_example = gpt4_generated_train_data_list_human_written_example[:n_train]


                    gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4 + gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_mini_gpt4 + gpt4_generated_train_data_list_step_by_step + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_human_written_example

                    if 'mmlu_pro_total_combine_good' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_mini_gpt4 + gpt4_generated_train_data_list_gpt4
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude + gpt4_generated_train_data_list_gpt4_example + gpt4_generated_train_data_list_step_by_step
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example + gpt4_generated_train_data_list_gpt4 + gpt4_generated_train_data_list_mini_gpt4

                    if 'mmlu_pro_total_combine_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list_gpt4[:166] + gpt4_generated_train_data_list_calude[166:333] + gpt4_generated_train_data_list_mini_gpt4[333:498] + gpt4_generated_train_data_list_step_by_step[498:665] + gpt4_generated_train_data_list_gpt4_example[665:831] + gpt4_generated_train_data_list_human_written_example[831:]
                    
                    if 'mmlu_pro_total_combine_good_1000' in variation_suffix:
                        if 'mistral' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude[:333] + gpt4_generated_train_data_list_mini_gpt4[333:666] + gpt4_generated_train_data_list_gpt4[666:]
                        if 'llama_3' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example[:333] + gpt4_generated_train_data_list_gpt4_example[333:666] + gpt4_generated_train_data_list_step_by_step[666:]
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_calude[:333] + gpt4_generated_train_data_list_gpt4_example[333:666] + gpt4_generated_train_data_list_step_by_step[666:]
                        if 'qwen' in model_name:
                            gpt4_generated_train_data_list = gpt4_generated_train_data_list_human_written_example[:333] + gpt4_generated_train_data_list_gpt4[333:666]  + gpt4_generated_train_data_list_mini_gpt4[666:]



                    a = 1

                
                if 'gsm8k_total_combine' in variation_suffix:
                    train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/train_filtered.json'
                    train_data_list = load_GSM8K(train_path, n_train)

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_False_999999_march_27.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list1 = json.load(file)
                    gpt4_generated_train_data_list1 = gpt4_generated_train_data_list1[:n_train]

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/anthropic_gpt4_generated_gsm8k_False_1000_r1.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list2 = json.load(file)
                    gpt4_generated_train_data_list2 = gpt4_generated_train_data_list2[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/openai_mini_gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list3 = json.load(file)
                    gpt4_generated_train_data_list3 = gpt4_generated_train_data_list3[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list4 = json.load(file)
                    gpt4_generated_train_data_list4 = gpt4_generated_train_data_list4[:n_train]
                    
                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list5 = json.load(file)
                    gpt4_generated_train_data_list5 = gpt4_generated_train_data_list5[:n_train]
                    

                    gpt4_generated_train_data_list = gpt4_generated_train_data_list1 + gpt4_generated_train_data_list2 + gpt4_generated_train_data_list3 + gpt4_generated_train_data_list4 + train_data_list + gpt4_generated_train_data_list5
                    

                    a = 1
                    if 'gsm8k_total_combine_bad' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list1 + train_data_list
                    
                    if 'gsm8k_total_combine_good' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list1 + gpt4_generated_train_data_list2 + gpt4_generated_train_data_list3

                    # if 'gsm8k_total_combine4' in variation_suffix:
                    #     gpt4_generated_train_data_list = gpt4_generated_train_data_list1 + gpt4_generated_train_data_list3
                    
                    # if 'gsm8k_total_combine5' in variation_suffix:
                    #     gpt4_generated_train_data_list = gpt4_generated_train_data_list1 + gpt4_generated_train_data_list4


                    if 'gsm8k_total_combine_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list1[:166] + gpt4_generated_train_data_list2[166:333] + gpt4_generated_train_data_list3[333:498] + gpt4_generated_train_data_list4[498:665] + train_data_list[665:831] + gpt4_generated_train_data_list5[831:]
                    
                    if 'gsm8k_total_combine_bad_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list1[:500] + train_data_list[500:]
                    
                    if 'gsm8k_total_combine_good_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list1[:333] + gpt4_generated_train_data_list2[333:666] + gpt4_generated_train_data_list3[666:]

                
                if 'plan_bench_total_combine' in variation_suffix:
                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH_GENERATION/gpt4_generated_plan_bench_False_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list1 = json.load(file)
                    gpt4_generated_train_data_list1 = gpt4_generated_train_data_list1[:n_train]

                    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH_GENERATION/anthropic_generated_plan_bench_False_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list2 = json.load(file)
                    gpt4_generated_train_data_list2 = gpt4_generated_train_data_list2[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/openai_mini_gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list3 = json.load(file)
                    gpt4_generated_train_data_list3 = gpt4_generated_train_data_list3[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list4 = json.load(file)
                    gpt4_generated_train_data_list4 = gpt4_generated_train_data_list4[:n_train]
                    
                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list5 = json.load(file)
                    gpt4_generated_train_data_list5 = gpt4_generated_train_data_list5[:n_train]

                    gpt4_generated_data_train_path =  f'{root_path}/varient/write_in_gpt4_style.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list6 = json.load(file)
                    gpt4_generated_train_data_list6 = gpt4_generated_train_data_list6[:n_train]

                    gpt4_generated_data_train_path =  f'{root_path}/varient/openai_human_written_examples.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list7 = json.load(file)
                    gpt4_generated_train_data_list7 = gpt4_generated_train_data_list7[:n_train]


                    gpt4_generated_train_data_list = gpt4_generated_train_data_list1 + gpt4_generated_train_data_list2 + gpt4_generated_train_data_list3 + gpt4_generated_train_data_list4  + gpt4_generated_train_data_list5 + gpt4_generated_train_data_list6 + gpt4_generated_train_data_list7
                    

                    if 'plan_bench_total_combine_good' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list2 + gpt4_generated_train_data_list7 + gpt4_generated_train_data_list6

                    if 'plan_bench_total_combine_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list1[:142] + gpt4_generated_train_data_list2[142:284] + gpt4_generated_train_data_list3[284:428] + gpt4_generated_train_data_list4[428:570]  + gpt4_generated_train_data_list5[570:712] + gpt4_generated_train_data_list6[712:854] + gpt4_generated_train_data_list7[854:]
                    
                    
                    if 'plan_bench_total_combine_good_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list2[:333] + gpt4_generated_train_data_list7[333:666] + gpt4_generated_train_data_list6[666:]
                
                if 'math_total_combine_total_combine' in variation_suffix:

                    train_path = f'{root_path}/train_algebra_total_filtered.json'
                    train_data_list = load_MATH(train_path, 999999, zeroshot = False)

                    gpt4_generated_data_train_path = f'{root_path}/gpt4_generated_math_algebra_False_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list1 = json.load(file)
                    gpt4_generated_train_data_list1 = gpt4_generated_train_data_list1[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/anthropic_gpt4_generated_math_algebra_False_1000_r1.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list2 = json.load(file)
                    gpt4_generated_train_data_list2 = gpt4_generated_train_data_list2[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/openai_mini_gpt4.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list3 = json.load(file)
                    gpt4_generated_train_data_list3 = gpt4_generated_train_data_list3[:n_train]

                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_step_by_step_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list4 = json.load(file)
                    gpt4_generated_train_data_list4 = gpt4_generated_train_data_list4[:n_train]
                    
                    gpt4_generated_data_train_path = f'{root_path}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
                    with open(gpt4_generated_data_train_path, 'r') as file:
                        gpt4_generated_train_data_list5 = json.load(file)
                    gpt4_generated_train_data_list5 = gpt4_generated_train_data_list5[:n_train]

                    

                    gpt4_generated_train_data_list = gpt4_generated_train_data_list1 + gpt4_generated_train_data_list2 + gpt4_generated_train_data_list3 + gpt4_generated_train_data_list4  + gpt4_generated_train_data_list5 + train_data_list
                    

                    if 'math_total_combine_good' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list3[:333] + gpt4_generated_train_data_list1[333:666] + gpt4_generated_train_data_list2[666:]

                    if 'math_total_combine_total_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list1[:166] + gpt4_generated_train_data_list2[166:333] + gpt4_generated_train_data_list3[333:498] + gpt4_generated_train_data_list4[498:665] + train_data_list[665:831] + gpt4_generated_train_data_list5[831:]
                    
                    if 'math_total_combine_total_good_1000' in variation_suffix:
                        gpt4_generated_train_data_list = gpt4_generated_train_data_list2[:333] + gpt4_generated_train_data_list7[333:666] + gpt4_generated_train_data_list6[666:]
                data_list = gpt4_generated_train_data_list

    if 'best_varient' in train_method:
        if 'mistral' in model_name:
            full_path = f'{root_path}/best_varient/mistral_best.json'
        if 'llama_3_instruct' in model_name:
            full_path = f'{root_path}/best_varient/llama_3_instruct_best.json'
        
        with open(full_path, 'r') as file:
            data_list = json.load(file)
        data_list = data_list[:n_train]
    return data_list, full_path


def load_evaluation_dataset(evaluation_type, n_data, test_task_name, train_task_name, train_method, HOME_DIRECTORY, variation_suffix):
    ECQA_validation_path = f'{HOME_DIRECTORY}/dataset/ECQA/validation.json'
    ESNLI_validation_path = f'{HOME_DIRECTORY}/dataset/ESNLI/validation.json'
    AQuaRAT_validation_path = f'{HOME_DIRECTORY}/dataset/AQuaRAT/dev.json'
    MMLU_validation_path = f'{HOME_DIRECTORY}/dataset/MMLU/validation.json'
    MMLU_MORAL_SCENARIOS_validation_path = f'{HOME_DIRECTORY}/dataset/MMLU_MORAL_SCENARIOS/validation.json' 
    MATH_intermediate_algebra_validation_path = f'{HOME_DIRECTORY}/dataset/MATH_INTERMEDIATE_ALGEBRA/validation.json'
    MATH_geometry_validation_path = f'{HOME_DIRECTORY}/dataset/MATH_GEOMETRY/validation.json'
    MBPP_validation_path = f'{HOME_DIRECTORY}/dataset/MBPP/validation.json'

    GSM8K_test_path = f'{HOME_DIRECTORY}/dataset/GSM8K/test_filtered.json'
    AQuaRAT_test_path = f'{HOME_DIRECTORY}/dataset/AQuaRAT/test.json'
    MATH_algebra_test_path = f'{HOME_DIRECTORY}/dataset/MATH_ALGEBRA/test_algebra_total_filtered.json'
    MATH_geometry_test_path = f'{HOME_DIRECTORY}/dataset/MATH_GEOMETRY/test.json'
    MATH_intermediate_algebra_test_path = f'{HOME_DIRECTORY}/dataset/MATH_INTERMEDIATE_ALGEBRA/test.json'
    API_BANK_test_path = f'{HOME_DIRECTORY}/dataset/API_BANK/test/test-data_level-3-batch-inf.json'
    ESNLI_test_path = f'{HOME_DIRECTORY}/dataset/ESNLI/test.json'
    BOOLQ_test_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/test.json'
    PIQA_test_path = f'{HOME_DIRECTORY}/dataset/PIQA/validation.json'
    WINOGRANDE_test_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/validation.json'
    MMLU_test_path = f'{HOME_DIRECTORY}/dataset/MMLU/test.json'
    MMLU_MORAL_SCENARIOS_test_path = f'{HOME_DIRECTORY}/dataset/MMLU_MORAL_SCENARIOS/test.json' 
    AGIEVAL_test_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/test.json'
    ECQA_test_path = f'{HOME_DIRECTORY}/dataset/ECQA/test.json'
    SQUAD_test_path = f'{HOME_DIRECTORY}/dataset/SQUAD/validation.json'
    DROP_validation_path = f'{HOME_DIRECTORY}/dataset/DROP/validation.json'
    DROP_test_path = f'{HOME_DIRECTORY}/dataset/DROP/test.json'
    MBPP_test_path = f'{HOME_DIRECTORY}/dataset/MBPP/test.json'

    if evaluation_type == 'validation':
        if test_task_name.lower() == 'gsm8k':
            test_data_list = load_GSM8K(GSM8K_test_path, n_data)
        
        if test_task_name.lower() == 'aquarat':
            test_data_list = load_AQuaRAT(AQuaRAT_validation_path, n_data)

        if 'math_algebra' == test_task_name.lower():
            test_data_list = load_MATH(MATH_algebra_test_path, n_data)
        if 'math_geometry' == test_task_name.lower():
            test_data_list = load_MATH(MATH_geometry_validation_path, n_data)
        if 'math_intermediate_algebra' == test_task_name.lower():
            test_data_list = load_MATH(MATH_intermediate_algebra_validation_path, n_data)

        if test_task_name.lower() == 'api_bank':
            test_data_list = load_API_BANK(API_BANK_test_path, n_data)
       
        if test_task_name.lower() == 'esnli':
            if 'gold_label' in variation_suffix:
                test_data_list = load_ESNLI(ESNLI_validation_path, n_data, use_gold_label = True) 
            else:
                test_data_list = load_ESNLI(ESNLI_validation_path, n_data, use_gold_label = False) 
            test_data_list = test_data_list[:1000]
        
        if test_task_name.lower() == 'boolq':
            if 'groundtruth' == train_method:
                test_data_list = load_BOOLQ(BOOLQ_test_path, n_data, finetune = True) 
            else:
                test_data_list = load_BOOLQ(BOOLQ_test_path, n_data) 
            test_data_list = test_data_list[:1000]
        if test_task_name.lower() == 'piqa':
            if 'groundtruth' == train_method:
                test_data_list = load_PIQA(PIQA_test_path, n_data, finetune = True) 
            else:
                test_data_list = load_PIQA(PIQA_test_path, n_data) 
            test_data_list = test_data_list[:1000]
        if test_task_name.lower() == 'winogrande':
            if 'groundtruth' == train_method:
                test_data_list = load_WINOGRANDE(WINOGRANDE_test_path, n_data, finetune = True) 
            else:
                test_data_list = load_WINOGRANDE(WINOGRANDE_test_path, n_data) 
            test_data_list = test_data_list[:1000]
        # if test_task_name.lower() == 'triviaqa':
        #     zeroshot_accuracy = triviaqa_zeroshot_accuracy
        #     if 'finetune' in train_method:
        #         test_data_list = load_TRIVIAQA(TRIVIAQA_test_path, n_data, finetune = True) 
        #     else:
        #         test_data_list = load_TRIVIAQA(TRIVIAQA_test_path, n_data) 
            # test_data_list = test_data_list[:1000]
        if test_task_name.lower() == 'mmlu':
            if 'groundtruth' == train_method:
                test_data_list = load_MMLU(MMLU_validation_path, n_data, finetune = True) 
            else:
                test_data_list = load_MMLU(MMLU_validation_path, n_data) 
            test_data_list = test_data_list[:1000]

        if test_task_name.lower() == 'mmlu_moral_scenarios':
            if 'groundtruth' == train_method:
                test_data_list = load_MMLU(MMLU_MORAL_SCENARIOS_validation_path, n_data, finetune = True) 
            else:
                test_data_list = load_MMLU(MMLU_MORAL_SCENARIOS_validation_path, n_data) 
            test_data_list = test_data_list[:1000]

        if test_task_name.lower() == 'agieval':
            if 'groundtruth' == train_method:
                test_data_list = load_AGIEVAL(AGIEVAL_test_path, n_data, finetune = True) 
            else:
                test_data_list = load_AGIEVAL(AGIEVAL_test_path, n_data) 
            test_data_list = test_data_list[:1000]
        
        if test_task_name.lower() == 'squad':
            if 'groundtruth' == train_method:
                test_data_list = load_SQUAD(SQUAD_test_path, n_data, finetune = True)
            else:
                test_data_list = load_SQUAD(SQUAD_test_path, n_data, finetune = False)       

        if train_task_name.upper() =='DROP':
            if 'groundtruth' == train_method:
                test_data_list = load_DROP(DROP_validation_path, n_data, finetune_with_gt = True)
            else:
                test_data_list = load_DROP(DROP_validation_path, n_data, finetune_with_gt = False)  

        if test_task_name.lower() == 'ecqa':
            if 'groundtruth' == train_method:
                if 'gold_label' in variation_suffix:
                    test_data_list = load_ECQA(ECQA_validation_path, n_data, finetune = True, use_gt_rationale = False)
                else:
                    test_data_list = load_ECQA(ECQA_validation_path, n_data, finetune = True, use_gt_rationale = True)
            else:
                test_data_list = load_ECQA(ECQA_validation_path, n_data) 
            test_data_list = test_data_list[:1000]
        
        if test_task_name.lower() == 'mbpp':
            test_data_list = load_MBPP(MBPP_validation_path, n_data) 
            test_data_list = test_data_list[:1000]
        

        if test_task_name.lower() == 'mmlu_pro' or test_task_name.lower() == 'arc_challenge' or test_task_name.lower() == 'hellaswag' or test_task_name.lower() == 'theoremqa' or 'plan_bench' in test_task_name.lower() or test_task_name.lower() == 'mmlu_pro_law':
            validation_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/validation.json'
            with open(validation_path, 'r') as file:
                test_data_list_total = json.load(file)
            test_data_list = test_data_list_total[:n_data]

            if 'groundtruth' == train_method:
                for kkk, test_item in enumerate(test_data_list):
                    qqq = test_item['question'] + "\nPlease answer directly and place the final answer at the end after 'Final Answer:'"
                    test_data_list[kkk]['question'] = qqq
            else:
                if 'plan_bench_verification' in test_task_name.lower():
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\n\nPlease inference first then check if the plan is valid follow by an explaination at the end after the word 'Final Answer:'"
                        test_data_list[kkk]['question'] = qqq
                elif 'plan_bench_execution' in test_task_name.lower():
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\n\nplease inference first then put the resulting state at the end after 'Final Answer:'"
                        test_data_list[kkk]['question'] = qqq
                elif 'plan_bench' in test_task_name.lower():
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\n\nPlease infer first, then place the plan at the end, after 'Final Answer:'. The plan you place after 'Final Answer:' should be written in triplet format, which contains (action, object_1, object_2). For example, (unstack red blue) means that you unstack the red object from the blue object."
                        test_data_list[kkk]['question'] = qqq
                else:
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\nPlease inference first, then place the final answer at the end after 'Final Answer:'"
                        test_data_list[kkk]['question'] = qqq

            
    if evaluation_type == 'test':
        if n_data > 1000 or n_data == 1000:
            n_data_1000 = 1000 
        elif n_data < 1000:
            n_data_1000 = n_data
        if test_task_name.lower() == 'gsm8k':
            test_data_list = load_GSM8K(GSM8K_test_path, n_data)

        if test_task_name.lower() == 'aquarat':
            test_data_list = load_AQuaRAT(AQuaRAT_test_path, n_data)

        if  test_task_name.lower() == 'math_algebra':
            test_data_list = load_MATH(MATH_algebra_test_path, n_data)
        if  test_task_name.lower() == 'math_geometry':
            test_data_list = load_MATH(MATH_geometry_test_path, n_data)
        if  test_task_name.lower() == 'math_intermediate_algebra':
            test_data_list = load_MATH(MATH_intermediate_algebra_test_path, n_data)
            
        if test_task_name.lower() == 'api_bank':
            test_data_list = load_API_BANK(API_BANK_test_path, n_data)
        if test_task_name.lower() == 'esnli':
            if train_task_name.lower() == 'esnli':
                if 'gold_label' in variation_suffix:
                    test_data_list = load_ESNLI(ESNLI_test_path, n_data_1000, use_gold_label = True)  
                else:
                    test_data_list = load_ESNLI(ESNLI_test_path, n_data_1000, use_gold_label = False)        
            else:
                test_data_list = load_ESNLI(ESNLI_test_path, n_data_1000, use_gold_label = False) 
        if test_task_name.lower() == 'boolq':
            if train_task_name.lower() == 'boolq':
                if 'groundtruth' == train_method:
                    test_data_list = load_BOOLQ(BOOLQ_test_path, n_data_1000, finetune = True)  
                else:
                    test_data_list = load_BOOLQ(BOOLQ_test_path, n_data_1000)        
            else:
                test_data_list = load_BOOLQ(BOOLQ_test_path, n_data_1000) 
        if test_task_name.lower() == 'piqa':
            if train_task_name.lower() == 'piqa':
                if 'groundtruth' == train_method:
                    test_data_list = load_PIQA(PIQA_test_path, n_data_1000, finetune = True)  
                else:
                    test_data_list = load_PIQA(PIQA_test_path, n_data_1000)        
            else:
                test_data_list = load_PIQA(PIQA_test_path, n_data_1000) 
        if test_task_name.lower() == 'winogrande':
            if train_task_name.lower() == 'winogrande':
                if 'groundtruth' == train_method:
                    test_data_list = load_WINOGRANDE(WINOGRANDE_test_path, n_data_1000, finetune = True)  
                else:
                    test_data_list = load_WINOGRANDE(WINOGRANDE_test_path, n_data_1000)        
            else:
                test_data_list = load_WINOGRANDE(WINOGRANDE_test_path, n_data_1000) 
        # if test_task_name.lower() == 'triviaqa':
        #     if train_task_name.lower() == 'triviaqa':
        #         if 'finetune' in train_method:
        #             test_data_list = load_TRIVIAQA(TRIVIAQA_test_path, n_data, finetune = True)  
        #         else:
        #             test_data_list = load_TRIVIAQA(TRIVIAQA_test_path, n_data)        
        #     else:
        #         test_data_list = load_TRIVIAQA(TRIVIAQA_test_path, n_data) 
        if test_task_name.lower() == 'mmlu':
            if train_task_name.lower() == 'mmlu':
                if 'groundtruth' == train_method:
                    test_data_list = load_MMLU(MMLU_test_path, n_data_1000, finetune = True)  
                else:
                    test_data_list = load_MMLU(MMLU_test_path, n_data_1000)        
            else:
                test_data_list = load_MMLU(MMLU_test_path, n_data_1000) 

        if test_task_name.lower() == 'mmlu_moral_scenarios':
            if train_task_name.lower() == 'mmlu_moral_scenarios':
                if 'groundtruth' == train_method:
                    test_data_list = load_MMLU(MMLU_MORAL_SCENARIOS_test_path, n_data_1000, finetune = True)  
                else:
                    test_data_list = load_MMLU(MMLU_MORAL_SCENARIOS_test_path, n_data_1000)        
            else:
                test_data_list = load_MMLU(MMLU_MORAL_SCENARIOS_test_path, n_data_1000) 

        if test_task_name.lower() == 'agieval':
            if train_task_name.lower() == 'agieval':
                if 'groundtruth' == train_method:
                    test_data_list = load_AGIEVAL(AGIEVAL_test_path, n_data_1000, finetune = True)  
                else:
                    test_data_list = load_AGIEVAL(AGIEVAL_test_path, n_data_1000)        
            else:
                test_data_list = load_AGIEVAL(AGIEVAL_test_path, n_data_1000) 
        if test_task_name.lower() == 'ecqa':
            if train_task_name.lower() == 'ecqa':
                if 'groundtruth' == train_method:
                    if 'gold_label' in variation_suffix:
                        test_data_list = load_ECQA(ECQA_test_path, n_data_1000, finetune = True, use_gt_rationale = False)
                    else:
                        test_data_list = load_ECQA(ECQA_test_path, n_data_1000, finetune = True, use_gt_rationale = True)
                else:
                    test_data_list = load_ECQA(ECQA_test_path, n_data_1000)        
            else:
                test_data_list = load_ECQA(ECQA_test_path, n_data_1000) 
        
        if test_task_name.lower() == 'squad':
            if train_task_name.lower() == 'squad':
                if 'groundtruth' == train_method:
                    test_data_list = load_SQUAD(SQUAD_test_path, n_data_1000, finetune = True)
                else:
                    test_data_list = load_SQUAD(SQUAD_test_path, n_data_1000, finetune = False)        
            else:
                test_data_list = load_SQUAD(SQUAD_test_path, n_data_1000, finetune = False)
        
        if test_task_name.lower() == 'drop':
            if train_task_name.lower() == 'drop':
                if 'groundtruth' == train_method:
                    test_data_list = load_DROP(DROP_test_path, n_data_1000, finetune_with_gt = True)
                else:
                    test_data_list = load_DROP(DROP_test_path, n_data_1000, finetune_with_gt = False)        
            else:
                test_data_list = load_DROP(DROP_test_path, n_data_1000, finetune_with_gt = False)
        
        if test_task_name.lower() == 'mbpp':
            test_data_list = load_MBPP(MBPP_test_path, n_data_1000) 
        
        if test_task_name.lower() == 'mmlu_pro' or test_task_name.lower() == 'arc_challenge' or test_task_name.lower() == 'hellaswag' or test_task_name.lower() == 'theoremqa' or 'plan_bench' in test_task_name.lower() or test_task_name.lower() == 'mmlu_pro_law':
            validation_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/test.json'
            with open(validation_path, 'r') as file:
                test_data_list_total = json.load(file)
            test_data_list = test_data_list_total[:n_data]

            if 'groundtruth' == train_method:
                for kkk, test_item in enumerate(test_data_list):
                    qqq = test_item['question'] + "\nPlease answer directly and place the final answer at the end after 'Final Answer:'"
                    test_data_list[kkk]['question'] = qqq
            else:
                if 'plan_bench_verification' in test_task_name.lower():
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\n\nPlease inference first then check if the plan is valid follow by an explaination at the end after the word 'Final Answer:'"
                        test_data_list[kkk]['question'] = qqq

                elif 'plan_bench_execution' in test_task_name.lower():
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\n\nplease inference first then put the resulting state at the end after 'Final Answer:'"
                        test_data_list[kkk]['question'] = qqq

                elif 'plan_bench' in test_task_name.lower():
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\n\nPlease infer first, then place the plan at the end, after 'Final Answer:'. The plan you place after 'Final Answer:' should be written in triplet format, which contains (action, object_1, object_2). For example, (unstack red blue) means that you unstack the red object from the blue object."
                        test_data_list[kkk]['question'] = qqq
                else:
                    for kkk, test_item in enumerate(test_data_list):
                        qqq = test_item['question'] + "\nPlease inference first, then place the final answer at the end after 'Final Answer:'"
                        test_data_list[kkk]['question'] = qqq

        test_data_list = test_data_list[:1000]
    return test_data_list


def load_gold_label_and_question_list(task_name, n_data_creation, provide_groundtruth_with_inference_steps = False):
    if task_name == 'plan_bench_generation':
        train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH_GENERATION/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])
    
    if 'plan_bench' in task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            qq = item['question']
            qq += "\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"
            question_list.append(qq)
            gold_label_list.append(item['gold_label'])
    
    if 'plan_bench_excution' in task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            qq = item['question']
            qq += "\n\nplease inference first then put the resulting state at the end after 'Final Answer:'"
            question_list.append(qq)
            gold_label_list.append(item['gold_label'])
    
    if 'plan_bench_verification' in task_name.lower():
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            qq = item['question']
            qq += "\nPlease inference first then check if the plan is valid follow by an explaination at the end after the word 'Final Answer:'"
            question_list.append(qq)
            gold_label_list.append(item['gold_label'])

    if 'math_algebra' in task_name:
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/train_algebra_total_filtered.json'
        train_data_list = load_MATH(train_path, n_data_creation, zeroshot = False)
        
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['numerical_final_answer'])

    if task_name == 'math_intermediate_algebra':
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_INTERMEDIATE_ALGEBRA/groundtruth.json'
        train_data_list = load_MATH(train_path, n_data_creation, zeroshot = False)
        
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['numerical_final_answer'])

    if task_name == 'math_geometry':
        train_path = f'{HOME_DIRECTORY}/dataset/MATH_GEOMETRY/groundtruth.json'
        train_data_list = load_MATH(train_path, n_data_creation, zeroshot = False)
        
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['numerical_final_answer'])

    elif task_name == 'esnli':
        train_path = f'{HOME_DIRECTORY}/dataset/ESNLI/train.json'
        train_data_list = load_ESNLI(train_path, n_data_creation)
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])

    elif task_name == 'boolq':
        train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
        train_data_list = load_BOOLQ(train_path, n_data_creation)

        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])

    elif task_name == 'piqa':
        train_path = f'{HOME_DIRECTORY}/dataset/PIQA/train.json'
        train_data_list = load_PIQA(train_path, n_data_creation)
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])

    elif task_name == 'winogrande':
        train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
        train_data_list = load_WINOGRANDE(train_path, n_data_creation)
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])

    elif task_name == 'mmlu':
        train_path = f'{HOME_DIRECTORY}/dataset/MMLU/groundtruth.json'
        train_data_list = load_MMLU(train_path, n_data_creation)
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])

    elif task_name == 'mmlu_moral_scenarios':
        train_path = f'{HOME_DIRECTORY}/dataset/MMLU_MORAL_SCENARIOS/groundtruth.json'
        train_data_list = load_MMLU(train_path, n_data_creation)
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])

    elif task_name == 'agieval':
        train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/train.json'
        train_data_list = load_AGIEVAL(train_path, n_data_creation)
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])
        
    elif 'gsm8k' in task_name:
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/train_filtered.json'
        train_data_list = load_GSM8K(train_path, n_data_creation)

        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['numerical_final_answer'])

    elif 'aquarat' in task_name:
        train_path = f'{HOME_DIRECTORY}/dataset/AQuaRAT/train.json'
        train_data_list = load_AQuaRAT(train_path, n_data_creation)

        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])

    elif task_name.upper() =='API_BANK':
        train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-api-train.json'
        train_data_list = load_API_BANK(train_path, n_data_creation)
        question_list = []
        gold_label_list = []
        for item in train_data_list:
            qq = item['question']
            qq += "\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"
            question_list.append(qq)
            gold_label_list.append(item['gold_label'])

    elif task_name.upper() =='ECQA':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/train.json'
        train_data_list = load_ECQA(train_path, n_data_creation)
        train_data_list = train_data_list[:]

        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])     

    elif task_name.upper() =='SQUAD':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/train.json'
        train_data_list = load_SQUAD(train_path, n_data_creation, finetune=False)
        train_data_list = train_data_list[:]

        question_list = []
        gold_label_list = []
        for item in train_data_list:
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])    

    elif task_name.upper() =='DROP':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        train_data_list = load_DROP(train_path, n_data_creation, finetune_with_gt=False)

        question_list = []
        gold_label_list = []
        for ii, item in enumerate(train_data_list):
            question_list.append(item['question'])
            gold_label_list.append(item['gold_label'])    

    elif task_name.upper() == 'HELLASWAG':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]

        question_list = []
        gold_label_list = []
        for ii, item in enumerate(train_data_list):
            qqq = item['question']
            qqq += "\nPlease inference first, then provide the final answer (1/2/3/4) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            question_list.append(qqq)
            gold_label_list.append(item['gold_label'])  
    
    elif task_name.upper() == 'MMLU_PRO':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]

        question_list = []
        gold_label_list = []
        for ii, item in enumerate(train_data_list):
            qqq = item['question']
            qqq += "\nPlease inference first, then provide the final answer (A/B/C/D/E/F/G/H/I/J) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            question_list.append(qqq)
            gold_label_list.append(item['gold_label'])  
    
    elif task_name.upper() == 'MMLU_PRO_LAW':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]

        question_list = []
        gold_label_list = []
        for ii, item in enumerate(train_data_list):
            qqq = item['question']
            qqq += "\nPlease inference first, then provide the final answer (A/B/C/D/E/F/G/H/I/J) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            question_list.append(qqq)
            gold_label_list.append(item['gold_label'])  
    
    elif task_name.upper() == 'THEOREMQA':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]

        question_list = []
        gold_label_list = []
        for ii, item in enumerate(train_data_list):
            qqq = item['question']
            qqq += "\nPlease inference first, then provide the final answer at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            question_list.append(qqq)
            gold_label_list.append(item['gold_label'])  
    
    elif task_name.upper() == 'ARC_CHALLENGE':
        train_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        with open(train_path, 'r') as file:
            train_data_list = json.load(file)
        train_data_list = train_data_list[:n_data_creation]

        question_list = []
        gold_label_list = []
        for ii, item in enumerate(train_data_list):
            qqq = item['question']
            qqq += "\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'"
            train_data_list[ii]['question'] = qqq
            question_list.append(qqq)
            gold_label_list.append(item['gold_label'])  
    
    elif task_name.upper() == 'MBPP':
        full_path = f'{HOME_DIRECTORY}/dataset/{task_name.upper()}/groundtruth.json'
        train_data_list = load_MBPP(full_path, n_data_creation)

        question_list = []
        gold_label_list = []
        for ii, item in enumerate(train_data_list):
            question_list.append(item['question'])
            gold_label_list.append(item['answer'])  

    groundtruth_with_inference_steps_list = []
    if provide_groundtruth_with_inference_steps:
        for item in train_data_list:
            answer_temp = item['answer']
            groundtruth_with_inference_steps_list.append(answer_temp)
        gold_label_list = groundtruth_with_inference_steps_list

    return gold_label_list, train_data_list, question_list




def add_gold_label(test_task_name, item, groundtruth_item):
    item['gold_label'] = groundtruth_item
    return item