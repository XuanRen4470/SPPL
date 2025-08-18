import sys
import os
import json
import random
import re
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from config.config import GPT_API, MODEL_ENGINE, MINI_MODEL_ENGINE, CLAUDE_MODEL_ENGINE
from utils.meta_prompt_template.template import provide_gpt4_or_human_example_template, provide_self_generated_example_template, provide_self_generated_example_template_plan_bench_or_api_bank
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from config.config import HOME_DIRECTORY


def create_answer_directly_response(question_list, api_type, gold_label_list = [], step_by_step = False, task_name = '', train_data_list = [], provide_groundtruth_with_inference_steps = False,  answer_without_groundtruth = False, temperature = 0.7, groundtruth_list = []):    
    if 'gpt4' in api_type or 'mini' in api_type:
        model_company = 'openai'
    if 'claude' in api_type or 'anthropic' in api_type: 
        model_company = 'anthropic'
    
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "I am a helpful assistant"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(
            api_key=f"{my_api_key}",
        )
        if 'anthropic' in api_type:
            @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
            def create_gpt_completion(qa_, model_engine, client, temperature):
                message = client.messages.create(
                    model=model_engine,
                    max_tokens=6000,
                    temperature=temperature,
                    messages=[
                            {"role": "user", "content": f"{qa_}"}
                    ]
                )
                answer = message.content[0].text
                return answer
        else:
            @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
            def create_gpt_completion(qa_, model_engine, client, temperature):
                message = client.messages.create(
                    model=model_engine,
                    max_tokens=2000,
                    temperature=temperature,
                    messages=[
                            {"role": "user", "content": f"{qa_}"}
                    ]
                )
                answer = message.content[0].text
                return answer

    gpt4_answer_list = []
    if step_by_step:
        step_by_step_insertion_1 = ' step by step'
        step_by_step_insertion_2 = 'Step by Step '
    else:
        step_by_step_insertion_1 = ''
        step_by_step_insertion_2 = ''
    
    for i, question in enumerate(question_list):
        gold_label = gold_label_list[i]
        temp = {}
        temp['question'] = question
        temp['input'] = ''
        # AQuaRAT
        if task_name == 'MATH_ALGEBRA' or task_name == 'MATH_GEOMETRY' or task_name == 'MATH_INTERMEDIATE_ALGEBRA' or 'GSM8K' in task_name or 'AQUARAT' in task_name:
            if 'GSM8K' in task_name and 'GSM8K' != task_name:
                prompt = question
            elif 'AQUARAT' in task_name:
                prompt = f"""We have the question {question} 


1. We wish you to answer the question{step_by_step_insertion_1}.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A)/(B)/(C)/(D)/(E) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: ({gold_label})""" 
            else:
                if not provide_groundtruth_with_inference_steps:
                    prompt = f"""We have the {question} 


1. We wish you to answer the question{step_by_step_insertion_1}.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (NUMBER_HERE) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: NUMBER_HERE""" 
                else:
                    prompt = f"""We have the 
question: {question} 
and the groundtruth: {groundtruth_list[i]['answer']}


1. Based on the groundtruth, we wish you to answer the question{step_by_step_insertion_1} in more detailed way.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (NUMBER_HERE) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: NUMBER_HERE""" 
        elif task_name == 'ANLI':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: (Entailment/Contradiction/Neutral) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'ESNLI':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (Entailment/Contradiction/Neutral) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'SCITAIL':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (Entailment/Neutral) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'BOOLQ':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (True/False) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'PIQA':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (a number) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'WINOGRANDE':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (a number) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'MMLU':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'MMLU_MORAL_SCENARIOS':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'AGIEVAL':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 

        elif task_name == 'API_BANK':
            
#             prompt = f"""We have the {question} and the groundtruth {gold_label}


# 1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

# {step_by_step_insertion_2}INFERENCE HERE
# Final Answer: {gold_label}""" 


            example = \
"""
The following four examples help you to understand the question. 

When receiving the questin, the gold label is the action for generating the next api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nGenerate next API Request: ",
"gold_label": "API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]",

after generating the last api-call, we receiving the next question. based on this question with the given api-call history, the gold label is referring to the next api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nGenerate next API Request: ",
"gold_label": "API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]",

after generating the last api-call, we receiving the next question. based on this question with the given api-call history, the gold label is referring to the next api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]->{'appointments': ['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']}\nGenerate next API Request: ",
"gold_label": "API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]",

after generating the last api-call, we receiving the next question. based on this question with the given api-call history, the gold label is referring to the fianl api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]->{'appointments': ['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']}\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]->{'name': 'HealthcareProviderAppointmentScheduler', 'description': 'API for scheduling appointments with healthcare providers.', 'input_parameters': {'appointment_datetime': {'type': 'datetime', 'description': 'The datetime for the appointment.'}, 'healthcare_provider': {'type': 'str', 'description': 'The name of the healthcare provider.'}}, 'output_parameters': {'confirmation_number': {'type': 'str', 'description': 'The confirmation number for the appointment.'}}}\nGenerate next API Request: ",
"gold_label": "API-Request: [HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist')]",


ok, we have show you the following four example to help you understanding the question.   now please help me to do the following

"""
            prompt = \
f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            if step_by_step_insertion_1 != '':
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You will solve the problem in step by step manner.
3. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
4. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            prompt = example + prompt
            temp['question'] = question
            temp['input'] = ''
        elif task_name == 'API_BANK_VANILLA':
            question = question + f"""

The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be generate for the next API-Request? The question might already provide the previous API-CALL history."""
            prompt = f"""We have the {question} and the groundtruth {gold_label}

1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif 'PLAN_BENCH' in task_name:
            prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            if step_by_step_insertion_1 != '':
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You will solve the problem in step by step manner.
3. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
4. You will inference first then put the Final Answer: ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        elif task_name == 'ECQA':
            # temp = train_data_list[i]
            prompt = f"""We have the {question} and the groundtruth {gold_label}

1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        
        elif task_name == 'SQUAD':
            # temp = train_data_list[i]
            prompt = f"""We have the question "{question}" and the groundtruth {gold_label}

1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        
        elif task_name == 'DROP':
            prompt = f"""We have the question "{question}" and the groundtruth {gold_label}

1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        
        elif task_name == 'MMLU_PRO':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D/E/F/G/H/I/J) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
        elif task_name == 'MMLU_PRO_LAW':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D/E/F/G/H/I/J) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        
        elif task_name == 'HELLASWAG':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (1/2/3/4) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 

        elif task_name == 'ARC_CHALLENGE':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        
        elif task_name == 'MBPP':
            prompt = question
            if step_by_step_insertion_1 != '':
                prompt += '\n\n Your code should be in step by step style.'
            

            if provide_groundtruth_with_inference_steps:
                prompt = \
f"""we wish to train our model so we need better groundtruth
                
We have the 
question: {prompt} 
and the groundtruth: {groundtruth_list[i]['answer']}


Based on the groundtruth, we wish you to answer the question{step_by_step_insertion_1} in better way.

Please create the better answer directly without saying anything else""" 

        if answer_without_groundtruth:
            # original_question_item = train_data_list[i]['original_question']
            # prompt = original_question_item
            # temp['question'] = original_question_item

            prompt = question
            temp['question'] = question
            temp['input'] = ''
        
        if 'mini_gpt' in api_type:
            answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
        elif 'gpt4' in api_type:
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        elif 'claude' in api_type:
            answer = create_gpt_completion(prompt, CLAUDE_MODEL_ENGINE, client, temperature = temperature)
        
        if 'python' in answer:
            if 'mini_gpt' in api_type:
                answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
            elif 'gpt4' in api_type:
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            elif 'claude' in api_type:
                answer = create_gpt_completion(prompt, CLAUDE_MODEL_ENGINE, client, temperature = temperature)

        temp['answer'] = answer
        temp['gold_label'] = gold_label
        gpt4_answer_list.append(temp)

        print('-----------------------------------')
        print('groundtruth: ' + groundtruth_list[i]['answer'])
        print()
        print('answer: ' + answer)
        a = 1
    return gpt4_answer_list


def create_different_questions_for_simple_structure(question_list, variation_num, gold_label_list = [], model_company = 'openai', enable_mini_gpt4 = False, prompt_start_over_num = 5):
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client, conversation_history):
            response = client.chat.completions.create(
                model=model_engine,
                temperature=1,
                messages=conversation_history
                # messages=[
                #     {"role": "system", "content": "provide answer"},
                #     {"role": "user", "content": f"{qa_}"}
                # ]
            )
            answer = response.choices[0].message.content
            if len(answer) > 5000:
                response = client.chat.completions.create(
                model=model_engine,
                    temperature=1,
                    messages=conversation_history
                )
                answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(
            api_key=f"{my_api_key}",
        )

        def create_gpt_completion(qa_, model_engine, client, conversation_history):
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=1,
                # messages=[
                #         {"role": "user", "content": f"{qa_}"}
                # ]
                messages=conversation_history
            )
            answer = message.content[0].text
            return answer

    prompt_list = []
    if model_company == 'openai':   
        conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    else:
        conversation_history = []
    for i in range(variation_num):
        prompt = ''
        # prompt_list = []
        iii = i % prompt_start_over_num
        if i < len(gold_label_list):
        # iii = i % start_over_num
        # if i < iii:
            index = i
        else:
            index = i % len(gold_label_list)
            # index = i % start_over_num
        gold_label = gold_label_list[index]
        question = question_list[index]
        if iii == 0:
            prompt +=\
f"""
Example Question: "{question}" 
Example gold label: "{gold_label}"

The question and gold above just show you an example. In the real time, the question and the prompt you are given will be different.(very important. remember it)

We believe that the the target response has simple logic structure, then the training result is better. 

Your goal is to generate a prompt, so i can use it to guide gpt-4 to generate new groundtruth which sounds natural and has simple logic structure. We will place the prompt directly after the question and the gold label so that it can guide GPT-4.

please give me the prompt directly without saying things like "here is your promt".
"""
        else:
            prompt_replacement =\
f"""
Example Question: "{question}" 
Example gold label: "{gold_label}"

The question and gold above just show you an example. In the real time, the question and the prompt you are given will be different.(very important. remember it)

We believe that the the target response has simple logic structure, then the training result is better. 

Your goal is to generate a prompt, so i can use it to guide gpt-4 to generate new groundtruth which sounds natural and has simple logic structure. We will place the prompt directly after the question and the gold label so that it can guide GPT-4.

please give me the prompt directly without saying things like "here is your promt".
"""
            if model_company == 'openai':
                conversation_history[1] = {"role": "system", "content": prompt_replacement}
            else:
                conversation_history[0] = {"role": "user", "content": prompt_replacement}
            prompt +=\
"""I am trying to collect different prompt. please generate a(only one) different prompt which will guide GPT4 to generate the answer in different style. please give me the prompt directly without saying things like "here is your promt".
"""

        conversation_history.append({"role": "user", "content": prompt})
        if not enable_mini_gpt4:
            generated_prompt = create_gpt_completion(prompt, MODEL_ENGINE, client, conversation_history)
        else:
            generated_prompt = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client, conversation_history)
        # print(generated_prompt)
        prompt_list.append(generated_prompt)
    return prompt_list


# def create_different_questions(question_list, variation_num, api_type, initial_prediction_list, gold_label_list, prompt_start_over_num = 5, temperature = 0.7):    
#     if 'gpt4' in api_type or 'mini' in api_type:
#         model_company = 'openai'
#     if 'claude' in api_type: 
#         model_company = 'anthropic'    

#     if model_company == 'openai':
#         from openai import OpenAI
#         import openai
#         from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
#         client = OpenAI(api_key=GPT_API)
#         @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
#         def create_gpt_completion(qa_, model_engine, client, conversation_history):
#             response = client.chat.completions.create(
#                 model=model_engine,
#                 temperature=1,
#                 messages=conversation_history
#             )
#             answer = response.choices[0].message.content
#             if len(answer) > 5000:
#                 response = client.chat.completions.create(
#                 model=model_engine,
#                     temperature=1,
#                     messages=conversation_history
#                 )
#                 answer = response.choices[0].message.content
#             return answer
#     elif model_company == 'anthropic':
#         import anthropic
#         my_api_key = os.environ.get("ANTHROPIC_API_KEY")
#         client = anthropic.Anthropic(
#             api_key=f"{my_api_key}",
#         )

#         def create_gpt_completion(qa_, model_engine, client, conversation_history):
#             message = client.messages.create(
#                 model="claude-3-5-sonnet-20240620",
#                 max_tokens=2000,
#                 temperature=1,
#                 # messages=[
#                 #         {"role": "user", "content": f"{qa_}"}
#                 # ]
#                 messages=conversation_history
#             )
#             answer = message.content[0].text
#             return answer

#     prompt_list = []
#     if model_company == 'openai':   
#         conversation_history = [
#         {"role": "system", "content": "You are a helpful assistant."}
#     ]
#     else:
#         conversation_history = []
    
#     initial_prediction_1 = initial_prediction_list[0]
#     initial_prediction_2 = initial_prediction_list[1]
#     for i in range(variation_num):
#         prompt = ''
#         # prompt_list = []
#         iii = i % prompt_start_over_num
#         if i < len(gold_label_list):
#         # iii = i % start_over_num
#         # if i < iii:
#             index = i
#         else:
#             index = i % len(gold_label_list)
#             # index = i % start_over_num
#         gold_label = gold_label_list[index]
#         question = question_list[index]
#         if iii == 0:
#             prompt +=\
# f"""
# Example Question: "{question}" 
# Example gold label: "{gold_label}"

# The question and gold label above just show you an example. In the real time, the question and the prompt you are given will be different.(very important. remember it)

# Your goal is to generate a prompt, so i can use it to guide gpt-4 to generate new groundtruth with inference rationales and sounds natural. We will place the prompt directly after the question so that it can guide GPT-4 to generate groudntruth that is better than gold_label for trianing.

# please give me the prompt directly without saying things like "here is your promt".
# """
#         else:
#             prompt_replacement =\
# f"""
# Example Question: "{question}" 
# Example gold label: "{gold_label}"

# The question and gold label above just show you an example. In the real time, the question and the prompt you are given will be different.(very important. remember it)

# Your goal is to generate a prompt, so i can use it to guide gpt-4 to generate new groundtruth with inference rationales and sounds natural. We will place the prompt directly after the question so that it can guide GPT-4 to generate groudntruth that is better than gold_label for trianing.

# please give me the prompt directly without saying things like "here is your promt".
# """
#             if model_company == 'openai':
#                 conversation_history[1] = {"role": "system", "content": prompt_replacement}
#             else:
#                 conversation_history[0] = {"role": "user", "content": prompt_replacement}
#             prompt +=\
# """I am trying to collect different prompt. please generate a(only one) different prompt which will guide GPT4 to generate the answer in different style. please give me the prompt directly without saying things like "here is your promt".
# """

#         conversation_history.append({"role": "user", "content": prompt})

#         if 'mini_gpt' in api_type:
#             generated_prompt = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client, conversation_history)
#         elif 'gpt4' in api_type:
#             generated_prompt = create_gpt_completion(prompt, MODEL_ENGINE, client, conversation_history)
#         elif 'claude' in api_type:
#             generated_prompt = create_gpt_completion(prompt, CLAUDE_MODEL_ENGINE, client, conversation_history, temperature = temperature)
#         prompt_list.append(generated_prompt)
#     return prompt_list


def create_different_questions_given_initial_prediction(question_list, variation_num, api_type, initial_prediction_list, gold_label_list, prompt_start_over_num = 5, temperature = 0.7):    
    if 'gpt4' in api_type or 'mini' in api_type:
        model_company = 'openai'
    if 'claude' in api_type: 
        model_company = 'anthropic'    

    if model_company == 'openai':
        from openai import OpenAI
        import openai
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client, conversation_history):
            response = client.chat.completions.create(
                model=model_engine,
                temperature=1,
                messages=conversation_history
            )
            answer = response.choices[0].message.content
            if len(answer) > 5000:
                response = client.chat.completions.create(
                model=model_engine,
                    temperature=1,
                    messages=conversation_history
                )
                answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(
            api_key=f"{my_api_key}",
        )

        def create_gpt_completion(qa_, model_engine, client, conversation_history):
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2000,
                temperature=1,
                # messages=[
                #         {"role": "user", "content": f"{qa_}"}
                # ]
                messages=conversation_history
            )
            answer = message.content[0].text
            return answer

    prompt_list = []
    if model_company == 'openai':   
        conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    else:
        conversation_history = []
    
    
    for i in range(variation_num):
        iii = i % prompt_start_over_num
        index = i % len(gold_label_list)
        gold_label1 = gold_label_list[index]
        question1 = question_list[index]
        initial_prediction_1 = initial_prediction_list[index]
        if index == len(gold_label_list) - 1:
            index_ = 0
        else:
            index_ = index + 1
        gold_label2 = gold_label_list[index_]
        question2 = question_list[index_]
        initial_prediction_2 = initial_prediction_list[index_]
        prompt =\
f"""
Example 1: 
- Question: "{question1}\nA SECRET PROMPT HERE"
- Groundtruth: "{gold_label1}"
- Initial Prediction: "{initial_prediction_1}"

Example 2: 
- Question: "{question2}\nA SECRET PROMPT HERE"
- Groundtruth: "{gold_label2}"
- Initial Prediction: "{initial_prediction_2}"

Task: I will give you two examples. Each example contains a Question, a Groundtruth, and an Initial Prediction. A secret prompt is added after each Question. The secret prompt instructs a language model to convert the Groundtruth into a style similar to the Initial Prediction. Note: The Initial Prediction is not available at test time. 

Your task: Based on the provided Questions and Groundtruths, create a secret prompt that will convert any Groundtruth into a style similar to the Initial Prediction (even though the Initial Prediction is not directly available during the test). 

Output only the secret prompt without saying things like "here is your promt".
"""
        
        if iii == 0:
            conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]
        else:
            if model_company == 'openai':
                conversation_history[1] = {"role": "system", "content": prompt}
            else:
                conversation_history[0] = {"role": "user", "content": prompt}
            prompt =\
"""Your guess is wrong. plese generate a new secrete prompt.

please give me the secrete prompt directly without saying things like "here is your promt".
"""

        conversation_history.append({"role": "user", "content": prompt})

        if 'mini_gpt' in api_type:
            generated_prompt = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client, conversation_history)
        elif 'gpt4' in api_type:
            generated_prompt = create_gpt_completion(prompt, MODEL_ENGINE, client, conversation_history)
        elif 'claude' in api_type:
            generated_prompt = create_gpt_completion(prompt, CLAUDE_MODEL_ENGINE, client, conversation_history, temperature = temperature)
        conversation_history.append({"role": "assistant", "content": generated_prompt})

        prompt_list.append(generated_prompt)
    return prompt_list
# for iiiii in prompt_list:
#     print()
#     print('-----------------------')
#     print(iiiii)

def generate_different_answer(question_list, prompt_end, api_type, groundtruth_list, gold_label_list, temperature = 0.7):
    import time
    if 'gpt4' in api_type or 'mini' in api_type:
        model_company = 'openai'
    if 'claude' in api_type: 
        model_company = 'anthropic'  

    if model_company == 'openai':
        from openai import OpenAI
        import openai
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "provide answer"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
        
    elif model_company == 'anthropic':
        import anthropic
        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(
            api_key=f"{my_api_key}",
        )

        def create_gpt_completion(qa_, model_engine, client, retries=4, delay=5):
            for attempt in range(retries):
                try:
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=2000,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": f"{qa_}"}
                        ]
                    )
                    answer = message.content[0].text
                    return answer
                except:
                    a = 1
                    # break
                time.sleep(delay)

    answer_list = []
    for i, question in enumerate(question_list):
        gold_label = gold_label_list[i]
        groundtruth = groundtruth_list[i]
        prompt =\
f"""
Question: "{question} {prompt_end}" 
groundtruth: "{groundtruth}"

Please make sure the gold label is placed at the end of your answer like this 
Final Answer: {gold_label}
"""
        
        if 'mini_gpt' in api_type:
            answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
        elif 'gpt4' in api_type:
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        elif 'claude' in api_type:
            answer = create_gpt_completion(prompt, CLAUDE_MODEL_ENGINE, client)

        answer_list.append(answer)
    return answer_list

def formating_answer(answer, gold_label, task_name):
    if 'final answer' not in answer.lower():
        answer += f'\n\nFinal Answer: {gold_label}'
    return answer


def create_response_varient(question_list, variation_suffix, api_type, task_name, gold_label_list, train_data_list, temperature = 0.7, original_question_list = [], original_gold_label_list = [], gpt4_prediction_list = []):
    gpt4_prediction_list_temp = []
    for item in gpt4_prediction_list:
        gpt4_prediction_list_temp.append(item['answer'])
    gpt4_prediction_list = gpt4_prediction_list_temp
    varient_name = variation_suffix.replace('variation_', '')
    if 'gpt4' in api_type or 'mini' in api_type:
        model_company = 'openai'
    if 'claude' in api_type: 
        model_company = 'anthropic'
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "provide answer"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        import os
        import time
        import requests

        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=f"{my_api_key}")
        def create_gpt_completion(qa_, model_engine, client, retries=4, delay=5, temperature = temperature):
            for attempt in range(retries):
                try:
                    message = client.messages.create(
                        model=model_engine,
                        max_tokens=2000,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": f"{qa_}"}
                        ]
                    )
                    answer = message.content[0].text
                    return answer
                except:
                    a = 1
                    # break
                time.sleep(delay)
            
            return None
            
    data_list = []
    for i, question in enumerate(question_list):
        gold_label = gold_label_list[i]
        groundtruth = train_data_list[i]['answer']

        temp = {}
        if varient_name == 'gpt4_style_in_context_examples' or varient_name == 'openai_human_written_examples':
            prompt = provide_gpt4_or_human_example_template(task_name, varient_name, gold_label, question)
        elif 'self_generated' in varient_name:
            if 'mistral' in varient_name:
                model_name = 'mistral'
            elif 'llama_3' in varient_name:
                model_name = 'llama_3_instruct'
            elif 'qwen' in varient_name:
                model_name = 'qwen'
            
            if 'plan' in task_name.lower() or 'api_bank' in task_name.lower():
                file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record_good_examples/{task_name.lower()}_{model_name}_initial_prediction_2.json'
                with open(file_path_temp, 'r') as f:
                    data_list_ = json.load(f)
                
                initial_prediction = data_list_['initial_prediction']
                a1 = initial_prediction[0]
                a2 = initial_prediction[1]
                template_holder = {}
                template_holder['answer1'] = a1
                template_holder['answer2'] = a2
            else:
                file_path_temp = f'{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/initial_prediction_record/{task_name.lower()}_{model_name}_initial_prediction_use_correct_initial_prediction_500.json'

                with open(file_path_temp, 'r') as f:
                    data_list_ = json.load(f)

                initial_prediction = data_list_['initial_prediction']
                correct_index = data_list_['correct_index']

                # Initialize variables
                q1, q2 = None, None
                a1, a2 = None, None
                g1, g2 = None, None

                # Ensure gold_label1 and gold_label2 are different
                for ii in range(1, len(correct_index)):
                    index1, index2 = correct_index[0], correct_index[ii]
                    q1, q2 = original_question_list[index1], original_question_list[index2]
                    a1, a2 = initial_prediction[0], initial_prediction[ii]
                    g1, g2 = original_gold_label_list[index1], original_gold_label_list[index2]

                    a1 = formating_answer(a1, g1, task_name)
                    a2 = formating_answer(a2, g2, task_name)
                    if g1 != g2:
                        break

                template_holder = {}
                template_holder['answer1'] = a1
                template_holder['answer2'] = a2
                template_holder['gold_label1'] = g1
                template_holder['gold_label2'] = g2
                template_holder['question1'] = q1
                template_holder['question2'] = q2
            if 'plan' in task_name.lower() or 'api_bank' in task_name.lower():
                prompt = provide_self_generated_example_template_plan_bench_or_api_bank(gold_label, question, template_holder)
            else:
                prompt = provide_self_generated_example_template(gold_label, question, template_holder)
            
        elif varient_name == 'step_by_step':
            prompt = f"""We have the question and the groundtruth. Please reformat the groundtruth in step by step manner with details.

Question: {question}
Groundtruth: {groundtruth}



1. We wish you to regenerate a new groundtruth. The new groundtruth solve the problem step by step. If you believe the groundtruth is not detail enough, you could add details.
2. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
3. (important format) You must generate the groundtruth with the step by step inference process directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) You will inference first then put the Final Answer: {gold_label}

at the end like this

INFERENCE HERE
Final Answer: {gold_label}"""
        
        elif varient_name == 'rewirte_groundtruth_in_own_words':
            prompt = f"""Given the question: {question}
and the groundtruth: {groundtruth}

Please states the prediction in your own words. The groundtruth is 100% correct. You should not change the problem solving logic of the groundtruth. just restates it in your own words.

1. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
2. (important format) You must generate the groundtruth directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'"""
            if 'mbpp' not in task_name.lower():
                prompt += f"""\n3. (important format) Please make sure the Final Answer: {gold_label} is placed at the end of the modified prediction."""

        elif varient_name == 'rewrite_in_natural_language':
            prompt = f"""Given the question: {question}
and the groundtruth: {groundtruth}

Please directly reformat the groundtruth in natural language words. The groundtruth is 100% correct.

1. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
2. (important format) You must reformat the groundtruth directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'"""

        elif varient_name == 'add_details':
            prompt = f"""We have the question and the groundtruth, but sometimes the groundtruth is not detailed enough. Please add more details to the groundtruth to make it better if you believe there needs to be more details.
Question: {question}
Groundtruth: {groundtruth}


1. (important format) We wish you to generate the detailed groundtruth directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
2. You will pretend as you do not know the groundtruth. We will to use your prediction as target labels to train our model.
3. (important format) You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}"""

        elif varient_name == 'redundant':
            prompt = f"""We have the question and the groundtruth. Given on the groundtruth, please reformat the groundtruth so that it answer the question in a step by step redundant manner. Be as repetitive and step by step and redundant as possible.


Question: {question}
Groundtruth: {groundtruth}


1. We wish you to reformat a new groundtruth. The new groundtruth are reformated a new groundtruth which solve the problem as steo by step and redundant as possible.
2. You will pretend as you do not know the groundtruth, because we will use your step by step redundant answer as target responses to train our model.
3. (important format) You must generate the groundtruth with the step by step redundant inference process directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) You will inference first then put the Final Answer: {gold_label}

at the end like this

INFERENCE HERE
Final Answer: {gold_label}
"""
        
        elif varient_name == 'simple_response':
            if 'plan_bench' not in task_name.lower() and 'api_bank' not in task_name.lower():
                prompt = \
f"""We have the question and the groundtruth. Our goal is to creating the most effective training dataset for training smaller model. There are 3 rules for how to construct the best training dataset for smaller model.
1. The dataset should only contains the necessary inference process which do not contains the too many explainations. The only skills we wish the smaller model to learn is how to inference with natural language. The additional chat-bot like explaination is not helpful for improving the inference of smaller model on reasoning tasks.
2. The inference process should construct like natural language. We notice that when the groundtruth is like natural language, smaller model can learn better.
3. You have to output the modified groundtruth directly such that it does not contain any additional words such as 'sure, I can help you with this' or 'here is your groundtruth'.


Question: {question}
Groundtruth: {groundtruth}


Please output the better groundtruth directly according to the 3 rules that I told you before. 
The format is like this.

INFERENCE HERE
Final Answer: {gold_label}"""
            else:
                prompt = \
f"""We have the question and the groundtruth. Our goal is to creating the most effective training dataset for training smaller model. There are 3 rules for how to construct the best training dataset for smaller model.
1. The dataset should only contains the necessary inference process which do not contains the too many explainations. The only skills we wish the smaller model to learn is how to inference with natural language. The additional chat-bot like explaination is not helpful for improving the inference of smaller model on reasoning tasks.
2. The inference process should construct like natural language. We notice that when the groundtruth is like natural language, smaller model can learn better.
3. You have to output the modified groundtruth directly such that it does not contain any additional words such as 'sure, I can help you with this' or 'here is your groundtruth'.


Question: {question}
Groundtruth: {groundtruth}


Please output the better groundtruth directly according to the 3 rules that I told you before. 
The better groundtruth should constructed such that it inference first, then provide the final answer at the end.
The format is like this.

INFERENCE HERE
Final Answer: {gold_label}"""

        elif varient_name == 'paraphrase':
            gpt4_prediction = gpt4_prediction_list[i]
            prompt = f"""Given the question: {question}
and the groundtruth: {gpt4_prediction}

Could you please paraphrase the groundtruth? You have to ensure the paraphrased groundtruth follow the exactly the same logic but different language style.

1. You will pretend as you do not know the groundtruth, because we will use your paraphrased prediction as target labels to train our model.
2. (important format) You must generate the paraphrased groundtruth directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the groundruth'"""
        


        if 'api_bank' in task_name.lower():
            if varient_name == 'step_by_step' or varient_name == 'simple_response':
                example = \
"""
The following four examples help you to understand the question. 

When receiving the questin, the gold label is the action for generating the next api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nGenerate next API Request: ",
"gold_label": "API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]",

after generating the last api-call, we receiving the next question. based on this question with the given api-call history, the gold label is referring to the next api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nGenerate next API Request: ",
"gold_label": "API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]",

after generating the last api-call, we receiving the next question. based on this question with the given api-call history, the gold label is referring to the next api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]->{'appointments': ['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']}\nGenerate next API Request: ",
"gold_label": "API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]",

after generating the last api-call, we receiving the next question. based on this question with the given api-call history, the gold label is referring to the fianl api-call
"question": "\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]->{'appointments': ['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']}\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]->{'name': 'HealthcareProviderAppointmentScheduler', 'description': 'API for scheduling appointments with healthcare providers.', 'input_parameters': {'appointment_datetime': {'type': 'datetime', 'description': 'The datetime for the appointment.'}, 'healthcare_provider': {'type': 'str', 'description': 'The name of the healthcare provider.'}}, 'output_parameters': {'confirmation_number': {'type': 'str', 'description': 'The confirmation number for the appointment.'}}}\nGenerate next API Request: ",
"gold_label": "API-Request: [HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist')]",


ok, we have show you the following four example to help you understanding the question.   now please help me to do the following

"""
                prompt = example + prompt

        if 'mini_gpt' in api_type:
            answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
        elif 'gpt4' in api_type:
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        elif 'claude' in api_type:
            answer = create_gpt_completion(prompt, CLAUDE_MODEL_ENGINE, client, temperature = temperature)
        # print('---------------------------------gpt4_self_generated_answer------------------------------------------')
        # print('answer: ', answer)
        # print()
        # print('///////////////////////////////gold_label////////////////////////////////////////////')
        # print()
        # print('gold_label: ', gold_label)
        if varient_name == 'rewrite_in_natural_language':
            answer += f"""

Final Answer: {gold_label}"""

        temp['question'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['gold_label'] = gold_label
        if 'rewrite' in varient_name:
            temp['groundtruth'] = groundtruth
        data_list.append(temp)
        # print('-----------------------------------------------------')
        # print(answer)
        # print()
        # break
        a = 1
    return data_list


def clean_initial_prediction(initial_prediction_list, question_list, api_type, task_name, temperature = 0.7):
    
    if 'gpt4' in api_type or 'mini' in api_type:
        model_company = 'openai'
    if 'claude' in api_type: 
        model_company = 'anthropic'
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "provide answer"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        import os
        import time
        import requests

        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=f"{my_api_key}")
        def create_gpt_completion(qa_, model_engine, client, retries=4, delay=5, temperature = temperature):
            for attempt in range(retries):
                try:
                    message = client.messages.create(
                        model=model_engine,
                        max_tokens=2000,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": f"{qa_}"}
                        ]
                    )
                    answer = message.content[0].text
                    return answer
                except:
                    a = 1
                    # break
                time.sleep(delay)
            
            return None
        
    modified_initial_prediction_list = []
    for i, question in enumerate(question_list):
        initial_prediction = initial_prediction_list[i]
        prompt = \
f"""We have an initial prediction, but for some reason, we need to filter out the words that is not relevant to inferencing. We wish to only extract the part that is relevant to inference. Sometimes the initial prediction has some logic errors, but it is ok. no need to correct it. we only need you to filter out the part that is not related to inferencing. Please give me the modified initial prediction without any explaination. no need to change any words. 

This is the initial prediction that you need to modify: "{initial_prediction}"




Please directly give me the modified initial prediction"""
# f"""We have the question and the initial prediction. We wish to only extract the part that the prediction is used to inference from question to the answer. Please filter out the part that is unrelevant to the inferencing such as explaination and only left the part that is relevant to the inferencing. Sometimes the initial prediction is wrong, but it is ok. no need to correct it. we only need you to filter out the part that is not related to inferencing. Please give me the modified initial prediction without any explaination. no need to change any words. you just need to filter out the part that is unrelavant to inferencing.

# Question: {question}
# initial_prediction: "{initial_prediction}"

# please modify the initial prediction above so that it only inlude the inferencing part with no explaination.
# """
        print('-------------------Initial Prediction--------------------------')
        print('Initial Prediction: ' + initial_prediction)
        if 'mini_gpt' in api_type:
            answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
        elif 'gpt4' in api_type:
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        elif 'claude' in api_type:
            answer = create_gpt_completion(prompt, CLAUDE_MODEL_ENGINE, client, temperature = temperature)
        print('--------------------Modified Answer-------------------------')
        print('Modified Answer: ' + answer)
        modified_initial_prediction_list.append(answer)
    return modified_initial_prediction_list

def paraphrase_data(minimum_change_list, task_name = ''):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    gpt4_answer_list = []
    for i, item in enumerate(minimum_change_list):
        prediction = item['answer']
        question = item['question']
        temp = {}
        temp['input'] = ''
        if task_name != 'CODE':
            prompt = f"""We have the prediction: {prediction}


1. please paraphrase it without changing the order of sentence struecture.
2. do not change anymeaning even if the text is logically wrong. just paraphrase it
3. You must answer the question directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'"""
        else:
            prompt = f"""We got a code problem and we have the prediction: {prediction}


1. We wish to get a paraphrase the prediction. please paraphrase it without changing the order of sentence struecture.
2. Please not change the code. you can only change the sentences.
3. do not change any meaning even if the text is logically wrong. just paraphrase it
4. You must answer the question directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
5. You still need to keep the FINAL ANSWER at the end.
6. Remember this is a code problem. if you change the code, the code will fail."""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp['question'] = question
        temp['input'] = ''
        temp['answer'] = answer
        if task_name == 'CODE':
            temp = item
            temp['answer'] == answer
        gpt4_answer_list.append(temp)
    return gpt4_answer_list
