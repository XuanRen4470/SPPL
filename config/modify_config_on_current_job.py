from config.config import train_config, test_config


original_per_device_train_batch_size = train_config['per_device_train_batch_size']
original_gradient_accumulation_steps = train_config['gradient_accumulation_steps']

def set_config(current_task_name, device_num, seed_num, model_name = '', data_n_train = 1000000, load_in_8bit = False):

    multiplier = 1
    if 'mistral' in model_name:
        multiplier = 2
   
    gsm8k_max_input_length = 512
    gsm8k_max_output_length = 1024
    gsm8k_max_length = gsm8k_max_input_length + gsm8k_max_output_length
    gsm8k_per_device_eval_batch_size = 2 * multiplier

    math_max_input_length = 512
    math_max_output_length = 1024
    math_max_length = math_max_input_length + math_max_output_length
    math_per_device_eval_batch_size = 2 * multiplier

    esnli_max_input_length = 512
    esnli_max_output_length = 1024
    esnli_max_length = esnli_max_input_length + esnli_max_output_length
    esnli_per_device_eval_batch_size = 2 * multiplier

    ecqa_max_input_length = 512
    ecqa_max_output_length = 1024
    ecqa_max_length = ecqa_max_input_length + ecqa_max_output_length
    ecqa_per_device_eval_batch_size = 2 * multiplier

    api_bank_max_input_length = 1536
    api_bank_max_output_length = 1024
    api_bank_max_length = api_bank_max_input_length + api_bank_max_output_length
    api_bank_per_device_eval_batch_size = 2 * multiplier

    aquarat_max_input_length = 512
    aquarat_max_output_length = 1024
    aquarat_max_length = aquarat_max_input_length + aquarat_max_output_length
    aquarat_per_device_eval_batch_size = 2 * multiplier

    math_geo_max_input_length = 768
    math_geo_max_output_length = 1536
    math_geo_max_length = math_geo_max_input_length + math_geo_max_output_length
    math_geo_per_device_eval_batch_size = 2 * multiplier

    anli_max_input_length = 512
    anli_max_output_length = 1024
    anli_max_length = anli_max_input_length + anli_max_output_length
    anli_per_device_eval_batch_size = 3 * multiplier

    mnli_max_input_length = 512
    mnli_max_output_length = 1024
    mnli_max_length = mnli_max_input_length + mnli_max_output_length
    mnli_per_device_eval_batch_size = 3 * multiplier

    scitail_max_input_length = 512
    scitail_max_output_length = 1024
    scitail_max_length = scitail_max_input_length + scitail_max_output_length
    scitail_per_device_eval_batch_size = 3 * multiplier

    scitail_max_input_length = 512
    scitail_max_output_length = 1024
    scitail_max_length = scitail_max_input_length + scitail_max_output_length
    scitail_per_device_eval_batch_size = 3 * multiplier

    piqa_max_input_length = 512
    piqa_max_output_length = 1024
    piqa_max_length = piqa_max_input_length + piqa_max_output_length
    piqa_per_device_eval_batch_size = 3 * multiplier

    winogrande_max_input_length = 512
    winogrande_max_output_length = 1024
    winogrande_max_length = winogrande_max_input_length + winogrande_max_output_length
    winogrande_per_device_eval_batch_size = 3 * multiplier

    boolq_max_input_length = 512
    boolq_max_output_length = 1024
    boolq_max_length = boolq_max_input_length + boolq_max_output_length
    boolq_per_device_eval_batch_size = 2 * multiplier

    trivia_max_input_length = 1024
    trivia_max_output_length = 1024
    trivia_max_length = trivia_max_input_length + trivia_max_output_length
    trivia_per_device_eval_batch_size = 3 * multiplier

    squad_max_input_length = 512
    squad_max_output_length = 1024
    squad_max_length = squad_max_input_length + squad_max_output_length
    squad_per_device_eval_batch_size = 3 * multiplier

    mmlu_max_input_length = 512
    mmlu_max_output_length = 1024
    mmlu_max_length = mmlu_max_input_length + mmlu_max_output_length
    mmlu_per_device_eval_batch_size = 2 * multiplier

    agieval_max_input_length = 512
    agieval_max_output_length = 1024
    agieval_max_length = agieval_max_input_length + agieval_max_output_length
    agieval_per_device_eval_batch_size = 3 * multiplier

    agieval_sat_max_input_length = 1280
    agieval_sat_max_output_length = 1024
    agieval_sat_max_length = agieval_sat_max_input_length + agieval_sat_max_output_length
    agieval_sat_per_device_eval_batch_size = 2 * multiplier

    plan_bench_max_input_length = 2024
    plan_bench_max_output_length = 1024
    plan_bench_max_length = plan_bench_max_input_length + plan_bench_max_output_length
    plan_bench_per_device_eval_batch_size = 2 * multiplier


    mmlu_pro_max_input_length = 512
    mmlu_pro_max_output_length = 1024
    mmlu_pro_max_length = mmlu_pro_max_input_length + mmlu_pro_max_output_length
    mmlu_pro_per_device_eval_batch_size = 2 * multiplier

    arc_challenge_max_input_length = 512
    arc_challenge_max_output_length = 1024
    arc_challenge_max_length = arc_challenge_max_input_length + arc_challenge_max_output_length
    arc_challenge_per_device_eval_batch_size = 2 * multiplier

    hellaswag_max_input_length = 512
    hellaswag_max_output_length = 1024
    hellaswag_max_length = hellaswag_max_input_length + hellaswag_max_output_length
    hellaswag_per_device_eval_batch_size = 2 * multiplier

    theoremqa_max_input_length = 512
    theoremqa_max_output_length = 1024
    theoremqa_max_length = theoremqa_max_input_length + theoremqa_max_output_length
    theoremqa_per_device_eval_batch_size = 2 * multiplier


    drop_max_input_length = 512
    drop_max_output_length = 1024
    drop_max_length = drop_max_input_length + drop_max_output_length
    drop_per_device_eval_batch_size = 3 * multiplier


    mbpp_max_input_length = 512
    mbpp_max_output_length = 512
    mbpp_max_length = mbpp_max_input_length + mbpp_max_output_length
    mbpp_per_device_eval_batch_size = 4 * multiplier

    train_config['device_num'] = device_num
    test_config['device_num'] = device_num
    train_config['seed_num'] = seed_num
    test_config['seed_num'] = seed_num

    if 'API_BANK' in current_task_name.upper():
        # test_config['max_length'] = api_bank_max_length
        test_config['max_new_tokens'] = api_bank_max_output_length
        test_config['max_input_length'] = api_bank_max_input_length
        test_config['per_device_eval_batch_size'] = api_bank_per_device_eval_batch_size

        train_config['max_length'] = api_bank_max_length
        train_config['per_device_eval_batch_size'] = api_bank_per_device_eval_batch_size
    if 'MBPP' in current_task_name.upper():
        test_config['max_new_tokens'] = mbpp_max_output_length
        test_config['max_input_length'] = mbpp_max_input_length
        test_config['per_device_eval_batch_size'] = mbpp_per_device_eval_batch_size

        train_config['max_length'] = mbpp_max_length
        train_config['per_device_eval_batch_size'] = mbpp_per_device_eval_batch_size
    if 'PLAN_BENCH' in current_task_name.upper():
        test_config['max_new_tokens'] = plan_bench_max_output_length
        test_config['max_input_length'] = plan_bench_max_input_length
        test_config['per_device_eval_batch_size'] = plan_bench_per_device_eval_batch_size
        test_config['max_length'] = plan_bench_max_length

        train_config['max_length'] = plan_bench_max_length
        train_config['per_device_eval_batch_size'] = plan_bench_per_device_eval_batch_size
    elif 'GSM8K' in current_task_name.upper():
        # test_config['max_length'] = gsm8k_max_length
        test_config['max_new_tokens'] = gsm8k_max_output_length
        test_config['max_input_length'] = gsm8k_max_input_length
        test_config['per_device_eval_batch_size'] = gsm8k_per_device_eval_batch_size

        train_config['max_length'] = gsm8k_max_length
        train_config['per_device_eval_batch_size'] = gsm8k_per_device_eval_batch_size
    elif 'AQUARAT' in current_task_name.upper():
        test_config['max_new_tokens'] = aquarat_max_output_length
        test_config['max_input_length'] = aquarat_max_input_length
        test_config['per_device_eval_batch_size'] = aquarat_per_device_eval_batch_size

        train_config['max_length'] = aquarat_max_length
        train_config['per_device_eval_batch_size'] = aquarat_per_device_eval_batch_size
    elif 'MATH_GEOMETRY' in current_task_name.upper():
        # test_config['max_length'] = math_max_length
        test_config['max_new_tokens'] = math_geo_max_output_length
        test_config['max_input_length'] = math_geo_max_input_length
        test_config['per_device_eval_batch_size'] = math_geo_per_device_eval_batch_size

        train_config['max_length'] = math_geo_max_length
        train_config['per_device_eval_batch_size'] = math_geo_per_device_eval_batch_size
    elif 'MATH' in current_task_name.upper():
        # test_config['max_length'] = math_max_length
        test_config['max_new_tokens'] = math_max_output_length
        test_config['max_input_length'] = math_max_input_length
        test_config['per_device_eval_batch_size'] = math_per_device_eval_batch_size

        train_config['max_length'] = math_max_length
        train_config['per_device_eval_batch_size'] = math_per_device_eval_batch_size
    elif 'ANLI' in current_task_name.upper():
        # test_config['max_length'] = anli_max_length
        test_config['max_new_tokens'] = anli_max_output_length
        test_config['max_input_length'] = anli_max_input_length
        test_config['per_device_eval_batch_size'] = anli_per_device_eval_batch_size

        train_config['max_length'] = anli_max_length
        train_config['per_device_eval_batch_size'] = anli_per_device_eval_batch_size
    elif 'MNLI' in current_task_name.upper():
        test_config['max_new_tokens'] = mnli_max_output_length
        test_config['max_input_length'] = mnli_max_input_length
        test_config['per_device_eval_batch_size'] = mnli_per_device_eval_batch_size

        train_config['max_length'] = mnli_max_length
        train_config['per_device_eval_batch_size'] = mnli_per_device_eval_batch_size
    elif 'ESNLI' in current_task_name.upper():
        test_config['max_new_tokens'] = esnli_max_output_length
        test_config['max_input_length'] = esnli_max_input_length
        test_config['per_device_eval_batch_size'] = esnli_per_device_eval_batch_size

        train_config['max_length'] = esnli_max_length
        train_config['per_device_eval_batch_size'] = esnli_per_device_eval_batch_size
    elif 'SCITAIL' in current_task_name.upper():
        test_config['max_new_tokens'] = scitail_max_output_length
        test_config['max_input_length'] = scitail_max_input_length
        test_config['per_device_eval_batch_size'] = scitail_per_device_eval_batch_size

        train_config['max_length'] = scitail_max_length
        train_config['per_device_eval_batch_size'] = scitail_per_device_eval_batch_size
    elif 'PIQA' in current_task_name.upper():
        test_config['max_new_tokens'] = piqa_max_output_length
        test_config['max_input_length'] = piqa_max_input_length
        test_config['per_device_eval_batch_size'] = piqa_per_device_eval_batch_size

        train_config['max_length'] = piqa_max_length
        train_config['per_device_eval_batch_size'] = piqa_per_device_eval_batch_size
    elif 'BOOLQ' in current_task_name.upper():
        test_config['max_new_tokens'] = boolq_max_output_length
        test_config['max_input_length'] = boolq_max_input_length
        test_config['per_device_eval_batch_size'] = boolq_per_device_eval_batch_size

        train_config['max_length'] = boolq_max_length
        train_config['per_device_eval_batch_size'] = boolq_per_device_eval_batch_size
    elif 'WINOGRANDE' in current_task_name.upper():
        test_config['max_new_tokens'] = winogrande_max_output_length
        test_config['max_input_length'] = winogrande_max_input_length
        test_config['per_device_eval_batch_size'] = winogrande_per_device_eval_batch_size

        train_config['max_length'] = winogrande_max_length
        train_config['per_device_eval_batch_size'] = winogrande_per_device_eval_batch_size
    
    elif 'TRIVIAQA' in current_task_name.upper():
        test_config['max_new_tokens'] = trivia_max_output_length
        test_config['max_input_length'] = trivia_max_input_length
        test_config['per_device_eval_batch_size'] = trivia_per_device_eval_batch_size

        train_config['max_length'] = trivia_max_length
        train_config['per_device_eval_batch_size'] = trivia_per_device_eval_batch_size

    elif 'SQUAD' in current_task_name.upper():
        test_config['max_new_tokens'] = squad_max_output_length
        test_config['max_input_length'] = squad_max_input_length
        test_config['per_device_eval_batch_size'] = squad_per_device_eval_batch_size

        train_config['max_length'] = squad_max_length
        train_config['per_device_eval_batch_size'] = squad_per_device_eval_batch_size
    
    elif 'DROP' in current_task_name.upper():
        test_config['max_new_tokens'] = drop_max_output_length
        test_config['max_input_length'] = drop_max_input_length
        test_config['per_device_eval_batch_size'] = drop_per_device_eval_batch_size

        train_config['max_length'] = drop_max_length
        train_config['per_device_eval_batch_size'] = drop_per_device_eval_batch_size

    elif 'MMLU' in current_task_name.upper():
        test_config['max_new_tokens'] = mmlu_max_output_length
        test_config['max_input_length'] = mmlu_max_input_length
        test_config['per_device_eval_batch_size'] = mmlu_per_device_eval_batch_size

        train_config['max_length'] = mmlu_max_length
        train_config['per_device_eval_batch_size'] = mmlu_per_device_eval_batch_size

    elif 'AGIEVAL' in current_task_name.upper():
        test_config['max_new_tokens'] = agieval_max_output_length
        test_config['max_input_length'] = agieval_max_input_length
        test_config['per_device_eval_batch_size'] = agieval_per_device_eval_batch_size

        train_config['max_length'] = agieval_max_length
        train_config['per_device_eval_batch_size'] = agieval_per_device_eval_batch_size

    elif 'AGIEVAL_SAT' in current_task_name.upper():
        test_config['max_new_tokens'] = agieval_sat_max_output_length
        test_config['max_input_length'] = agieval_sat_max_input_length
        test_config['per_device_eval_batch_size'] = agieval_sat_per_device_eval_batch_size

        train_config['max_length'] = agieval_sat_max_length
        train_config['per_device_eval_batch_size'] = agieval_sat_per_device_eval_batch_size

    elif 'ECQA' in current_task_name.upper():
        test_config['max_new_tokens'] = ecqa_max_output_length
        test_config['max_input_length'] = ecqa_max_input_length
        test_config['per_device_eval_batch_size'] = ecqa_per_device_eval_batch_size

        train_config['max_length'] = ecqa_max_length
        train_config['per_device_eval_batch_size'] = ecqa_per_device_eval_batch_size
    

    elif 'MMLU_PRO' in current_task_name.upper():
        test_config['max_new_tokens'] = mmlu_pro_max_output_length
        test_config['max_input_length'] = mmlu_pro_max_input_length
        test_config['per_device_eval_batch_size'] = mmlu_pro_per_device_eval_batch_size

        train_config['max_length'] = mmlu_pro_max_length
        train_config['per_device_eval_batch_size'] = mmlu_pro_per_device_eval_batch_size
    
    elif 'ARC_CHALLENGE' in current_task_name.upper():
        test_config['max_new_tokens'] = arc_challenge_max_output_length
        test_config['max_input_length'] = arc_challenge_max_input_length
        test_config['per_device_eval_batch_size'] = arc_challenge_per_device_eval_batch_size

        train_config['max_length'] = arc_challenge_max_length
        train_config['per_device_eval_batch_size'] = arc_challenge_per_device_eval_batch_size

    elif 'HELLASWAG' in current_task_name.upper():
        test_config['max_new_tokens'] = hellaswag_max_output_length
        test_config['max_input_length'] = hellaswag_max_input_length
        test_config['per_device_eval_batch_size'] = hellaswag_per_device_eval_batch_size

        train_config['max_length'] = hellaswag_max_length
        train_config['per_device_eval_batch_size'] = hellaswag_per_device_eval_batch_size

    elif 'THEOREMQA' in current_task_name.upper():
        test_config['max_new_tokens'] = theoremqa_max_output_length
        test_config['max_input_length'] = theoremqa_max_input_length
        test_config['per_device_eval_batch_size'] = theoremqa_per_device_eval_batch_size

        train_config['max_length'] = theoremqa_max_length
        train_config['per_device_eval_batch_size'] = theoremqa_per_device_eval_batch_size

    if 'mistral' in model_name:
        train_config['model_name'] = 'Mistral-7b-Instruct-v2'
        # per_device_train_batch_size = train_config['per_device_train_batch_size']
        # gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        per_device_train_batch_size_temp = original_per_device_train_batch_size
        gradient_accumulation_steps_temp = original_gradient_accumulation_steps
        per_device_train_batch_size_temp *= 2
        gradient_accumulation_steps_temp /= 2
        train_config['per_device_train_batch_size'] = int(per_device_train_batch_size_temp)
        train_config['gradient_accumulation_steps'] = int(gradient_accumulation_steps_temp)
        

        if 'AGIEVAL' in current_task_name.upper() or 'MMLU' == current_task_name.upper() or 'CODE' in current_task_name.upper() or 'MBPP' in current_task_name.upper() or data_n_train < 301:
            train_config['per_device_train_batch_size'] = int(5)
            train_config['gradient_accumulation_steps'] = int(2)
        if current_task_name.upper() == 'BOOLQ':
            train_config['per_device_train_batch_size'] = int(4)
            train_config['gradient_accumulation_steps'] = int(8)

        train_config['template'] = 'mistral'

        test_config['model_name'] = 'Mistral-7b-Instruct-v2'
        test_config['template'] = 'mistral'
    
    if 'llama_3_instruct' in model_name:
        train_config['model_name'] = 'Meta-Llama-3-8B-Instruct'
        # per_device_train_batch_size = train_config['per_device_train_batch_size']
        # gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        per_device_train_batch_size_temp = original_per_device_train_batch_size
        gradient_accumulation_steps_temp = original_gradient_accumulation_steps
        per_device_train_batch_size_temp *= 2
        gradient_accumulation_steps_temp /= 2
        train_config['per_device_train_batch_size'] = int(per_device_train_batch_size_temp)
        train_config['gradient_accumulation_steps'] = int(gradient_accumulation_steps_temp)
        

        if 'AGIEVAL' in current_task_name.upper() or 'MMLU' == current_task_name.upper() or 'CODE' in current_task_name.upper() or 'MBPP' in current_task_name.upper() or data_n_train < 301:
            train_config['per_device_train_batch_size'] = int(5)
            train_config['gradient_accumulation_steps'] = int(2)

        train_config['template'] = 'llama3'
        test_config['model_name'] = 'Meta-Llama-3-8B-Instruct'
        test_config['template'] = 'llama3'

    if 'qwen' in model_name:
        train_config['model_name'] = 'Qwen2.5-7B-Instruct'
        # per_device_train_batch_size = train_config['per_device_train_batch_size']
        # gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        per_device_train_batch_size_temp = original_per_device_train_batch_size
        gradient_accumulation_steps_temp = original_gradient_accumulation_steps
        per_device_train_batch_size_temp *= 2
        gradient_accumulation_steps_temp /= 2
        train_config['per_device_train_batch_size'] = int(per_device_train_batch_size_temp)
        train_config['gradient_accumulation_steps'] = int(gradient_accumulation_steps_temp)
        

        if 'AGIEVAL' in current_task_name.upper() or 'MMLU' == current_task_name.upper() or 'CODE' in current_task_name.upper() or 'MBPP' in current_task_name.upper() or data_n_train < 301:
            train_config['per_device_train_batch_size'] = int(5)
            train_config['gradient_accumulation_steps'] = int(2)

        train_config['template'] = 'qwen'
        test_config['model_name'] = 'Qwen2.5-7B-Instruct'
        test_config['template'] = 'qwen'
    
    if 'phi_4' in model_name:
        train_config['model_name'] = 'Microsoft_Phi-4'
        per_device_train_batch_size_temp = original_per_device_train_batch_size
        gradient_accumulation_steps_temp = original_gradient_accumulation_steps
        per_device_train_batch_size_temp *= 2
        gradient_accumulation_steps_temp /= 2
        train_config['per_device_train_batch_size'] = int(per_device_train_batch_size_temp)
        train_config['gradient_accumulation_steps'] = int(gradient_accumulation_steps_temp)
        

        if 'AGIEVAL' in current_task_name.upper() or 'MMLU' == current_task_name.upper() or 'CODE' in current_task_name.upper() or 'MBPP' in current_task_name.upper() or data_n_train < 301:
            train_config['per_device_train_batch_size'] = int(5)
            train_config['gradient_accumulation_steps'] = int(2)

        train_config['template'] = 'phi'
        test_config['model_name'] = 'Phi-4'
        test_config['template'] = 'phi'

    train_config['per_device_train_batch_size'] = 3
    train_config['gradient_accumulation_steps'] = 10

    if 'plan_bench' in current_task_name.lower():
        if 'mistral' in model_name:
            test_config['per_device_eval_batch_size'] = 3
        if 'llama_3' in model_name or 'qwen' in model_name:
            test_config['per_device_eval_batch_size'] = 2
    if 'phi_4' in model_name:
        test_config['per_device_eval_batch_size'] = 1
    #     train_config['per_device_train_batch_size'] = 2
    #     train_config['gradient_accumulation_steps'] = 15
    test_config['use_cache'] = True

    if load_in_8bit:
        template = \
"""{
    "load_in_4bit": False,
    "load_in_8bit": True
}"""
        test_config['load_in_8bit'] = template
        train_config['load_in_8bit'] = template

    if 'MBPP' not in current_task_name.upper():
        if test_config['max_input_length'] < 1024:
            test_config['max_input_length'] = 1024
    
    return train_config, test_config

