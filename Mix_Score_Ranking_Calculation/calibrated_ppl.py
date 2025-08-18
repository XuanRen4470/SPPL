from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
import re
import numpy as np

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import MODEL_DIRECTORY, HOME_DIRECTORY
from utils.in_context_perplexity_measurement_function import calibrated_perplexity_calculation, probability_calculation, calibrated_perplexity_calculation_given_probability, probability_ppl_calculation_gpt2_sliding, topk_entropy_calculation, calibrated_perplexity_with_entropy_threshold_calculation, calibrated_perplexity_calculation_given_entropy, probability_calculation_, topk_width_and_mass_calculation, multi_path_probability_calculation, gpt2_ppl_calculation, probability_calculation_modern, probability_ppl_calculation_gpt2_sliding, length_calibration_ppl_calculation, probability_calculation_modern_sliding_window, importance_ratio_ppl_calculation
from utils.in_context_data_loader import perplexity_calculation_in_context_data_loader

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

not_cap_perplexity = True
calc_IDF = True
CAR_beta = 3
n_similar_self_generated_examples = 2


# task_name_list: specify the tasks you want to evaluate
train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank', 'plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse']  
# train_task_list = ['math_geometry', 'api_bank']#, 'plan_bench_generalization']#, 'esnli']    


# train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']  


# train_task_list = ['math_algebra']  
# train_task_list = ['api_bank']  




# train_task_list = ['gsm8k', 'math_algebra', 'mmlu', 'winogrande', 'piqa', 'agieval', 'squad', 'ecqa', 'boolq', 'arc_challenge', 'mmlu_pro_law', 'drop', 'hellaswag', 'mbpp', 'mmlu_moral_scenarios', 'math_geometry', 'api_bank']  


# train_task_list = ['plan_bench_generation', 'plan_bench_generalization', 'plan_bench_optimality']
# train_task_list = ['plan_bench_optimality']
# train_task_list = ['esnli']
# train_task_list = ['api_bank']
# train_task_list = ['plan_bench_replaning']

# train_task_list = ['plan_bench_generation', 'plan_bench_optimality', 'plan_bench_generalization', 'plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse']

# train_task_list = ['plan_bench_verification', 'plan_bench_execution', 'plan_bench_replaning', 'plan_bench_reuse']


# n_train = 4#1000
# n_train = 100#1000
# n_train = 30#0
n_train = 500#00
# n_train = 499#00
# n_train = 10#00

function_template = 'no_calibration'

function_template = '2'
function_template = '3'
function_template = '4'
function_template = '5'
function_template = '6'
function_template = '7'


calibrate_method = ''
calibrate_method = 'entropy_clip_higher'
calibrate_method = 'entropy_clip_lower'
calibrate_method = 'divide_entropy'
calibrate_method = 'probability'
calibrate_method = 'multi_path'
calibrate_method = 'gpt2'
calibrate_method = 'length_calibration'
calibrate_method = 'importance_ratio'



model_name_list = ['mistral', 'llama_3_instruct', 'qwen']
# model_name_list = ['mistral']
# model_name_list = ['gpt2']




def load_model(model_name):
    model_base = None
    if 'mistral' in model_name:
        model_path = f"{MODEL_DIRECTORY}/Mistral-7b-Instruct-v2"
    elif 'llama_3_instruct' in model_name:
        model_path = f"{MODEL_DIRECTORY}/Meta-Llama-3-8B-Instruct"
    elif 'phi_4' in model_name or 'phi-4' in model_name:
        model_path = f"{MODEL_DIRECTORY}/Phi-4"
    elif 'qwen' in model_name:
        model_path = f'{MODEL_DIRECTORY}/Qwen2.5-7B-Instruct'
    elif 'gpt2' in model_name:
        model_path = f'{MODEL_DIRECTORY}/GPT2'

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "gpt2" in model_name.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token      # id = 50256
        tokenizer.padding_side = "left"                    # 推荐
        extra_kwargs = dict()                              # GPT-2 通常 fp32

        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **extra_kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id     # 与 tokenizer 保持一致
        model.to(device).eval()  

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use float16 for reduced memory usage
            device_map="auto"  # Automatic device mapping for optimal performance
        )
    model.to(device)
    return model, tokenizer, model_base



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for model_name in model_name_list:
# # for model_name in ['qwen', 'llama_3_instruct']:
# # for model_name in ['mistral']:
# # for model_name in ['llama_3_instruct']:
# # for model_name in ['qwen']:
# # for model_name in ['mistral', 'llama_3_instruct']:
# # for model_name in ['gpt2']:
#     model, tokenizer, model_base = load_model(model_name)
#     for train_task_name in train_task_list:
#         record_book = {}
#         token_level_prob_record_book = {}
#         layerwise_cos_similarity_record_book = {}

#         print()
#         print()
#         print(f'----------------------------------------------------------{train_task_name}----------------------------------------------------------')            
        
#         dataset_list, train_config, test_config, test_task_name, gpt4_prediction_list = perplexity_calculation_in_context_data_loader(train_task_name, n_train, False, -1, '')

#         import torch, os

#         token_prob_dict = {}
#         calibrated_result = {}
#         # dataset_list = [dataset_list[1]]
#         for data_name, data_list, *_ in dataset_list:
#             print(f'—— {data_name} ——')

#             if 'gpt2' in model_name:
#                 token_prob_dict[data_name] = probability_ppl_calculation_gpt2_sliding(
#                     data_list, model, tokenizer, model_name, device='cuda'
#                 )
#             else:
#                 token_prob_dict[data_name] = probability_calculation(
#                     data_list, model, tokenizer, model_name, device='cuda'
#                 )
#                 # data_list = data_list[:50]

                
#                 # results, num_of_high_prob_tokens_list, avg_num_of_high_prob_tokens= probability_calculation_modern_sliding_window(
#                 #     data_list, model, tokenizer, model_name, device='cuda', threshold= 0.98
#                 # )
#                 # results, num_of_high_prob_tokens_list, avg_num_of_high_prob_tokens= probability_calculation_modern(
#                 #     data_list, model, tokenizer, model_name, device='cuda', threshold= 0.98
#                 # )

#                 # calibrated_result[data_name] = {}
#                 # calibrated_result[data_name]['result'] = results
#                 # calibrated_result[data_name]['num_of_high_prob_tokens_list'] = num_of_high_prob_tokens_list
#                 # calibrated_result[data_name]['avg_num_of_high_prob_tokens'] = avg_num_of_high_prob_tokens
#                 # print('avg_num_of_high_prob_tokens', avg_num_of_high_prob_tokens)

            

#             # topk_ent, token_entropy_list = topk_entropy_calculation(
#             #     data_list, model, tokenizer, model_name,
#             #     k=20, device='cuda', return_token_level=True
#             # )
#             # print(f'每条样本的平均 top-20 熵: {topk_ent[:5]} ...')

#             # pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
#             #         f"record_book/token_level_entropy_record/{n_train}_entropy_{data_name}_{model_name}_{train_task_name}.pt")
#             # os.makedirs(os.path.dirname(pt_path), exist_ok=True)

#             # # torch.save(`token_entropy_list`, pt_path)         # 写
#             # torch.save(token_entropy_list, pt_path)         # 写
#             # print("✓ saved to", pt_path)





#         # pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
#         #         f"record_book/length_calibration/{n_train}_record_info_{model_name}_{train_task_name}.pt")

#         # os.makedirs(os.path.dirname(pt_path), exist_ok=True)
        
#         # torch.save(calibrated_result, pt_path)         # 写
#         # print("✓ saved to", pt_path)



#         pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
#                     f"record_book/token_level_probability_record/{n_train}_probability_{model_name}_{train_task_name}.pt")

#         os.makedirs(os.path.dirname(pt_path), exist_ok=True)

#         torch.save(token_prob_dict, pt_path)         # 写
#         print("✓ saved to", pt_path)


# # # #             sample_width_list, token_width_list, token_mass_list, top3_labels, top3_probs, eff_probs =  topk_width_and_mass_calculation(
# # # #                     data_list,
# # # #                     model,
# # # #                     tokenizer,
# # # #                     model_name,
# # # #                     threshold=0.9,
# # # #                     device='cuda',
# # # #                     effective_probability_threshold = 0.1,
# # # #                     return_token_level=True)
    
# # # #             k = 1

# # # #             pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
# # # #                     f"record_book/token_level_entropy_record/{n_train}_effective_prob_record_{data_name}_{model_name}_{train_task_name}.pt")
# # # #             os.makedirs(os.path.dirname(pt_path), exist_ok=True)
# # # #             torch.save(eff_probs, pt_path)         # 写
# # # #             print("✓ saved to", pt_path)



                
#     import gc
#     del model
#     del tokenizer
#     del model_base

#     # Trigger garbage collection
#     gc.collect()

#     # Empty CUDA cache
#     torch.cuda.empty_cache()






# token_entropy_list_total = []
# # for model_name in ['mistral']:    
# for model_name in model_name_list:
#     index_range_list = [(0, 50)]
#     for initial_index, last_index in index_range_list:
#         for train_task_name in train_task_list:            
#             dataset_list, train_config, test_config, test_task_name, gpt4_prediction_list = perplexity_calculation_in_context_data_loader(train_task_name, n_train, False, -1, '')

#             for data_name, data_list, original_file_path, origianl_data_list, suffix in dataset_list:

#                 # -------- 与保存时保持一致的路径构造 --------
#                 pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
#                         f"record_book/token_level_entropy_record/{n_train}_entropy_{data_name}_{model_name}_{train_task_name}.pt")

#                 if not os.path.isfile(pt_path):
#                     raise FileNotFoundError(f"❌ 找不到文件：{pt_path}")

#                 # -------- 加载 --------
#                 token_entropy_list = torch.load(pt_path, map_location='cpu')   # 如果想直接放 GPU 改成 map_location='cuda'
#                 token_entropy_list = token_entropy_list[initial_index:last_index]
#                 for iiitem in token_entropy_list:
#                     token_entropy_list_total += iiitem
#     threshold = float(np.percentile(token_entropy_list_total, 80))
#     a = 1


#     pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
#             f"record_book/token_level_entropy_record/{model_name}_threshold.pt")
#     os.makedirs(os.path.dirname(pt_path), exist_ok=True)

#     torch.save(threshold, pt_path)        
#     a = 1

















# for model_name in [model_name]:
for model_name in model_name_list:
# for model_name in ['gpt2']:
# for model_name in ['qwen', 'llama_3_instruct']:
# for model_name in ['mistral']:
# for model_name in ['llama_3_instruct']:
# for model_name in ['qwen']:
# for model_name in ['mistral', 'llama_3_instruct']:

    
    model, tokenizer, model_base = load_model(model_name)
    
    index_range_list = [(0, 50)]
    # index_range_list = [(0, 10)]
    for initial_index, last_index in index_range_list:
        for train_task_name in train_task_list:
            record_book = {}
            token_level_prob_record_book = {}
            layerwise_cos_similarity_record_book = {}
 
            print()
            print()
            print(f'----------------------------------------------------------{train_task_name}----------------------------------------------------------')            
            
            dataset_list, train_config, test_config, test_task_name, gpt4_prediction_list = perplexity_calculation_in_context_data_loader(train_task_name, n_train, False, -1, '')

            # dataset_list = [dataset_list[-2]]
            for data_name, data_list, original_file_path, origianl_data_list, suffix in dataset_list:
                print()
                print(f'----------------------------------------------------------{data_name}----------------------------------------------------------')

                data_list = data_list[initial_index:last_index]

                

                

                pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
                        f"record_book/token_level_probability_record/{n_train}_probability_{model_name}_{train_task_name}.pt")

                token_prob_dict = torch.load(pt_path, map_location='cpu')   # 读


                token_prob_list = token_prob_dict[data_name]
                token_prob_list = token_prob_list[initial_index:last_index]

                

                # average_perplexity = calibrated_perplexity_calculation_given_probability(token_prob_list, function_template = function_template, reference_token_prob_list = token_prob_list)
                # # average_perplexity = calibrated_perplexity_calculation_given_probability(token_prob_list)

                # average_perplexity = calibrated_perplexity_calculation(data_list, model, tokenizer, model_name, device=device)


                





                a = 1

                if calibrate_method == 'higher' or calibrate_method == 'lower' or calibrate_method == 'divide_entropy':

                    # -------- 与保存时保持一致的路径构造 --------
                    pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
                            f"record_book/token_level_entropy_record/{n_train}_entropy_{data_name}_{model_name}_{train_task_name}.pt")

                    if not os.path.isfile(pt_path):
                        raise FileNotFoundError(f"❌ 找不到文件：{pt_path}")

                    # -------- 加载 --------
                    token_entropy_list = torch.load(pt_path, map_location='cpu')   # 如果想直接放 GPU 改成 map_location='cuda'
                    token_entropy_list = token_entropy_list[initial_index:last_index]
                    
                    pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
                            f"record_book/token_level_entropy_record/{model_name}_threshold.pt")
                    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
                    threshold = torch.load(pt_path, map_location='cpu')  
                
                if calibrate_method == 'higher':
                    average_perplexity = calibrated_perplexity_with_entropy_threshold_calculation(token_prob_list, token_entropy_list, threshold, clip = 'higher')
                elif calibrate_method == 'lower':
                    average_perplexity = calibrated_perplexity_with_entropy_threshold_calculation(token_prob_list, token_entropy_list, threshold, clip = 'lower')
                elif calibrate_method == 'divide_entropy':
                    average_perplexity = calibrated_perplexity_calculation_given_entropy(token_prob_list, token_entropy_list, threshold, function_template = function_template)
                elif calibrate_method == 'probability':
                    # average_perplexity, avg_token_lenth, avg_ppl, avg_calibrated_ppl = probability_calculation_(token_prob_list)
                    average_perplexity, avg_token_lenth, avg_ppl, avg_calibrated_ppl = probability_calculation(token_prob_list)

                elif calibrate_method == 'multi_path':
                    pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
                        f"record_book/token_level_entropy_record/{n_train}_effective_prob_record_{data_name}_{model_name}_{train_task_name}.pt")
                    multi_path_prob_dict = torch.load(pt_path, map_location='cpu')   # 读
                    multi_path_prob_list = multi_path_prob_dict[initial_index:last_index]
                    avg_ppl, avg_calibrated_ppl = multi_path_probability_calculation(token_prob_list, multi_path_prob_list)
                elif calibrate_method == 'gpt2':
                    pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
                        f"record_book/{n_train}_probability_gpt2_{train_task_name}.pt")
                    gpt2_token_prob_dict = torch.load(pt_path, map_location='cpu')   # 读

                    gpt2_token_prob_list = gpt2_token_prob_dict[data_name]
                    gpt2_token_prob_list = gpt2_token_prob_list[initial_index:last_index]

                    avg_ppl, avg_calibrated_ppl = gpt2_ppl_calculation(token_prob_list, gpt2_token_prob_list)

                elif calibrate_method == 'length_calibration':
                    pt_path = (f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/"
                    f"record_book/length_calibration/{n_train}_record_info_{model_name}_{train_task_name}.pt")
                    calibrated_result = torch.load(pt_path)
         
                    num_of_high_prob_tokens_list = calibrated_result[data_name]['num_of_high_prob_tokens_list']

                    avg_ppl, avg_calibrated_ppl = length_calibration_ppl_calculation(token_prob_list, num_of_high_prob_tokens_list)

                    # avg_ppl, avg_calibrated_ppl = length_calibration_ppl_calculation(token_prob_list)

                elif calibrate_method == 'importance_ratio':
                    # avg_ppl, avg_calibrated_ppl = importance_ratio_ppl_calculation(token_prob_list, ratio = 0.1)
                    avg_ppl, avg_calibrated_ppl = importance_ratio_ppl_calculation(token_prob_list)
                    


                



                key = f'{train_task_name}_{data_name}'

                if calibrate_method == 'gpt2':
                    record_book[key] = avg_calibrated_ppl
                    print(f'avg_ppl: {avg_ppl}')
                    print(f'calibrated_perplexity: {avg_calibrated_ppl}')
                elif calibrate_method == 'multi_path':
                    record_book[key] = avg_calibrated_ppl
                    print(f'avg_ppl: {avg_ppl}')
                    print(f'calibrated_perplexity: {avg_calibrated_ppl}')
                elif calibrate_method == 'length_calibration':
                    record_book[key] = avg_calibrated_ppl
                    print(f'avg_ppl: {avg_ppl}')
                    print(f'calibrated_perplexity: {avg_calibrated_ppl}')
                elif calibrate_method == 'importance_ratio':
                    record_book[key] = avg_calibrated_ppl.cpu().item()
                    print(f'avg_ppl: {avg_ppl}')
                    print(f'calibrated_perplexity: {avg_calibrated_ppl}')
                else:
                    record_book[key] = average_perplexity
                    print(f'calibrated_perplexity: {average_perplexity}')
                    print(f'avg_token_lenth: {avg_token_lenth}')
                    print(f'avg_ppl: {avg_ppl}')
                    print(f'avg_calibrated_ppl: {avg_calibrated_ppl}')

                a = 1
                with open(f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/{n_train}_calibrated_ppl_{model_name}_{train_task_name}_{initial_index}_{last_index}_{function_template}.json", 'w') as f:
                    json.dump(record_book, f, indent=4)     




                # save_path = f"{HOME_DIRECTORY}/Mix_Score_Ranking_Calculation/Mix_Score_record/record_book/calibrated_ppl/{calibrate_method}_{n_train}_calibrated_ppl_with_entropy_threshold_{model_name}_{train_task_name}_{initial_index}_{last_index}.json"

                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # with open(save_path, 'w') as f:
                #     json.dump(record_book, f, indent=4)   


            a = 1

    import gc
    del model
    del tokenizer
    del model_base

    # Trigger garbage collection
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()


    a = 1



