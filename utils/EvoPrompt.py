import configparser
import os
import random
import re

from .score import get_score
from .tools import prompt_in_list
from .prompt import get_prompt

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)

DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

def EvoPrompt(init_set):

    ttl_model, ttl_data, ttl_pair_os_prompt_scores = init_set
    get_llm_reply, _ = ttl_model

    # 停止條件
    stop_score = 100
    stop_run_num = 0

    # # 把目前的 prompt score pair 照 score 由高排到低
    sorted_pairs = sorted(ttl_pair_os_prompt_scores, key=lambda x: x['score'], reverse=True)
    # 控制什麼時候要生新的 prompt
    print("before while")
    while (sorted_pairs[0]['score'] < stop_score and stop_run_num >= 0):
        print("\n"*3)
        print(f"{stop_run_num=}")

        # 總共抽四筆 prompt: p_best, p_i, p_1, p_2
        p_best = sorted_pairs[0]['prompt']

        # # 有扣 best
        # new_population = [sorted_pairs[0]]
        # rest_sorted_pair = sorted_pairs[1:]
        
        # 沒扣 best
        new_population = []
        rest_sorted_pair = sorted_pairs[:]
        
        for i in rest_sorted_pair:
            p_i = i['prompt']

            remaining_pairs = [pair for pair in rest_sorted_pair if pair['prompt'] != p_i]
            p_1, p_2 = random.sample(remaining_pairs, 2)
            p_1 = p_1["prompt"]
            p_2 = p_2["prompt"]

            print()
            print(f"{p_best=}")
            print(f"{p_i=}")
            print(f"{p_1=}")
            print(f"{p_2=}")
            print()
            
            prompt_4_create_new_os_prompt = get_prompt(3, p_best, p_i, p_1, p_2)
            print(f"{prompt_4_create_new_os_prompt=}\n")
            # reply = get_llm_reply(prompt_4_create_new_os_prompt, 4096, 1)
            reply = get_llm_reply(prompt_4_create_new_os_prompt, 8192, 1)
            print(f"{reply=}\n")

            # 正則表達抓<prompt></prompt>間的字
            # match = re.search(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
            # if match:
            #     new_prompt = match.group(1)
            
            match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
            while(match_all == []):
                reply = get_llm_reply(prompt_4_create_new_os_prompt, 8192, 1)
                match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
            new_prompt = match_all[-1]

            print(f"{new_prompt=}")

            # 紀錄
            # print("="*50)
            # print(new_prompt)

            # if prompt_in_list(ttl_pair_os_prompt_scores, new_prompt) is False:
            #     score = get_score(ttl_model, new_prompt, ttl_data, 3000, 10)
            #     # print("*"*50)
            #     # print(f"{new_prompt}\n{score}")
            #     prompt_score = {
            #         'prompt': new_prompt, 
            #         'score': score
            #     }
            #     ttl_pair_os_prompt_scores.append(prompt_score)
            #     sorted_pairs = sorted(ttl_pair_os_prompt_scores,  key=lambda x: x['score'],  reverse=True)

            #     # 保留原本 prompt score pair 的總數
            #     ttl_pair_os_prompt_scores = sorted_pairs[:-1]
            # else:
            #     print("prompt exist")
                
            new_score = get_score(ttl_model, new_prompt, ttl_data, 3000, 10)
            s_i = i["score"]
            
            if new_score <= s_i:
                new_population.append(i)
            else:
                new_population.append({
                    "prompt": new_prompt,
                    "score": new_score
                })

        # 把目前的 prompt score pair 照 score 由高排到低
        # sorted_pairs = sorted(ttl_pair_os_prompt_scores, key=lambda x: x['score'], reverse=True)
        sorted_pairs = sorted(new_population, key=lambda x: x['score'], reverse=True)
        stop_run_num -= 1
    print("after while")

    return ttl_pair_os_prompt_scores