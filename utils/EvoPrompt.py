""" DE and GA in EvoPrompt
"""

import configparser
import os
import random
import re
import numpy as np

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

def get_new_prompt(llm, prompt_4_create_new_os_prompt, temperature=0.5):
    """ call model 並 用正則表達抓<prompt></prompt>間的字
    Var
        temperature:
            EvoPrompt 論文的溫度設 0.5
    """
    if DEBUGGER=="True": print("enter get_new_prompt")

    reply = llm.reply(prompt_4_create_new_os_prompt, temperature)
    match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
    while(match_all == []):
        reply = llm.reply(prompt_4_create_new_os_prompt, temperature)
        match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
    new_prompt = match_all[-1]
    
    if DEBUGGER=="True": print("exit get_new_prompt")
    return new_prompt, reply

def get_distinct_new_propmt(llm, population, prompt_4_create_new_os_prompt, temperature=0.5, **kwargs):
    """ 用來確保 new_prompt 不在 population 裡
    Var
        temperature:
            EvoPrompt 論文的溫度設 0.5
        
        kwargs:
            用來 debug 的參數

    """
    if DEBUGGER=="True": print("enter get_distinct_new_propmt")

    # 嘗試生出新的 prompt
    num_inc_temp = int((1-temperature)//0.1)    # 生不出新的 prompt 就提高溫度
    num_pop = len(population)   # 溫度設最高之後給 pop 裡的每個參數各一次機會

    # temperature 升到 1.0 都有重複，pop 裡的每個 prompt 都生過一次，最後再試一次（類似鴿籠原理）
    num_try = num_inc_temp + num_pop + 1

    temperature = 0.5
    new_prompt, reply = get_new_prompt(llm, prompt_4_create_new_os_prompt, temperature)
    for _ in range(num_try):
        if prompt_in_list(population, new_prompt) is False:
            break
        temperature = min(1, temperature + 0.1) # prompt 重複的話就增加變化度
        new_prompt, reply = get_new_prompt(llm, prompt_4_create_new_os_prompt, temperature)
    
    # 如果經過 num_try 次還是重複，就 raise error
    if prompt_in_list(population, new_prompt) is True:
        print(f"{population=}")

        for k, v in kwargs.items():
            print(f"{k} = {v}")

        print(f"{new_prompt=}")
        print(f"{reply=}")
        print(f"{prompt_4_create_new_os_prompt=}")
        raise ValueError(f"想不到新的 prompt 了")
        
    if DEBUGGER=="True": print("exit get_distinct_new_propmt")
    return new_prompt

def init_condition(ttl_model, ttl_dataset, sorted_pairs):
    """ 用來初始化 EvoPrompt
    """
    if DEBUGGER=="True": print("enter init_condition")

    llm, _ = ttl_model
    train_split = ttl_dataset["train_split"]
    dev_split = ttl_dataset["dev_split"]
    if dev_split==[]:
        dev_split = train_split

    # 回歸初始狀態，看 record 的時候才能一眼看出哪個 iteration 變了
    old_population = [
        {
            "prompt": pair['prompt'],
            "train_score": pair['train_score'],
            "dev_score": pair['dev_score'],
        } for pair in sorted_pairs
    ]

    if DEBUGGER=="True": print("exit init_condition")
    return llm, train_split, dev_split, old_population


def EvoDE(ttl_model, ttl_dataset, sorted_pairs):
    """ 總共抽四筆 prompt: p_best, p_i, p_1, p_2
    """
    if DEBUGGER=="True": print("enter EvoDE")

    # llm, _ = ttl_model
    # train_split = ttl_dataset["train_split"]
    # dev_split = ttl_dataset["dev_split"]
    # if dev_split==[]:
    #     dev_split = train_split

    # # 回歸初始狀態，看 record 的時候才能一眼看出哪個 iteration 變了
    # old_population = [
    #     {
    #         "prompt": pair['prompt'],
    #         "train_score": pair['train_score'],
    #         "dev_score": pair['dev_score'],
    #     } for pair in sorted_pairs
    # ]

    llm, train_split, dev_split, old_population = init_condition(ttl_model, ttl_dataset, sorted_pairs)

    size_population = len(old_population)
    new_population = []

    p_best = sorted_pairs[0]['prompt']
    for i in range(size_population):
        p_i = old_population[i]['prompt']
        remaining_pairs = [pair for pair in old_population if pair['prompt'] != p_i]

        p_1, p_2 = np.random.choice(remaining_pairs, size=2, replace=False)
        p_1 = p_1["prompt"]
        p_2 = p_2["prompt"]

        prompt_4_create_new_os_prompt = get_prompt.create("EvoDE", p_best, p_i, p_1, p_2)
        new_prompt = get_distinct_new_propmt(llm, old_population, prompt_4_create_new_os_prompt)
            
        # 紀錄
        # 製作 new_pop: 取 train_score 高者
        train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
        if train_score <= old_population[i]["train_score"]:
            new_population.append(old_population[i])
        else:
            dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
            new_population.append({
                "prompt": new_prompt,
                "train_score": train_score,
                "dev_score": dev_score,
                "basic_prompt": p_i,
                "parent": [p_1, p_2]
            })

    if DEBUGGER=="True": print("exit EvoDE")
    return new_population

def get_weight(ttl_score):
    ttl_weight = np.array(ttl_score, dtype=float)
    if np.sum(ttl_weight!=0)<=1:
        ttl_weight[ttl_weight==0]=0.01  # 權重 0 的話會抽不到

    ttl_weight = ttl_weight/np.sum(ttl_weight)
    return ttl_weight

def EvoGA(ttl_model, ttl_dataset, sorted_pairs):
    """ 總共抽兩筆 prompt: p_1, p_2
    """
    if DEBUGGER=="True": print("enter EvoGA")

    # llm, _ = ttl_model
    # train_split = ttl_dataset["train_split"]
    # dev_split = ttl_dataset["dev_split"]
    # if dev_split==[]:
    #     dev_split = train_split

    # # 回歸初始狀態，看 record 的時候才能一眼看出哪個 iteration 變了
    # old_population = [
    #     {
    #         "prompt": pair['prompt'],
    #         "train_score": pair['train_score'],
    #         "dev_score": pair['dev_score'],
    #     } for pair in sorted_pairs
    # ]
    
    llm, train_split, dev_split, old_population = init_condition(ttl_model, ttl_dataset, sorted_pairs)

    size_population = len(sorted_pairs)
    new_population = []
    
    for _ in range(size_population):
        ttl_score = [pair['train_score'] for pair in sorted_pairs]
        ttl_weight = get_weight(ttl_score)

        p_1, p_2 = np.random.choice(old_population, size=2, replace=False, p=ttl_weight)
        p_1 = p_1["prompt"]
        p_2 = p_2["prompt"]

        prompt_4_create_new_os_prompt = get_prompt.create("EvoGA", p_1, p_2)
        new_prompt = get_distinct_new_propmt(llm, old_population, prompt_4_create_new_os_prompt)

        # 紀錄
        train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
        dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
        prompt_score = {
            'prompt': new_prompt,
            'train_score': train_score,
            'dev_score': dev_score,
            "parent": [p_1, p_2]
        }
        new_population.append(prompt_score)

    # 製作 new_pop: 取 top-half
    new_population = sorted(old_population+new_population,  key=lambda x: x['train_score'],  reverse=True)
    new_population = new_population[:size_population]
    
    if DEBUGGER=="True": print("exit EvoGA")
    return new_population
