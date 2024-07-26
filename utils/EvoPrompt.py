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

def get_new_prompt(llm, prompt_4_create_new_os_prompt):
    """ call model 並 用正則表達抓<prompt></prompt>間的字
    """
    if DEBUGGER=="True": print("enter get_new_prompt")

    reply = llm.reply(prompt_4_create_new_os_prompt, 0.5) # EvoPrompt 論文的溫度設 0.5
    match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
    while(match_all == []):
        reply = llm.reply(prompt_4_create_new_os_prompt, 0.5) # EvoPrompt 論文的溫度設 0.5
        match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
    new_prompt = match_all[-1]
    
    if DEBUGGER=="True": print("exit get_new_prompt")
    return new_prompt
    

def EvoDE(ttl_model, ttl_dataset, sorted_pairs):
    """ 總共抽四筆 prompt: p_best, p_i, p_1, p_2
    """
    if DEBUGGER=="True": print("enter EvoDE")

    llm, _ = ttl_model
    train_split = ttl_dataset["train_split"]
    dev_split = ttl_dataset["dev_split"]
    if dev_split==[]:
        dev_split = train_split

    p_best = sorted_pairs[0]['prompt']

    # 沒扣 best
    new_population = []
    rest_sorted_pair = sorted_pairs[:]
    for i in rest_sorted_pair:
        p_i = i['prompt']

        remaining_pairs = [pair for pair in rest_sorted_pair if pair['prompt'] != p_i]
        p_1, p_2 = random.sample(remaining_pairs, 2)
        p_1 = p_1["prompt"]
        p_2 = p_2["prompt"]

        prompt_4_create_new_os_prompt = get_prompt.create("EvoDE", p_best, p_i, p_1, p_2)
        new_prompt = get_new_prompt(llm, prompt_4_create_new_os_prompt)
        
        # 紀錄
        train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
        if train_score <= i["train_score"]:
            new_population.append(i)
        else:
            dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
            new_population.append({
                "prompt": new_prompt,
                "train_score": train_score,
                'dev_score': dev_score,
                "basic_prompt": p_i
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

    llm, _ = ttl_model
    train_split = ttl_dataset["train_split"]
    dev_split = ttl_dataset["dev_split"]
    if dev_split==[]:
        dev_split = train_split

    size_population = len(sorted_pairs)

    new_population = sorted_pairs[:]
    for i in range(size_population):
        ttl_prompt = [pair['prompt'] for pair in sorted_pairs]
        ttl_score = [pair['train_score'] for pair in sorted_pairs]
        ttl_weight = get_weight(ttl_score)
        p_1, p_2 = np.random.choice(ttl_prompt, size=2, replace=False, p=ttl_weight)

        prompt_4_create_new_os_prompt = get_prompt.create("EvoGA", p_1, p_2)
        new_prompt = get_new_prompt(llm, prompt_4_create_new_os_prompt)
        
        # 紀錄
        train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
        dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
        prompt_score = {
            'prompt': new_prompt,
            'train_score': train_score,
            'dev_score': dev_score
        }
        new_population.append(prompt_score)
        
    new_population = sorted(new_population,  key=lambda x: x['train_score'],  reverse=True)
    new_population = new_population[:size_population]
    
    if DEBUGGER=="True": print("exit EvoGA")
    return new_population
