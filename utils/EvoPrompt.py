""" DE and GA in EvoPrompt
"""

import configparser
import os
# import random
import re
import numpy as np
from abc import ABC, abstractmethod

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

def get_distinct_new_propmt(llm, population, prompt_4_create_new_os_prompt, temperature=0.5, **for_debug):
    """ 用來確保 new_prompt 不在 population 裡
    Var
        temperature:
            EvoPrompt 論文的溫度設 0.5
        
        for_debug:
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

        for k, v in for_debug.items():
            print(f"{k}={v}")

        print(f"{new_prompt=}")
        print(f"{reply=}")
        print(f"{prompt_4_create_new_os_prompt=}")
        raise ValueError(f"想不到新的 prompt 了")
        
    if DEBUGGER=="True": print("exit get_distinct_new_propmt")
    return new_prompt

def EvoDE(ttl_model, ttl_dataset, sorted_pairs):
    """ 總共用四筆 prompt: p_best, p_i, p_1, p_2
    """
    if DEBUGGER=="True": print("enter EvoDE")

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

        for_debug ={
            "p_1": p_1,
            "p_2": p_2,
            "p_best": p_best,
            "p_i": p_i
        }
        prompt_4_create_new_os_prompt = get_prompt.create("EvoDE", p_1, p_2, p_best, p_i)
        new_prompt = get_distinct_new_propmt(llm, old_population, prompt_4_create_new_os_prompt, **for_debug)
            
        # 紀錄
        # 製作 new_pop: 取 train_score 高者
        train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
        if train_score <= old_population[i]["train_score"]:
            new_population.append(old_population[i])
        else:
            dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
            prompt_score = {
                "prompt": new_prompt,
                "train_score": train_score,
                "dev_score": dev_score,
                "info": {
                    "p_1": p_1,
                    "p_2": p_2,
                    "p_best": p_best,
                    "p_i": p_i
                }
            }
            new_population.append(prompt_score)

    if DEBUGGER=="True": print("exit EvoDE")
    return new_population

def get_weight(ttl_score):
    ttl_weight = np.array(ttl_score, dtype=float)
    if np.sum(ttl_weight!=0)<=1:
        ttl_weight[ttl_weight==0]=0.01  # 權重 0 的話會抽不到

    ttl_weight = ttl_weight/np.sum(ttl_weight)
    return ttl_weight

def EvoGA(ttl_model, ttl_dataset, sorted_pairs):
    """ 總共用兩筆 prompt: p_1, p_2
    """
    if DEBUGGER=="True": print("enter EvoGA")
    
    llm, train_split, dev_split, old_population = init_condition(ttl_model, ttl_dataset, sorted_pairs)

    size_population = len(sorted_pairs)
    new_population = []
    
    for _ in range(size_population):
        ttl_score = [pair['train_score'] for pair in sorted_pairs]
        ttl_weight = get_weight(ttl_score)

        p_1, p_2 = np.random.choice(old_population, size=2, replace=False, p=ttl_weight)
        p_1 = p_1["prompt"]
        p_2 = p_2["prompt"]

        for_debug ={
            "p_1": p_1,
            "p_2": p_2
        }
        prompt_4_create_new_os_prompt = get_prompt.create("EvoGA", p_1, p_2)
        new_prompt = get_distinct_new_propmt(llm, old_population, prompt_4_create_new_os_prompt, **for_debug)

        # 紀錄
        train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
        dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
        prompt_score = {
            'prompt': new_prompt,
            'train_score': train_score,
            'dev_score': dev_score,
            "info": {
                "p_1": p_1,
                "p_2": p_2,
            }
        }
        new_population.append(prompt_score)

    # 製作 new_pop: 取 top-half
    new_population = sorted(old_population+new_population,  key=lambda x: x['train_score'],  reverse=True)
    new_population = new_population[:size_population]
    
    if DEBUGGER=="True": print("exit EvoGA")
    return new_population

def evaluate(prompt, ttl_model, train_split, dev_split):
    """ 用來評估 prompt 的分數
    """
    train_score = get_score(ttl_model, prompt, train_split, 3000, 10)
    dev_score = get_score(ttl_model, prompt, dev_split, 3000, 10)
    prompt_score = {
        "prompt": prompt,
        "train_score": train_score,
        "dev_score": dev_score
    }
    return prompt_score

def formulate_pair(pair):
    """ prompt 可能紀錄很多資訊只留下 prompt 跟 *_score 方便讀檔的時候知道從哪個 iteration 開始變化
    """
    formulated_pair = {
        "prompt": pair['prompt'],
        "train_score": pair['train_score'],
        "dev_score": pair['dev_score'],
    }
    return formulated_pair

# def CoEvo(ttl_model, ttl_dataset, sorted_pairs, init_population):
#     """ 總共用五筆 prompt: p_worst, p_init, p_contr, p_best, p_i

#     p_worst + p_init -> p_contr

#     """
#     if DEBUGGER=="True": print("enter CoEvo")
#     llm, train_split, dev_split, old_population = init_condition(ttl_model, ttl_dataset, sorted_pairs)

#     size_population = len(old_population)
#     new_population = []

    
#     p_best = sorted_pairs[0]['prompt']
#     p_worst = old_population[-1]['prompt']
#     # p_init = np.random.choice(init_population, size=1)    # 有 size 的話輸出會是 np.ndarray
#     p_init = np.random.choice(init_population)
#     p_init = p_init['prompt']
#     for i in range(size_population):
#         p_i = old_population[i]['prompt']

#         for_debug ={
#             "p_worst": p_worst,
#             "p_init": p_init
#         }
#         prompt_4_create_p_init = get_prompt.create("EvoGA", p_worst, p_init)
#         # 之前的經驗是 GA 做到最後也會重複，重複的話就相當於在 old_population 裡抽 p_, p_best
#         # 又 currnet population 跟 init_population 同源
#         # 所以欣璇覺得極有可能重複，不能單用 get_new_prompt
#         p_contr = get_distinct_new_propmt(llm, old_population, prompt_4_create_p_init, **for_debug)

#         for_debug ={
#             "p_contr": p_contr,
#             "p_best": p_best,
#             "p_i": p_i
#         }
#         prompt_4_create_new_os_prompt = get_prompt.create("CoEvo", p_contr, p_best, p_i)
#         new_prompt = get_distinct_new_propmt(llm, old_population, prompt_4_create_new_os_prompt, **for_debug)

#         # 紀錄
#         # train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
#         # if train_score <= old_population[i]["train_score"]:
#         #     new_population.append(old_population[i])
#         # else:
#         #     dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
#         #     prompt_score = {
#         #         "prompt": new_prompt,
#         #         "train_score": train_score,
#         #         "dev_score": dev_score,
#         #         "info": {
#         #             "p_i": p_i,
#         #             "p_best": p_best,
#         #             "p_contr": p_contr,
#         #             "p_worst": p_worst,
#         #             "p_init": p_init,
#         #         }
#         #     }
#         #     new_population.append(prompt_score)

#         train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
#         dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
#         prompt_score = {
#             "prompt": new_prompt,
#             "train_score": train_score,
#             "dev_score": dev_score,
#             "info": {
#                 "p_i": p_i,
#                 "p_best": p_best,
#                 "p_contr": p_contr,
#                 "p_worst": p_worst,
#                 "p_init": p_init,
#             }
#         }
#         new_population.append(prompt_score)
    
#     if DEBUGGER=="True": print("exit CoEvo")
#     return new_population

def CoEvo(ttl_model, ttl_dataset, sorted_pairs, init_population):
    """ 總共用五筆 prompt: p_worst, p_init, p_contr, p_best, p_i

    p_worst + p_init -> p_contr

    """
    if DEBUGGER=="True": print("enter CoEvo")
    llm, train_split, dev_split, old_population = init_condition(ttl_model, ttl_dataset, sorted_pairs)

    size_population = len(old_population)
    new_population = []

    
    pair_best = sorted_pairs[0]
    pair_worst = old_population[-1]
    # p_init = np.random.choice(init_population, size=1)    # 有 size 的話輸出會是 np.ndarray
    # pair_init = np.random.choice(init_population)
    for i in range(size_population):
        pair_init = np.random.choice(init_population)
        pair_i = old_population[i]

        for_debug ={
            "p_worst": pair_worst["prompt"],
            "p_init": pair_init["prompt"]
        }
        prompt_4_create_p_init = get_prompt.create("EvoGA", pair_worst["prompt"], pair_init["prompt"])
        # 之前的經驗是 GA 做到最後也會重複，重複的話就相當於在 old_population 裡抽 p_, p_best
        # 又 currnet population 跟 init_population 同源
        # 所以欣璇覺得極有可能重複，不能單用 get_new_prompt
        p_contr = get_distinct_new_propmt(llm, old_population, prompt_4_create_p_init, **for_debug)

        for_debug ={
            "p_contr": p_contr,
            "p_best": pair_best["prompt"],
            "p_i": pair_i["prompt"]
        }
        prompt_4_create_new_os_prompt = get_prompt.create("CoEvo", p_contr, pair_best["prompt"], pair_i["prompt"])
        new_prompt = get_distinct_new_propmt(llm, old_population, prompt_4_create_new_os_prompt, **for_debug)

        # 紀錄

        prompt_score = evaluate(new_prompt, ttl_model, train_split, dev_split)
        info = {
            "p_i": formulate_pair(pair_i),
            "p_best": formulate_pair(pair_best),
            "p_contr": evaluate(p_contr, ttl_model, train_split, dev_split),
            "p_worst": formulate_pair(pair_worst),
            "p_init": formulate_pair(pair_init),
        }
        prompt_score["info"] = info
        new_population.append(prompt_score)
    
    if DEBUGGER=="True": print("exit CoEvo")
    return new_population

# 練習寫 ABC

class EvoPrompt(ABC):

    def __init__(self, type_evo, ttl_model, ttl_dataset, population, init_population=None):
        """ 用來初始化 EvoPrompt
        Var
            type_evo: str
                "EvoDE", "EvoGA", "CoEvo"

            ttl_model: tuple(llm, tokenizer)
                llm: function
                embedding_model: Encoder in utils.call_model.embedding
            
            ttl_dataset: dict
                {
                    "train_split": list,
                    "dev_split": list
                }
                formation of the element in list:
                    {
                        "content": str,
                        "question": str,
                        "answer": str
                    }
            
            population, init_population: list
                formation of the element in list is dict with at least:
                    {
                        "prompt": str,
                        "train_score": float,
                        "dev_score": float
                    }

        """

        self.type_evo = type_evo
        self.init_population = init_population.deepcopy()
        self.ttl_model = ttl_model
        self.train_split = ttl_dataset["train_split"].deepcopy()
        if ttl_dataset["dev_split"]==[]:
            self.dev_split = self.train_split.deepcopy()
        else:
            self.dev_split = ttl_dataset["dev_split"].deepcopy()

        # 回歸初始狀態，看 record 的時候才能一眼看出哪個 iteration 變了
        self.population = [ self.formulate_pair(pair) for pair in population ]
        self.population = sorted(self.population,  key=lambda x: x['train_score'],  reverse=True)
        self.size_population = len(self.population)

    def formulate_pair(self, pair):
        """ prompt 可能紀錄很多資訊只留下 prompt 跟 *_score 方便讀檔的時候知道從哪個 iteration 開始變化
        """
        if type(pair)==str: # 有時候會是 str
            formulated_pair = self.evaluate(pair)
        else:
            formulated_pair = {
                "prompt": pair['prompt'],
                "train_score": pair['train_score'],
                "dev_score": pair['dev_score'],
            }
        return formulated_pair
    
    def prompt_in_list(self, new_prompt):
        """
        Var
            new_prompt: str
                the prompt to be check whether or not the in ttl_prompt
        """
        # 預設不在
        in_list = False
        # 然後檢查
        for pire in self.population:
            if pire['prompt'] == new_prompt:
                in_list = True
                break
        return in_list

    def get_new_prompt(self, prompt_4_create_new_os_prompt, temperature=0.5):
        """ call model 並 用正則表達抓<prompt></prompt>間的字
        Var
            temperature:
                EvoPrompt 論文的溫度設 0.5
        """
        if DEBUGGER=="True": print("enter get_new_prompt")

        llm, _ = self.ttl_model

        reply = llm.reply(prompt_4_create_new_os_prompt, temperature)
        match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
        while(match_all == []):
            reply = llm.reply(prompt_4_create_new_os_prompt, temperature)
            match_all = re.findall(r'<\s*prompt\s*>(.*?)<\s*/\s*prompt\s*>', reply, re.DOTALL)
        new_prompt = match_all[-1]
        
        if DEBUGGER=="True": print("exit get_new_prompt")
        return new_prompt, reply

    def get_distinct_new_propmt(self, prompt_4_create_new_os_prompt, temperature=0.5, **for_debug):
        """ 用來確保 new_prompt 不在 population 裡
        Var
            temperature:
                EvoPrompt 論文的溫度設 0.5
            
            for_debug:
                用來 debug 的參數

        """
        # 嘗試生出新的 prompt
        num_inc_temp = int((1-temperature)//0.1)    # 生不出新的 prompt 就提高溫度

        # 溫度設最高之後給 pop 裡的每個參數各一次機會
        # temperature 升到 1.0 都有重複，pop 裡的每個 prompt 都生過一次，最後再試一次（類似鴿籠原理）
        num_try = num_inc_temp + self.size_population + 1

        temperature = 0.5
        new_prompt, reply = self.get_new_prompt(prompt_4_create_new_os_prompt, temperature)
        for _ in range(num_try):
            if self.prompt_in_list(new_prompt) is False:
                break
            temperature = min(1, temperature + 0.1) # prompt 重複的話就增加變化度
            new_prompt, reply = self.get_new_prompt(prompt_4_create_new_os_prompt, temperature)
        
        # 如果經過 num_try 次還是重複，就 raise error
        if self.prompt_in_list(new_prompt) is True:
            print(f"{self.population=}")

            for k, v in for_debug.items():
                print(f"{k}={v}")

            print(f"{new_prompt=}")
            print(f"{reply=}")
            print(f"{prompt_4_create_new_os_prompt=}")
            raise ValueError(f"想不到新的 prompt 了")

        return new_prompt

    def evaluate(self, prompt):
        """ 用來評估 prompt 的分數
        """
        train_score = get_score(self.ttl_model, prompt, self.train_split, 3000, 10)
        dev_score = get_score(self.ttl_model, prompt, self.dev_split, 3000, 10)
        prompt_score = {
            "prompt": prompt,
            "train_score": train_score,
            "dev_score": dev_score
        }
        return prompt_score

    @abstractmethod
    def sample_prompt(self, i=None):
        """
        Var
            i: int
                用來 sample prompt 的 index
        
        ---
        Return
            ttl_name: list[str]
                prompt 的名字

            ttl_pair: list[dict]
                dict formation:
                {
                    "prompt": str,
                    "train_score": float,
                    "dev_score": float
                }
        """
        pass

    @abstractmethod
    def update(self, new_population):
        """ 更新 self.population
        """
        pass

    def f(self):
        new_population = []
        for i in range(self.size_population):
            ttl_name, ttl_pair = self.sample_prompt(i)

            # evolution
            ttl_prompt = [ pair["prompt"] for pair in ttl_pair ]
            prompt_4_create_new_os_prompt = get_prompt.create(self.type_evo, *ttl_prompt)

            for_debug = {
                name: pair["prompt"]
                for name, pair in zip(ttl_name, ttl_pair)
            }
            new_prompt = get_distinct_new_propmt(self.ttl_model[0], self.population, prompt_4_create_new_os_prompt, **for_debug)

            # record
            prompt_score = self.evaluate(new_prompt)
            info = {
                name: self.formulate_pair(pair)
                for name, pair in zip(ttl_name, ttl_pair)
            }
            prompt_score["info"] = info
            new_population.append(prompt_score)
        self.update(new_population)
    
    def get_population(self):
        return self.population.deepcopy()

class sub_EvoDE(EvoPrompt):
    
    def sample_prompt(self, i):
        """ outupt 要按照 template 的順序
        """
        pair_best = self.population[0]
        pair_i = self.population[i]

        remaining_pairs = [pair for pair in self.population if pair['prompt'] != pair_i['prompt']]
        pair_1, pair_2 = np.random.choice(remaining_pairs, size=2, replace=False)

        ttl_pair = [pair_1, pair_2, pair_best, pair_i]
        ttl_name = ["p_1", "p_2", "p_best", "p_i"]

        return ttl_name, ttl_pair
    
    def update(self, new_population):
        """ 兩個 population 對應比較，取 train_score 高者
        """
        for i in range(self.size_population):
            if new_population[i]["train_score"] > self.population[i]["train_score"]:
                self.population[i] = new_population[i]
        self.population = sorted(self.population,  key=lambda x: x['train_score'],  reverse=True)

class sub_EvoGA(EvoPrompt):

    def sample_prompt(self):
        ttl_score = [pair['train_score'] for pair in self.population]
        ttl_weight = get_weight(ttl_score)
        pair_1, pair_2 = np.random.choice(self.population, size=2, replace=False, p=ttl_weight)

        ttl_pair = [pair_1, pair_2]
        ttl_name = ["p_1", "p_2"]

        return ttl_name, ttl_pair
    
    def update(self, new_population):
        """ 兩個 population 中、所有 prompt 中，前 self.size_population 高分的 prompt
        """
        self.population = sorted(self.population + new_population,  key=lambda x: x['train_score'],  reverse=True)
        self.population = self.population[:self.size_population]
    
class sub_CoEvo(EvoPrompt):

    def sample_prompt(self, i):
        """
        """
        pair_worst = self.population[-1]
        pair_init = np.random.choice(self.init_population)

        for_debug ={
            "p_worst": pair_worst["prompt"],
            "p_init": pair_init["prompt"]
        }
        prompt_4_create_p_init = get_prompt.create("EvoGA", pair_worst["prompt"], pair_init["prompt"])
        # 之前的經驗是 GA 做到最後也會重複，重複的話就相當於在 old_population 裡抽 p_, p_best
        # 又 currnet population 跟 init_population 同源
        # 所以欣璇覺得極有可能重複，不能單用 get_new_prompt
        p_contr = get_distinct_new_propmt(self.ttl_model[0], self.population, prompt_4_create_p_init, **for_debug)
        pair_contr = self.evaluate(p_contr)

        pair_i = self.population[i]


        ttl_pair = [pair_worst, pair_init, pair_contr, pair_i]
        ttl_name = ["p_worst", "p_init", "p_contr", "p_i"]

        return ttl_name, ttl_pair
    
    def update(self, new_population):
        """ 兩個 population 對應比較，取 train_score 高者
        """
        for i in range(self.size_population):
            if new_population[i]["train_score"] > self.population[i]["train_score"]:
                self.population[i] = new_population[i]
        self.population = sorted(self.population,  key=lambda x: x['train_score'],  reverse=True)