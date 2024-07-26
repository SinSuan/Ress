""" 信彰的方法
"""

import configparser
import os

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

def ReSS(ttl_model, ttl_dataset, sorted_pair):
    """
    Var:
        example_num: int
            你要給LLM看幾個example
    """
    if DEBUGGER=="True": print("enter ReSS")
    
    example_num=5
    llm, _ = ttl_model
    train_split = ttl_dataset["train_split"]
    dev_split = ttl_dataset["dev_split"]
    if dev_split==[]:
        dev_split = train_split

    # 整理example格式
    example=""
    for p in sorted_pair[:example_num]:
        example+=f"""[Old prompt]:"{p['prompt']}"\n[Scores]:{p['train_score']}\n\n"""

    # 製作 new_prompt
    prompt_4_create_new_os_prompt = get_prompt.create("ReSS",  example)
    new_prompt = llm.reply(prompt_4_create_new_os_prompt, 1)

    # 紀錄
    if prompt_in_list(sorted_pair, new_prompt) is False:
        train_score = get_score(ttl_model, new_prompt, train_split, 3000, 10)
        dev_score = get_score(ttl_model, new_prompt, dev_split, 3000, 10)
        prompt_score = {
            'prompt': new_prompt,
            'train_score': train_score,
            'dev_score': dev_score
        }
        sorted_pair.append(prompt_score)

    if DEBUGGER=="True": print("exit ReSS")
    return sorted_pair