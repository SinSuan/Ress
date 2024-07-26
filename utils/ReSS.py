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

def ReSS(init_set, path_stop_file):
    
    ttl_model, ttl_data, ttl_pair_os_prompt_scores = init_set
    get_llm_reply, _ = ttl_model

    # example_num=5 # 你要給LLM看幾個example
    example_num=2

    # 停止條件
    stop_score=100  # 練蠱終止條件(我是設超過baseline做100題後的分數，這邊看你資料量來設)
    # stop_run_num=20   # 或是設一個回合數來終止(本來我會讓他跑到天荒地老所以沒有用for loop)
    stop_run_num=2
    
    sorted_pair = sorted(ttl_pair_os_prompt_scores,  key=lambda x: x['score'],  reverse=True)
    while(
        sorted_pair[0]['score']<stop_score
        and stop_run_num>=0
        and not os.path.exists(path_stop_file)   # 人工 early stop
    ):

        # 整理example格式
        example=""
        for p in sorted_pair[:example_num]:
            example+=f"""[Old prompt]:"{p['prompt']}"\n[Scores]:{p['score']}\n\n"""

        # 製作 new_prompt
        prompt_4_create_new_os_prompt = get_prompt(2,  example)
        new_prompt=get_llm_reply(prompt_4_create_new_os_prompt, 4096, 1)

        # 紀錄
        print("="*50)
        print(new_prompt)
        if prompt_in_list(ttl_pair_os_prompt_scores, new_prompt) is False:
            score = get_score(ttl_model, new_prompt, ttl_data, 3000, 10)
            print("*"*50)
            print(f"{new_prompt}\n{score}")
            prompt_score = {
                'prompt': new_prompt, 
                'score': score
            }
            ttl_pair_os_prompt_scores.append(prompt_score)
        else:
            print("prompt exist")

        # 更新參數
        stop_run_num-=1
        sorted_pair = sorted(ttl_pair_os_prompt_scores,  key=lambda x: x['score'],  reverse=True)

    return ttl_pair_os_prompt_scores

def ReSS_short(ttl_model, train_set, dev_set, sorted_pair):
    """
    Var:
        example_num: int
            你要給LLM看幾個example
    """
    example_num=2
    get_llm_reply, _ = ttl_model


    # 整理example格式
    example=""
    for p in sorted_pair[:example_num]:
        example+=f"""[Old prompt]:"{p['prompt']}"\n[Scores]:{p['score']}\n\n"""

    # 製作 new_prompt
    prompt_4_create_new_os_prompt = get_prompt(2,  example)
    new_prompt=get_llm_reply(prompt_4_create_new_os_prompt, 4096, 1)

    # 紀錄
    # print("="*50)
    # print(new_prompt)
    if prompt_in_list(sorted_pair, new_prompt) is False:
        train_score = get_score(ttl_model, new_prompt, train_set, 3000, 10)
        dev_score = get_score(ttl_model, new_prompt, dev_set, 3000, 10)
        prompt_score = {
            'prompt': new_prompt,
            'train_score': train_score,
            'dev_score': dev_score
        }
        sorted_pair.append(prompt_score)
    else:
        print("prompt exist")

    return sorted_pair