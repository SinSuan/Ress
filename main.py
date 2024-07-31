"""
main funcition

load_dotenv(".env") 放在最前面這樣其他套件最前面才能載到 config
"""
import configparser
import json
import os
import random

from dotenv import load_dotenv
load_dotenv(".env")

from utils.EvoPrompt import EvoDE, EvoGA
from utils.ReSS import ReSS
from utils.call_model.embedding import Encoder
# from utils.prompt import get_prompt
from utils.call_model.llm import LLM
from utils.score import get_score
from utils.tools import get_file_name, prompt_in_list, time_now

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)

DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

def init_setting(type_llm: str, type_embedding: str, path_data: str, path_prompt: str):
    """
    Var
        all are string

    Return
        ttl_model: (llm, embedding_model)
            llm: function
            embedding_model: Encoder in utils.call_model.embedding
    """

    # 指定 llm
    llm = LLM(type_llm)

    # 指定 embedding model
    if type_embedding is None:
        embedding_model = None
    else:
        embedding_model = Encoder(type_embedding)

    ttl_model = (llm, embedding_model)

    # 指定訓練資料
    num_training_data = 180
    with open(path_data,  'r', encoding='utf-8') as file:
        data = json.load(file)
    # training_data=data[-num_training_data:]
    dataset = random.sample(data, num_training_data)
    train_split = dataset[:150]
    dev_split = dataset[150:]
    ttl_dataset = {
        "train_split": train_split,
        "dev_split": dev_split,
    }

    # 指定起始的 prompt
    with open(path_prompt,  'r') as file:
        ttl_prompt = json.load(file)

    # 製作 prompt-scores pairs
    ttl_prompt_scores=[]
    for prompt in ttl_prompt:
        # print(len(train_split))
        # print(len(dev_split))
        train_score = get_score(ttl_model, prompt, train_split, 3000, 10)
        dev_score = get_score(ttl_model, prompt, dev_split, 3000, 10)
        prompt_scores = {
            'prompt': prompt,
            'train_score': train_score,
            'dev_score': dev_score
        }
        ttl_prompt_scores.append(prompt_scores)

    return ttl_model, ttl_dataset, ttl_prompt_scores

def main():

    # initialization
    path_data = CONFIG["datapath"]["Final_Quality"]
    print(f"{path_data=}")
    print(f"get_file_name(path_data)={get_file_name(path_data)}")
    path_prompt = CONFIG["datapath"]["init_os_prompt"]
    type_llm = "Breeze"
    # type_embedding = "multi-qa-mpnet-base-dot-v1"
    type_embedding = "bgem3"
    # type_embedding = None
    
    ttl_model, ttl_dataset, ttl_pair_os_prompt_scores = init_setting(type_llm, type_embedding, path_data, path_prompt)

    # # 停止條件
    path_stop_file = os.path.join(os.getcwd(), 'stop_true.txt') # 人工 early stop 的檔案位置
    stop_score=100  # 練蠱終止條件(我是設超過baseline做100題後的分數，這邊看你資料量來設)
    # stop_run_num=20   # 或是設一個回合數來終止(本來我會讓他跑到天荒地老所以沒有用for loop)
    stop_run_num=0
    
    sorted_pair = sorted(ttl_pair_os_prompt_scores,  key=lambda x: x['train_score'],  reverse=True)
    while(
        sorted_pair[0]['train_score']<stop_score
        and stop_run_num>=0
        and not os.path.exists(path_stop_file)   # 人工 early stop
    ):
        # new_population = EvoDE(ttl_model, ttl_dataset, sorted_pair)
        new_population = ReSS(ttl_model, ttl_dataset, sorted_pair)
        sorted_pair = sorted(new_population,  key=lambda x: x['train_score'],  reverse=True)
        stop_run_num -= 1

    # 儲存結果
    t = time_now()
    path_folder = CONFIG["datapath"]["record_folder"]
    file_path = f"{path_folder}/{t}.json"
    data = {
        "corpus": get_file_name(path_data),
        "type_llm": type_llm,
        "type_embedding": type_embedding,
        "best_promt": sorted_pair[0],
        "record": sorted_pair
    }
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"\n\n\nthe result is saved at:\n{file_path}")

if __name__ == "__main__":
    main()
