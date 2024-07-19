"""
main funcition

load_dotenv(".env") 放在最前面這樣其他套件最前面才能載到 config
"""
from dotenv import load_dotenv
load_dotenv(".env")

import configparser
import json
import os

from utils.ReSS import ReSS
from utils.call_model.embedding import Encoder
# from utils.prompt import get_prompt
from utils.call_model.llm import get_api
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
        ttl_model: (get_llm_reply, embedding_model)
            get_llm_reply: function
            embedding_model: Encoder in utils.call_model.embedding
    """

    # 指定 llm
    get_llm_reply=get_api(type_llm)

    # 指定 embedding model
    if type_embedding is None:
        embedding_model = None
    else:
        embedding_model = Encoder(type_embedding)

    ttl_model = (get_llm_reply, embedding_model)

    # 指定訓練資料
    num_training_data = 3
    with open(path_data,  'r', encoding='utf-8') as file:
        data = json.load(file)
    training_data=data[-num_training_data:]

    # 指定起始的 prompt
    with open(path_prompt,  'r') as file:
        ttl_prompt = json.load(file)

    # 製作 prompt-scores pairs
    prompt_scores=[]
    for prompt in ttl_prompt:
        score = get_score(ttl_model, prompt, training_data, 3000, 10)
        prompt_scores.append({'prompt': prompt,  'score': score})

    return ttl_model, training_data, prompt_scores

def main():

    # initialization
    path_data = CONFIG["datapath"]["Final_Quality"]
    path_prompt = CONFIG["datapath"]["init_os_prompt"]
    type_llm = "Breeze"
    # type_embedding = "multi-qa-mpnet-base-dot-v1"
    type_embedding = "bgem3"
    # type_embedding = None
    
    init_set = init_setting(type_llm, type_embedding, path_data, path_prompt)

    # # 停止條件
    path_stop_file = os.path.join(os.getcwd(), 'stop_true.txt') # 人工 early stop 的檔案位置
    ttl_pair_os_prompt_scores = ReSS(init_set, path_stop_file)
    sorted_pire = sorted(ttl_pair_os_prompt_scores,  key=lambda x: x['score'],  reverse=True)

    # 儲存結果
    t = time_now()
    path_folder = CONFIG["datapath"]["record_folder"]
    file_path = f"{path_folder}/{t}.json"
    data = {
        "corpus": get_file_name(path_data),
        "type_llm": type_llm,
        "type_embedding": type_embedding,
        "best_promt": sorted_pire[0],
        "record": ttl_pair_os_prompt_scores
    }
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"\n\n\nthe result is saved at:\n{file_path}")

if __name__ == "__main__":
    main()
