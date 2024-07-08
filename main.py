import configparser
import json
import os

from utils.call_model.embedding import Encoder
from utils.prompt import get_prompt
from utils.call_model.llm import get_api
from utils.score import experiment
from utils.tools import prompt_in_list


CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
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
    if type_embedding == None:
        embedding_model = None
    else:
        embedding_model = Encoder(type_embedding)

    ttl_model = (get_llm_reply, embedding_model)

    # 指定訓練資料
    num_training_data = 3
    with open(path_data,  'r') as file:
        data = json.load(file)
    training_data=data[-num_training_data:]

    # 指定起始的 prompt
    with open(path_prompt,  'r') as file:
        prompt_list = json.load(file)

    # 製作 prompt、scores pairs
    prompt_scores=[]
    for prompt in prompt_list:
        score,  _,  _,  _ = experiment(ttl_model, prompt, training_data, 3000, 10)
        prompt_scores.append({'prompt': prompt,  'score': score})

    return ttl_model, training_data, prompt_scores

def main():

    # initialization
    path_data = '/user_data/itri/Ress/dataset/Final_Quality.json'
    path_prompt = '/user_data/itri/Ress/dataset/init_os_prompt_corpus.json'
    type_llm = "Breeze"
    type_embedding = "multi-qa-mpnet-base-dot-v1"
    ttl_model, training_data, prompt_scores = init_setting(type_llm, type_embedding, path_data, path_prompt)
    get_llm_reply, _ = ttl_model

    # example_num=5 # 你要給LLM看幾個example
    example_num=2

    # 停止條件
    stop_score=100  # 練蠱終止條件(我是設超過baseline做100題後的分數，這邊看你資料量來設)
    # stop_run_num=20   # 或是設一個回合數來終止(本來我會讓他跑到天荒地老所以沒有用for loop)
    stop_run_num=2
    path_stop_file = os.path.join(os.getcwd(), 'stop_true.txt') # 人工 early stoy 的位置

    # main loop
    stop_run_num-=1
    sorted_prompt_scores = sorted(prompt_scores,  key=lambda x: x['score'],  reverse=True)
    while(
        sorted_prompt_scores[0]['score']<stop_score
        and stop_run_num>=0
        and not os.path.exists(path_stop_file)   # 人工 early stoy
    ):

        # 整理example格式
        example=""
        for p in sorted_prompt_scores[:example_num]:
            example+=f"""[Old prompt]:"{p['prompt']}"\n[Scores]:{p['score']}\n\n"""

        # 製作 new_prompt
        prompt_4_create_new_os_prompt = get_prompt(2,  example)
        new_prompt=get_llm_reply(prompt_4_create_new_os_prompt, 4090, 1)

        # 紀錄
        print("="*50)
        print(new_prompt)
        if prompt_in_list(prompt_scores, new_prompt) is False:
            score,  _,  _,  _ = experiment(ttl_model, new_prompt, training_data, 3000, 10)
            print("*"*50)
            print(f"{new_prompt}\n{score}")
            prompt_score = {
                'prompt': new_prompt, 
                'score': score
            }
            prompt_scores.append(prompt_score)
        else:
            print("prompt exist")

        # 更新參數
        stop_run_num-=1
        sorted_prompt_scores = sorted(prompt_scores,  key=lambda x: x['score'],  reverse=True)

if __name__ == "__main__":
    
    main()
