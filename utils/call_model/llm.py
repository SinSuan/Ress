import configparser
import json
import os
import requests

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
CONFIG.read(PATH_CONFIG, encoding='utf-8')
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

URL = CONFIG["breeze"]["lab"]
SUGGEST_SYSTEM_PROMPT = CONFIG["breeze"]["SUGGEST_SYSTEM_PROMPT"]

def count_token(user_prompt):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": user_prompt,
        "parameters": {
            "max_new_tokens": 0
        }
    }
    try:
        URL_TOKENIZER = 'https://nlplab-llm.nlpnchu.org/tokenize'
        response = requests.post(URL_TOKENIZER, headers=headers, data=json.dumps(data))
        # print(f"{response=}")
        response_data = response.json()
        # print(f"{response_data=}")
        num_token = len(response_data)
    except:
        num_token = None
        pass
    
    return num_token

def api_breeze(user_prompt, temperature=None, max_new_tokens=1000):
# def api_breeze(user_prompt, temperature=None, max_new_tokens=1000):
    """
    Var
        max_new_tokens: int
            預設為 None，代表不限制（這個tgi max_tokens 強制固定為 8192）
    """
    if DEBUGGER=="True": print("enter api_breeze")
    # print(f"{user_prompt=}")
    payload_input=f"<s> {SUGGEST_SYSTEM_PROMPT} [INST] {user_prompt} [/INST]"
    # print(f"{payload_input=}")
    num_token_input = count_token(payload_input)
    # print(f"{num_token_input=}")
    max_new_tokens = min(max_new_tokens, 8192-num_token_input)
    # print(f"{max_new_tokens=}")
    
    parameter = {
        "do_sample": temperature is not None,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }

    payload = json.dumps({
        "inputs": payload_input,
        "parameters": parameter
    })
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    try:
        # response = requests.request("POST", URL, headers=headers, data=payload, timeout=120)
        # URL = "https://nlplab-llm.nlpnchu.org/generate_stream"
        response = requests.request("POST", URL, headers=headers, data=payload, stream=True)
        # print(f"{response=}")
        # print(f"{response.text=}")
        j_result=json.loads(response.text)
        if "generated_text" in j_result.keys():
            r = j_result['generated_text']
        else:
            r = j_result
    except Exception as e:
        print(f"{user_prompt=}")
        print(f"Except in api_breeze =\n{e}")
        raise(e)
    
    
    # response = requests.request("POST", URL, headers=headers, data=payload, timeout=120)
    # j_result=json.loads(response.text)
    # r = j_result['generated_text']

    
    if DEBUGGER=="True": print("exit api_breeze")
    return r

class LLM:
    
    def __init__(self, type_llm) -> None:
        if type_llm=="Breeze":
            self.model = api_breeze
    
    def reply(self, *args):
        result = self.model(*args)
        return result
