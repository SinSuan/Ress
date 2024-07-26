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

def api_breeze(user_prompt, max_new_tokens=4096, temperature=None):
    if DEBUGGER=="True": print("enter api_breeze")

    payload_input=f"<s> {SUGGEST_SYSTEM_PROMPT} [INST] {user_prompt} [/INST]"
    parameter = {
        "do_sample": temperature is not None,
        "max_new_tokens":max_new_tokens,
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
        response = requests.request("POST", URL, headers=headers, data=payload, timeout=120)
        print(f"{response=}")
        j_result=json.loads(response.text)
        print(f"{j_result=}")
        if "generated_text" in j_result.keys():
            r = j_result['generated_text']
        else:
            r = j_result
    except Exception as e:
        print(f"Except in api_breeze = {e}")

        print(f"r = \n{r}")
    if DEBUGGER=="True": print("exit api_breeze")
    return r

class LLM:
    
    def __init__(self, type_llm) -> None:
        if type_llm=="Breeze":
            self.model = api_breeze
    
    def reply(self, *args):
        result = self.model(*args)
        return result
