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
    if DEBUGGER=="True":
        print("enter api_breeze")

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
        j_result=json.loads(response.text)
        if "generated_text" in j_result.keys():
            r = j_result['generated_text']
        else:
            r = j_result
    except Exception as e:
        print(f"Except in api_breeze = {e}")

        print(f"r = \n{r}")
    if DEBUGGER=="True":
        print("exit api_breeze")
    return r

def get_api(model_name):
    if DEBUGGER=="True":
        print("enter get_api")

    if model_name=="Breeze":
        get_llm_reply=api_breeze
    # elif model_name=="Taide":
    #     from api.Taide3_API import get_taide3
    #     get_llm_reply=get_taide3
    # elif model_name=="ChatGPT":
    #     from api.ChatGPT_API import get_reply_16k
    #     get_llm_reply=get_reply_16k
    # elif model_name=="Llama3":
    #     from api.Llama3_API import get_llama3
    #     get_llm_reply=get_llama3
    # elif model_name=="Mistral":
    #     from api.Mistral2_API import get_mistral
    #     get_llm_reply=get_mistral
    else:
        print("未選擇模型!")
    print(f"目前使用模型為:{model_name}")

    if DEBUGGER=="True":
        print("exit get_api")
    return get_llm_reply
