import configparser
import json
import requests

CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]
URL = CONFIG["breeze"]["lab"]
SUGGEST_SYSTEM_PROMPT = CONFIG["breeze"]["SUGGEST_SYSTEM_PROMPT"]

def api_breeze(input_content, max_new_tokens=4096, temperature=None):
# def api_breeze(input_content, max_new_tokens=4096, temperature=None):
    if DEBUGGER=="True":
        print("enter api_breeze")

    payload_input=f"<s> {SUGGEST_SYSTEM_PROMPT} [INST] {input_content} [/INST]"
    parameter = {
        "do_sample": False,
        "max_new_tokens":max_new_tokens,
    }
    if temperature is not None:
        parameter["temperature"] = temperature

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
        print(f"\texcept Exception = {e}")
        print(e)
    
    if DEBUGGER=="True":
        print("exit api_breeze")
    return r
