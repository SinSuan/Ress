import requests
import json
import configparser
from .Breeze_API import api_breeze

CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

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
