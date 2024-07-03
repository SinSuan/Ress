import requests
import json


def get_api(model_name):
    if model_name=="Breeze":
        from api.Breeze_API import get_breeze
        get_llm_reply=get_breeze
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
    return get_llm_reply

def get_embedding(text):
    url = "http://140.120.13.248:5174/v1/embeddings"
    payload = json.dumps({
    "input": f"{text}",
    "model": "bge-m3",
    "user": "null"
    })
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = json.loads(response.text)
    # print(response['data'][0]['embedding'])
    return response['data'][0]['embedding']