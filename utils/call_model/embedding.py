"""
unify the form to call embedding model
"""

import configparser
import json
import os
from typing import List

import requests
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

URL_BGEM3 = CONFIG["embedding"]["bgem3"]

def api_bgem3(text):
    """ call bge-m3 from lab
    """
    if DEBUGGER=="True": print("enter get_embedding")

    payload = json.dumps({
        "input": f"{text}",
        "model": "bge-m3",
        "user": "null"
    })
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", URL_BGEM3, headers=headers, data=payload, timeout=120)
    response = json.loads(response.text)
    embedding = response['data'][0]['embedding']

    if DEBUGGER=="True": print("exit get_embedding")
    return embedding

class Bgem3:
    """ bge-m3 from lab
    """
    def __init__(self) -> None:
        self.model = api_bgem3

    def encode(self, ttl_sentence: List[str])->List:
        ttl_embedding = []
        for sentence in ttl_sentence:
            # bgem3 無法處理空字串
            if sentence=="":
                sentence = " "
            embedding = self.model(sentence)
            ttl_embedding.append(embedding)
        ttl_embedding = np.array(ttl_embedding)
        return ttl_embedding

class OtherEmdedding:
    """ other embedding model downloaded from SentenceTransformer
    """
    def __init__(self, type_model:str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(type_model).to(device)

    def encode(self, ttl_sentence):
        """ 很多餘，但不打這個的話 Encoder 會跑不動
        """
        ttl_embedding = self.model.encode(ttl_sentence)
        return ttl_embedding

class Encoder:
    """ unify the form to call embedding model
    """
    def __init__(self, type_model:str) -> None:
        """
        Var
            type_model:
                "bgem3"
                others  ex: "multi-qa-mpnet-base-dot-v1"
        """

        if type_model=="bgem3":
            self.model = Bgem3()
        else:
            self.model = OtherEmdedding(type_model)

    def encode(self, ttl_sentence: List[str])->List:
        """ encode a list of strings at once
        """
        if DEBUGGER=="True": print("enter Encoder.encode")

        ttl_embedding = self.model.encode(ttl_sentence)

        if DEBUGGER=="True": print("exit Encoder.encode")
        return ttl_embedding
