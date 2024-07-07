import configparser
import json
from typing import List

import requests
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
URL = CONFIG["embedding"]["embedding"]
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

def api_bgem3(text):
    """ call bge-m3 from lab
    """
    if DEBUGGER=="True":
        print("enter get_embedding")

    payload = json.dumps({
        "input": f"{text}",
        "model": "bge-m3",
        "user": "null"
    })
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", URL, headers=headers, data=payload, timeout=120)
    response = json.loads(response.text)
    embedding = response['data'][0]['embedding']

    if DEBUGGER=="True":
        print("exit get_embedding")
    return embedding

class Bgem3:
    def __init__(self) -> None:
        self.model = api_bgem3

    def encode(self, sentences: List[str])->List:
        embedding_sentences = []
        # for sentence in sentences:
        for sentence in sentences:
            # bgem3 無法處理空字串
            if sentence=="":
                sentence = " "
            embedding_sentence = self.model(sentence)
            embedding_sentences.append(embedding_sentence)
        embedding_sentences = np.array(embedding_sentences)
        return embedding_sentences

class Other_Emdedding:
    def __init__(self, type_model:str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(type_model).to(device)

    def encode(self, sentences):
        """ 很多餘，但不打這個的話 encoder 會跑不動
        """
        embedding_sentences = self.model.encode(sentences)
        return embedding_sentences

class Encoder:
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
            self.model = Other_Emdedding(type_model)

    def encode(self, sentences: List[str])->List:
        embedding_sentences = self.model.encode(sentences)
        return embedding_sentences
