"""
只有用到 embedding model
沒有用到 llm
"""

import configparser
import os
from typing import List
# from sentence_transformers import util
from .tools import count_words
from .call_model.embedding import Encoder
import numpy as np

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
# DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]
DEBUGGER = "False"

def normalized(v):
    len_squared = 0
    for vi in v:
        len_squared += vi**2
    return np.array(v)/(len_squared**(1/2))

def dot_product(v1,v2):
    length = len(v1)
    if len(v2)!= length:
        raise("wrong pair")
    else:
        norm_v1 = normalized(v1)
        norm_v2 = normalized(v2)
        return np.sum(norm_v1*norm_v2)

# get_ttl_idx_check
def get_adjacent_similarity(ttl_embedding):
    """ 以內積計算句子倆倆相似度
    """
    if DEBUGGER=="True": print("enter get_adjacent_similarity")

    ttl_similarity = []
    for i in range(len(ttl_embedding) - 1):
        # similarity_of_adjacent = util.dot_score(ttl_embedding[i], ttl_embedding[i + 1])
        similarity_of_adjacent = dot_product(ttl_embedding[i], ttl_embedding[i + 1])
        similarity_of_adjacent = similarity_of_adjacent.item()  # tensor of torch.float64
        ttl_similarity.append(similarity_of_adjacent)   # float

    if DEBUGGER=="True": print("exit get_adjacent_similarity")
    return ttl_similarity

def find_idx_low_peak(arr):
    """ low peak indicates that:
        the similarity between two adjacent sentences is lower than that between the other neighbors
    """
    if DEBUGGER=="True": print("enter find_idx_low_peak")

    ttl_idx_low_peak = []
    for idx in range(1, len(arr) - 1):
        if arr[idx] < min(arr[idx - 1], arr[idx + 1]):
            ttl_idx_low_peak.append(idx)

    if DEBUGGER=="True": print("exit find_idx_low_peak")
    return ttl_idx_low_peak

def get_ttl_idx_check(ttl_sentence, embedding_model: Encoder=None)->List[int]:
    if DEBUGGER=="True": print("enter get_ttl_idx_check")

    if embedding_model is None:   # split_with_overlap_english
        ttl_idx_check = range(len(ttl_sentence))

    else:   # Semantic_Sentence_Split
        ttl_embedding = embedding_model.encode(ttl_sentence)
        ttl_similarity = get_adjacent_similarity(ttl_embedding)
        ttl_idx_check = find_idx_low_peak(ttl_similarity)

    if DEBUGGER=="True": print("exit get_ttl_idx_check")
    return ttl_idx_check

# create_ttl_chunk
def find_low_peak_4_next_chunk(rest_sentence, rest_idx_check, size_chunk):
    """
    Var
        rest_sentence: List[str]
            split document
            ( part_sentences[idx_start:] )
            
        rest_idx_check: List[int]
            the index of the sentence that need to be chunked
            ( ttl_idx_check[i:] )

        size_chunk: int
            the number of words in each chunk
    """
    if DEBUGGER=="True": print("enter find_low_peak_4_next_chunk")

    num_rest_idx_check = len(rest_idx_check)

    # 找這個 chunk 的 idx_end
    i_diff = 0
    idx_end = 0
    temp_sentence_sublist = rest_sentence[:0]
    # second while-loop
    while((i_diff<num_rest_idx_check) and (count_words(temp_sentence_sublist)<size_chunk)):
        # print(0)
        idx_end = rest_idx_check[i_diff] + 1
        temp_sentence_sublist = rest_sentence[:idx_end]
        i_diff += 1
        # print(1)

    if DEBUGGER=="True": print("exit find_low_peak_4_next_chunk")
    return i_diff

def create_single_chunk(
        ttl_sentence,
        pre_post_idx: tuple,
        overlap_pre: List[str]=[],
        overlap_post: List[str]=[]
):
    """
    Var
        sentences: List[str]
            split document
    
        pre_post_idx: tuple
            (idx_start, idx_end) of the sentence that need to be chunked
        
        overlap_pre: List[str]
            the sentences that overlap before the chunk
        
        overlap_post: List[str]
            the sentences that overlap after the chunk
    
    Return
        str: a chunk
    """
    if DEBUGGER=="True": print("enter create_single_chunk")

    idx_start, idx_end = pre_post_idx
    sentence_sublist = ttl_sentence[idx_start:idx_end]
    sentences_4_chunk = overlap_pre + sentence_sublist + overlap_post
    chunk = ". ".join(sentences_4_chunk)

    if DEBUGGER=="True": print("exit create_single_chunk")
    return chunk

def create_ttl_chunk(ttl_sentence, ttl_idx_check, size_chunk=3000, num_overlap=10)->List[str]:
    """
    Var
        sentences: List[str]
            split document
            
        ttl_idx_check: List[int]
            the index of the sentence that need to be chunked

        size_chunk: int
            the number of words in each chunk
        
        num_overlap: int
            the number of sentences that overlap between chunks

    Return
        List[str]: List of chunks
    """
    if DEBUGGER=="True": print("enter Sentence_Split")

    num_idx_check = len(ttl_idx_check)

    i = 0
    idx_start = 0
    idx_end = 0
    overlap_pre = []    # 前overlap句
    overlap_post = []   # 後overlap句
    ttl_chunk = []  # 所有 chunk

    # first while-loop
    while((i<num_idx_check) and (idx_end+num_overlap<len(ttl_sentence))):   # 有下一個句子

        # 找這個 chunk 的 idx_end
        rest_sentences = ttl_sentence[idx_start:]
        rest_ttl_idx_check = ttl_idx_check[i:]
        i += find_low_peak_4_next_chunk(rest_sentences, rest_ttl_idx_check, size_chunk)

        # 出 while-loop 的時候 > size_chunk 了，回到上一個 check (ex: peak)
        idx_end = ttl_idx_check[i-1] + 1

        # 組合 chunk
        overlap_post = ttl_sentence[idx_end:idx_end+num_overlap]
        chunk = create_single_chunk(ttl_sentence, (idx_start, idx_end), overlap_pre, overlap_post)
        ttl_chunk.append(chunk)

        # 更新參數
        idx_start = idx_end
        overlap_pre = ttl_sentence[idx_start-num_overlap:idx_start]

    # 補做最後一個 chunk
    if idx_end+num_overlap<len(ttl_sentence)-1:
        # 組合 chunk
        chunk = create_single_chunk(ttl_sentence, (idx_start, None), overlap_pre, [])
        ttl_chunk.append(chunk)

    if DEBUGGER=="True": print("exit Sentence_Split")
    return ttl_chunk

# get_ttl_chunk
def get_ttl_chunk(text, size_chunk=3000, num_overlap=10, embedding_model: Encoder=None)->List[str]:
    """
    Var
        text: str
            raw document
    
        size_chunk: int
            the number of words in each chunk
        
        num_overlap: int
            the number of sentences that overlap between chunks
        
        model: Encoder
            None: use split_with_overlap_english
            Encoder: use Semantic_Sentence_Split
    """
    if DEBUGGER=="True": print("enter get_ttl_chunk")

    # 將句子以句號分割
    ttl_sentence = text.split(".")
    # 將句子進行encode
    ttl_idx_check = get_ttl_idx_check(ttl_sentence, embedding_model)
    ttl_chunk = create_ttl_chunk(ttl_sentence, ttl_idx_check, size_chunk, num_overlap)

    if DEBUGGER=="True": print("exit get_ttl_chunk")
    return ttl_chunk
