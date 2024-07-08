import configparser
from sentence_transformers import util

# argument type
from typing import List

from .tools import count_words
from .call_model.embedding import Encoder

CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

# get_ttl_idx_check
def get_adjacent_similarity(embedding_sentences):
    """ 以內積計算句子倆倆相似度
    """
    if DEBUGGER=="True":
        print("enter get_adjacent_similarity")

    ttl_similarity = []
    for i in range(len(embedding_sentences) - 1):
        similarity_of_adjacent = util.dot_score(embedding_sentences[i], embedding_sentences[i + 1])
        similarity_of_adjacent = similarity_of_adjacent.item()  # tensor of torch.float64
        ttl_similarity.append(similarity_of_adjacent)   # float

    if DEBUGGER=="True":
        print("exit get_adjacent_similarity")
    return ttl_similarity

def find_idx_low_pick(arr):
    """ low pick indicates that
    the similarity between two adjacent sentences is lower than that between the other neighbors
    """
    if DEBUGGER=="True":
        print("enter find_idx_low_pick")

    ttl_idx_low_pick = []
    for idx in range(1, len(arr) - 1):
        if arr[idx] < min(arr[idx - 1], arr[idx + 1]):
            ttl_idx_low_pick.append(idx)

    if DEBUGGER=="True":
        print("exit find_idx_low_pick")
    return ttl_idx_low_pick

def get_ttl_idx_check(sentences, embedding_model: Encoder=None)->List[int]:
    if DEBUGGER=="True":
        print("enter get_ttl_idx_check")

    if embedding_model is None:   # split_with_overlap_english
        ttl_idx_check = range(len(sentences))

    else:   # Semantic_Sentence_Split
        embedding_sentences = embedding_model.encode(sentences)
        ttl_similarity = get_adjacent_similarity(embedding_sentences)
        ttl_idx_check = find_idx_low_pick(ttl_similarity)

    if DEBUGGER=="True":
        print("exit get_ttl_idx_check")
    return ttl_idx_check

# create_ttl_chunk
def find_low_pick_4_next_chunk(latter_part_sentences, latter_part_ttl_idx_check, chunk_size):
    """
    Var
        latter_part_sentences: List[str]
            split document
            ( part_sentences[idx_start:] )
            
        latter_part_ttl_idx_check: List[int]
            the index of the sentence that need to be chunked
            ( ttl_idx_check[i:] )

        chunk_size: int
            the number of words in each chunk
    """
    if DEBUGGER=="True":
        print("enter find_low_pick_4_next_chunk")

    num_latter_part_idx_check = len(latter_part_ttl_idx_check)

    # 找這個 chunk 的 idx_end
    i_diff = 0
    idx_end = 0
    temp_sentences_observed = latter_part_sentences[:0]
    # second while-loop
    while((i_diff<num_latter_part_idx_check) and (count_words(temp_sentences_observed)<chunk_size)):
        # print(0)
        idx_end = latter_part_ttl_idx_check[i_diff] + 1
        temp_sentences_observed = latter_part_sentences[:idx_end]
        i_diff += 1
        # print(1)

    if DEBUGGER=="True":
        print("exit find_low_pick_4_next_chunk")
    return i_diff

def create_single_chunk(
        sentences,
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
    if DEBUGGER=="True":
        print("enter create_single_chunk")

    idx_start, idx_end = pre_post_idx
    sentences_observed = sentences[idx_start:idx_end]
    sentences_4_chunk = overlap_pre + sentences_observed + overlap_post
    chunk = ". ".join(sentences_4_chunk)

    if DEBUGGER=="True":
        print("exit create_single_chunk")
    return chunk    

def create_ttl_chunk(sentences, ttl_idx_check, chunk_size=3000, overlap=10)->List[str]:
    """
    Var
        sentences: List[str]
            split document
            
        ttl_idx_check: List[int]
            the index of the sentence that need to be chunked

        chunk_size: int
            the number of words in each chunk
        
        overlap: int
            the number of sentences that overlap between chunks

    Return
        List[str]: List of chunks
    """
    if DEBUGGER=="True":
        print("enter Sentence_Split")

    num_idx_check = len(ttl_idx_check)

    i = 0
    idx_start = 0
    idx_end = 0
    overlap_pre = []    # 前overlap句
    overlap_post = []   # 後overlap句
    ttl_chunk = []  # 所有 chunk

    # first while-loop
    while((i<num_idx_check) and (idx_end+overlap<len(sentences))):   # 有下一個句子

        # 找這個 chunk 的 idx_end
        latter_part_sentences = sentences[idx_start:]
        latter_part_ttl_idx_check = ttl_idx_check[i:]
        i += find_low_pick_4_next_chunk(latter_part_sentences, latter_part_ttl_idx_check, chunk_size)

        # 出 while-loop 的時候 > chunk_size 了，回到上一個 check (ex: pick)
        idx_end = ttl_idx_check[i-1] + 1

        # 組合 chunk
        overlap_post = sentences[idx_end:idx_end+overlap]
        chunk = create_single_chunk(sentences, (idx_start, idx_end), overlap_pre, overlap_post)
        ttl_chunk.append(chunk)

        # 更新參數
        idx_start = idx_end
        overlap_pre = sentences[idx_start-overlap:idx_start]

    # 補做最後一個 chunk
    if idx_end+overlap<len(sentences)-1:
        # 組合 chunk
        chunk = create_single_chunk(sentences, (idx_start, None), overlap_pre, [])
        ttl_chunk.append(chunk)

    if DEBUGGER=="True":
        print("exit Sentence_Split")
    return ttl_chunk

# get_ttl_chunk
def get_ttl_chunk(text, chunk_size=3000, overlap=10, embedding_model: Encoder=None)->List[str]:
    """
    Var
        text: str
            raw document
    
        chunk_size: int
            the number of words in each chunk
        
        overlap: int
            the number of sentences that overlap between chunks
        
        model: Encoder
            None: use split_with_overlap_english
            Encoder: use Semantic_Sentence_Split
    """
    if DEBUGGER=="True":
        print("enter get_ttl_chunk")

    # 將句子以句號分割
    sentences = text.split(".")
    # 將句子進行encode
    ttl_idx_check = get_ttl_idx_check(sentences, embedding_model)
    ttl_chunk = create_ttl_chunk(sentences, ttl_idx_check, chunk_size, overlap)

    if DEBUGGER=="True":
        print("exit get_ttl_chunk")
    return ttl_chunk
