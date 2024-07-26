"""
論文的主要方法
"""

import configparser
import os
from .prompt import get_prompt
from .split_into_chunk import get_ttl_chunk

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]


def os_ap_sss_answer(ttl_model, os_prompt, data, size_chunck = 3000, num_overlap = 10):
    """
    Var
        ttl_model: (llm, embedding_model)
            llm: function
            embedding_model: Encoder in utils.call.embedding

        size_chunck: int
            the number of words in each chunk
            
        num_overlap: int
            the number of senetences that overlap between two chunks
    """
    if DEBUGGER=="True": print("enter os_ap_sss_answer")

    content = data['content']   # 參考文章
    question = data['question'] # 問題(多選題)
    llm, embedding_model = ttl_model

    # 切分摘要完要輸入給llm的內容
    new_content = content
    while len(new_content.split(" "))>size_chunck:
        
        ttl_chunck = get_ttl_chunk(new_content, size_chunck, num_overlap, embedding_model)
        # 這一輪的新內容
        new_content = ""
        for chunk in ttl_chunck:

            # 請llm幫我們把重要資訊留下
            prompt_4_summarize_chunk =  get_prompt.sum("old", chunk, os_prompt, question)
            chunk_summary = llm.reply(prompt_4_summarize_chunk)
            # if chunk_summary==None:
            #     continue
            new_content+= chunk_summary+" "
        
        # 防錯(如果LLM api無回傳 直接比照truncate)
        if new_content=="":
            new_content = " ".join(content.split(" ")[:3000])
            break

    # 找完有用的內容後，進行問答
    prompt_4_exam_multichoice = get_prompt.exam(new_content,  question)
    answer_from_llm = llm.reply(prompt_4_exam_multichoice)

    if DEBUGGER=="True": print("exit os_ap_sss_answer")
    return answer_from_llm
