"""
計算得分
紀錄回答情況
抓取 timeout 漏答的題目
"""

import configparser
import os
from tqdm.auto import tqdm
from utils.method import os_ap_sss_answer

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]


def pick_up(reply_llm):
    """ 一個選擇題有四個選項，這個函數是要從回答中抓出選項編號
    """
    if DEBUGGER=="True": print("enter pick_up")

    # 預設答錯
    idx_option = 5

    # 檢查有沒有答對
    for idx in range(1, 5):
        if str(idx) in reply_llm:
            idx_option = idx
            break

    if DEBUGGER=="True": print(f"exit pick_up,  {idx_option}")
    return idx_option

def answer(ttl_model, os_prompt, data, size_chunck, num_overlap):
    if DEBUGGER=="True": print("enter answer")
    try:
        result = os_ap_sss_answer(ttl_model, os_prompt, data, size_chunck, num_overlap)
        return result
    except Exception as e:
        print(f"Exception in answer:\n{e}")
        return "redo"

def experiment(ttl_model, os_prompt, ttl_data, size_chunck, num_overlap):
    """
    Var
        ttl_model: (llm, embedding_model)
            llm: function
            embedding_model: Encoder in utils.call.embedding
    
    應修正: my_result 的格式
    """

    my_result = []    # 輸出結果
    ttl_data_unanswer = []   # 發生錯誤的資料
    score = 0    # 總分數
    num = 0   # 資料計數器

    for data in tqdm(ttl_data):
        # try:
        #     # 輸入的內容與答案

        #     result = os_ap_sss_answer(ttl_model, os_prompt, data, size_chunck, num_overlap)

        #     # 對答案、計分數
        #     truth_answer_number = data['answer']
        #     if pick_up(result)==truth_answer_number:
        #         score += 1
        #     num += 1

        #     # print出目前得分正確率
        #     # print(f"score:{score}/{num} :({score/num*100}%)")
        #     my_result.append(result)

        # except Exception as e:
        #     # 之前 call api 偶爾會發生網路 timeout 讓我程式卡住，所以才設的
        #     print(e)
        #     ttl_data_unanswer.append(data)
        #     print(f"目前漏答數量:{len(ttl_data_unanswer)}")
        
        result = answer(ttl_model, os_prompt, data, size_chunck, num_overlap)
        while(result=="redo"):
            result = answer(ttl_model, os_prompt, data, size_chunck, num_overlap)
        # 對答案、計分數
        truth_answer_number = data['answer']
        if pick_up(result)==truth_answer_number:
            score += 1
        num += 1

        # print出目前得分正確率
        # print(f"score:{score}/{num} :({score/num*100}%)")
        my_result.append(result)
        
        

    return score, num, my_result, ttl_data_unanswer

def get_score(ttl_model, os_prompt, input_data, size_chunck, num_overlap):
    score,  _,  _,  _ = experiment(ttl_model, os_prompt, input_data, size_chunck, num_overlap)
    return score
