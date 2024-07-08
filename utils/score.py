
from tqdm.auto import tqdm

from utils.method import os_ap_sss_answer
import configparser
CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

def pick_up(reply_llm):
    if DEBUGGER=="True":
        print("enter pick_up")
    
    # 預設答錯
    idx_option = 5

    # 檢查有沒有答對
    for i in range(1, 5):
        if str(i) in reply_llm:
            idx_option = i
            break
    
    if DEBUGGER=="True":
        print(f"exit pick_up,  {idx_option}")
    return idx_option

def experiment(input_data, os_prompt, chunck_size, overlap,  embedding_model=None):
    """
    Var
        embedding_model: Encoder in utils.embedding
    """
    
    my_result=[]    # 輸出結果
    error_list=[]   # 錯誤資料
    score=0    # 總分數
    num=0   # 資料計數器

    for each_data in tqdm(input_data):
        try:
            # 輸入的內容與答案
            
            result=os_ap_sss_answer(os_prompt, each_data, chunck_size, overlap,  embedding_model)
            
            # 對答案、計分數
            truth_answer_number=each_data['answer']
            if pick_up(result)==truth_answer_number:
                score+=1
            num+=1

            # print出目前得分正確率
            print(f"score:{score}/{num} :({score/num*100}%)")
            my_result.append(result)
            
        except Exception as e:
            """ 
            之前 call api 偶爾會有網路 timeout 的問題
            讓我程式卡住才設的
            """
            error_list.append(each_data)
            print(e)
            print(f"目前漏答數量:{len(error_list)}")
    
    # return f"score:{scores}/{num} :({scores/num*100}%)", my_result, error_list
    return score,  num,  my_result,  error_list