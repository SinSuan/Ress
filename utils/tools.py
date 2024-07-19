""" other functions
"""

import configparser
import os

import pytz
from datetime import datetime

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

def count_words(sentences):
    """
    Var
        sentences: List[str]
            raw document
    
    Return
        int: the number of words in thees sentences
    """
    if DEBUGGER=="True":
        print("enter count_words")

    count = 0
    for sentence in sentences:
        count += len(sentence.split(" "))

    if DEBUGGER=="True":
        print("exit count_words")
    return count

def prompt_in_list(ttl_pire_prompt_score, new_prompt):
    """
    Var
        ttl_pire_prompt_score: List[Dict]
            list of "{'prompt': prompt, 'score': score}"

        new_prompt: str
            the prompt to be check whether or not the in ttl_prompt
    """
    if DEBUGGER=="True":
        print("enter prompt_in_list")

    # 預設不在
    in_list = False
    # 然後檢查
    for pire in ttl_pire_prompt_score:
        if pire['prompt'] == new_prompt:
            in_list = True
            break

    if DEBUGGER=="True":
        print("exit prompt_in_list")
    return in_list

def time_now():

    # Define the timezone for Taiwan
    taiwan_tz = pytz.timezone('Asia/Taipei')

    # Get the current time in Taiwan timezone
    taiwan_time = datetime.now(taiwan_tz)

    time_formated = taiwan_time.strftime('%Y_%m%d_%H%M')

    return time_formated

def get_file_name(path):
    """ get the corpus name only
    """
    import os

    # 完整文件路径
    path = '/path/to/your/file/config.ini'

    # 获取文件名（带扩展名）
    file_name_with_extension = os.path.basename(path)

    # 获取文件名（不带扩展名）
    file_name, _ = os.path.splitext(file_name_with_extension)

    return file_name
