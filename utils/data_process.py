"""
這個檔案目前沒有用到，但 data preprocess 時會用到 data_format
"""

import configparser
import json
import os

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]

# Read data from path
def get_data(path):
    if DEBUGGER=="True":
        print("enter get_data")

    with open(path,"r",encoding="utf-8")as file:
        data=json.load(file)

    if DEBUGGER=="True":
        print("exit get_data")
    return data

# format universal data format
def data_format(content,question,answer):
    if DEBUGGER=="True":
        print("enter data_format")
    return_format={
        "content":content,
        "question":question,
        "answer":answer
    }
    if DEBUGGER=="True":
        print("exit data_format")
    return return_format
