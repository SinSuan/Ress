import configparser
import json

CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
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
