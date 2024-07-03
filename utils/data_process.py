import json

# Read data from path
def get_data(path):
    with open(path,"r",encoding="utf-8")as file:
        data=json.load(file)
    return data

# format universal data format
def data_format(content,question,answer):
    return_format={
        "content":content,
        "question":question,
        "answer":answer
    }
    return return_format