import configparser
CONFIG = configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
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

def prompt_in_list(prompt_list, new_prompt):
    """
    Var
        prompt_list: List[Dict]
            list of "{'prompt': prompt, 'score': score}"

        new_prompt: str
            the prompt to be check whether or not the in prompt_list
    """
    if DEBUGGER=="True":
        print("enter prompt_in_list")

    # 預設不在
    in_list = False
    # 然後檢查
    for item in prompt_list:
        if item['prompt'] == new_prompt:
            in_list = True
            break

    if DEBUGGER=="True":
        print("exit prompt_in_list")
    return in_list
