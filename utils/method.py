import configparser

from .prompt import get_prompt
from .split import get_ttl_chunk
CONFIG  =  configparser.ConfigParser()
CONFIG.read("/user_data/itri/Ress/config.ini")
DEBUGGER  =  CONFIG["DEBUGGER"]["DEBUGGER"]

def os_ap_sss_answer(ttl_model, os_prompt, data, chunk_size = 3000, overlap = 10):
    """
    Var
        ttl_model: (llm, embedding_model)
            llm: function
            embedding_model: Encoder in utils.call.embedding

        chunk_size: int
            the number of words in each chunk
            
        overlap: int
            the number of senetences that overlap between two chunks
    """
    if DEBUGGER=="True":
        print("enter os_ap_sss_answer")

    input_content = data['content']   # 參考文章
    input_question = data['question'] # 問題(多選題)
    get_llm_reply, embedding_model = ttl_model

    # 切分摘要完要輸入給llm的內容
    new_content = input_content
    while len(new_content.split(" "))>chunk_size:

        content_chuncks = get_ttl_chunk(new_content, chunk_size, overlap, embedding_model)
        # 這一輪的新內容
        new_content = ""
        for chunk in content_chuncks:

            # 請llm幫我們把重要資訊留下
            prompt_4_summarize_chunk =  get_prompt(0, chunk, os_prompt, input_question)
            chunk_summary = get_llm_reply(prompt_4_summarize_chunk)
            # if chunk_summary==None:
            #     continue
            new_content+= chunk_summary+" "

        # 防錯(如果LLM api無回傳 直接比照truncate)
        if new_content=="":
            new_content = " ".join(input_content.split(" ")[:3000])
            break

    # 找完有用的內容後，進行問答
    prompt_4_exam_multichoice = get_prompt(1,  new_content,  input_question)
    answer_from_llm = get_llm_reply(prompt_4_exam_multichoice)

    if DEBUGGER=="True":
        print("exit os_ap_sss_answer")
    return answer_from_llm
