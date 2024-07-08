def get_prompt_4_summarize_chunk(chunk, os_prompt, input_question):
    """ extract the key information from the chunk
    Var
        chunk: str
            The chunk that needs to be summarized
            
        os_prompt: str
            The prompt to enhance the model's performance
            
        input_question: str
            The question that needs to be answered
    """
    
    prompt_4_summarize_chunk = \
f"""Article excerpt:
{chunk}

The above is the article excerpt related to my question.
Below is the question I want to ask.
Please select the text content that can answer this question.
{os_prompt}

Question:
{input_question}"""
    return prompt_4_summarize_chunk

def get_prompt_4_exam_multichoice(new_content, input_question):
    """
    Var
        new_content: str
            summarized chuncks of the corresponding content
            
        input_question: str
            The question that needs to be answered
    """

    prompt_4_exam_multichoice = \
f"""There will be an article question and four options. 
Please choose the option that answers the question based on the article.

article:
{new_content}

question:
{input_question}

Your answer must be the number of one of the options,meaning it should be either option1, option2, option3, or option4. 
The format for the answer should be as follows: Answer__optionX."""
    return prompt_4_exam_multichoice

def get_prompt_4_create_new_os_prompt(example):
    """
    Var
        example: str
            form: "[Old prompt]:"{p['prompt']}"\n[Scores]:{p['score']}"
    """
    
    prompt_4_create_new_os_prompt = \
f"""You are an expert at crafting prompts.
Based on the example task given below for an LLM, fill in the most suitable prompt in the place marked [new_prompt].
The following describes the task you will undertake:

"
Article excerpt:
[article_chunk]

The above is the article excerpt related to my question.
Below is the question I want to ask.
Please select the text content that can answer this question.
[new_prompt]

Question:
[input_question]
"

Here are some example prompts and their scores, ranging from 0 to 100, with higher scores indicating better performance.
Please help me think of a unique new_prompt where higher scores are better.

{example}

### You only need to return the new_prompt ###
DON'T return the [Scores] or explanation.
Your new_prompt:__"""
    return prompt_4_create_new_os_prompt

def get_prompt(type_task, *args):
    """
    Var
        type_task: str
            The type of task that the prompt is for.
            0: summarize_chunk
            1: exam_multichoice
            2: create_new_os_prompt
        
        *args: tuple
            The arguments that are needed to generate the prompt.
    """
    if type_task in [0, "sum"]:
        full_prompt = get_prompt_4_summarize_chunk(*args)
    elif type_task in [1, "exam"]:
        full_prompt = get_prompt_4_exam_multichoice(*args)
    elif type_task in [2, "new"]:
        full_prompt = get_prompt_4_create_new_os_prompt(*args)
    else:
        raise ValueError(f"Invalid type_task: {type_task}")
    
    return full_prompt
