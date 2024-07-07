def prompt_4_summarize_chunk(chunk, os_prompt, input_question):
    """ extract the key information from the chunk
    Var
        chunk: str
            The chunk that needs to be summarized
            
        os_prompt: str
            The prompt to enhance the model's performance
            
        input_question: str
            The question that needs to be answered
    """
    
    input_for_reader = \
f"""Article excerpt:
{chunk}

The above is the article excerpt related to my question.
Below is the question I want to ask.
Please select the text content that can answer this question.
{os_prompt}

Question:
{input_question}"""
    return input_for_reader

def prompt_4_exam_multichoice(new_content, input_question):
    """
    Var
        new_content: str
            summarized chuncks of the corresponding content
            
        input_question: str
            The question that needs to be answered
    """

    input_to_llm = \
f"""There will be an article question and four options. 
Please choose the option that answers the question based on the article.

article:
{new_content}

question:
{input_question}

Your answer must be the number of one of the options,meaning it should be either option1, option2, option3, or option4. 
The format for the answer should be as follows: Answer__optionX."""
    return input_to_llm

def prompt_4_create_new_prompt(example):
    """
    Var
        example: str
            form: "[Old prompt]:"{p['prompt']}"\n[Scores]:{p['score']}"
    """
    
    input_to_temperature_llm = \
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
    return input_to_temperature_llm

def get_prompt(type_task, *args):
    """
    Var
        type_task: str
            The type of task that the prompt is for.
            0: summarize_chunk
            1: exam_multichoice
            2: create_new_prompt
        
        *args: tuple
            The arguments that are needed to generate the prompt.
    """
    if type_task==0:
        full_prompt = prompt_4_summarize_chunk(*args)
    elif type_task==1:
        full_prompt = prompt_4_exam_multichoice(*args)
    elif type_task==2:
        full_prompt = prompt_4_create_new_prompt(*args)
    else:
        raise ValueError(f"Invalid type_task: {type_task}")
    
    return full_prompt
