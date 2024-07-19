"""
prompt 太常了，另外寫成 function
不存成 json 是因為仍要便於閱讀與編輯
"""

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

def get_EvoPrompt_4_create_new_os_prompt(p_best, p_i, p_1, p_2):
    """
    Var
        p_best: the prompt is with the highest score
        p_1: the parent prompt 1 is mutated 
        p_2: the parent prompt 2 is mutated
        p_i: the prompt is going to evolute
    """
    prompt_4_create_new_os_prompt = \
f"""
Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt.
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
Different parts:
"input text" vs "my complex sentence"
"simpler text" vs "simpler terms, but keep the meaning"

2. Randomly mutate the different parts:
"input text" -> "provided text"
"my complex sentence" -> "the difficult sentence"
"simpler text" -> "easier language"
"simpler terms, but keep the meaning" -> "simpler words while maintaining the meaning"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.
New Prompt: Transform the provided text into easier language while maintaining the meaning, making it accessible for non-native English speakers.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.
Final Prompt: <prompt>Convert the difficult sentence into simpler words while preserving the meaning, so it's easier for non-native English speakers to understand.</prompt>


Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: {p_1}
Prompt 2: {p_2}
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step2 and generate a new prompt.
Prompt 3: {p_best}
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: {p_i}

1. 
"""
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
        user_prompt = get_prompt_4_summarize_chunk(*args)
    elif type_task in [1, "exam"]:
        user_prompt = get_prompt_4_exam_multichoice(*args)
    elif type_task in [2, "new"]:
        user_prompt = get_prompt_4_create_new_os_prompt(*args)
    elif type_task in [3, "new_EvoPrompt"]:
        user_prompt = get_EvoPrompt_4_create_new_os_prompt(*args)
    else:
        raise ValueError(f"Invalid type_task: {type_task}")

    return user_prompt
