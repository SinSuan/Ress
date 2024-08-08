"""
偽 json 檔
"""

from collections import defaultdict
import re
import configparser
import os

CONFIG = configparser.ConfigParser()
PATH_CONFIG = os.getenv('path_2_config')

ENVIRONMENT = os.getenv('environment')
if ENVIRONMENT=="windows":
    CONFIG.read(PATH_CONFIG, encoding='utf-8')
else:
    CONFIG.read(PATH_CONFIG)
    
# DEBUGGER = CONFIG["DEBUGGER"]["DEBUGGER"]
DEBUGGER = "False"

# extract the key information from the chunk
    # chunk: The chunk that needs to be summarized
    # os_prompt: The prompt to enhance the model's performance
    # question_and_options: The question that needs to be answered
PROMPT_4_SUMMARIZE_CHUNK = {
    "old":[
"""Article excerpt:
{chunk}

{os_prompt}

Question:
{question_and_options}"""
    ],
    "new":[
"""Article excerpt:
{chunk}

The above is the article excerpt related to my question.
Below is the question I want to ask.
Please select the text content that can answer this question.
{os_prompt}

Question:
{question_and_options}"""
    ]
}

# answer multichoice
    # content: summarized chuncks of the corresponding content
    # input_question: The question that needs to be answered
PROMPT_4_EXAM_MULTICHOICE = \
"""There will be an article question and four options. 
Please choose the option that answers the question based on the article.

article:
{content}

question:
{input_question}

Your answer must be the number of one of the options,meaning it should be either option1, option2, option3, or option4. 
The format for the answer should be as follows: Answer__optionX."""

# create new os prompt
# ****format 順序要依照註解的順序****
PROMPT_4_CREATE_NEW_OS_PROMPT = {

    # example: form: "[Old prompt]:"{p['prompt']}"\n[Scores]:{p['score']}"
    "ReSS":[
"""You are an expert at crafting prompts.
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
    ],

    # p_1, p_2: the random selected parent prompts to be mutated
    # p_best: the prompt is with the highest score
    # p_i: the prompt to crossover
    "EvoDE":[
"""Please follow the instruction step-by-step to generate a better prompt.
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

1. """
    ],


    # p_contr: mutated from p_worst in high_pop and random p_init
    # p_best: the prompt is with the highest score
    # p_i: the prompt to crossover
    "CoEvo": [
""""Please follow the instruction step-by-btep to generate a better prompt.
1. Identify the same parts and the different parts between Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Randomly mutate the same parts and the different parts.
3. Combine the same parts and the different parts with Prompt 3, selectively replace it with the same parts and the different parts in step 2 and generate a new prompt.
Prompt 3: Rewrite my complex sentence in simpler terms, but keep the meaning.
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.

1.  Identifying the same parts and the different parts between Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
Same parts:
"rewrite"
"simpler"

Different parts:
"input text" vs "my complex sentence"
"simpler text" vs "simpler terms, but keep the meaning"

2. Randomly mutate the same parts and the different parts:
Same parts:
"rewrite" -> "transform"
"simpler" -> "easier"

Different parts:
"input text" -> "provided text"
"my complex sentence" -> "the difficult sentence"
"simpler text" -> "easier language"
"simpler terms, but keep the meaning" -> "simpler words while maintaining the meaning"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Rewrite my complex sentence in simpler terms, but keep the meaning.
New Prompt: Transform the difficult sentence in easier words while maintaining the meaning.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.
Final Prompt: <prompt>Transform the difficult sentence in easier words while maintaining the meaning to make it easier for people who do not speak English fluently to comprehend.</prompt>

Please follow the instruction step-by-btep to generate a better prompt.
1. Identify the same parts and the different parts between Prompt 1 and Prompt 2:
Prompt 1: {p_contr}
Prompt 2: {p_best}
2. Randomly mutate the same parts and the different parts.
3. Combine the same parts and the different parts with Prompt 3, selectively replace it with the same parts and the different parts in step 2 and generate a new prompt.
Prompt 3: {p_best}
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: {p_i}

1. """
    ],

    # p_1, p_2: the random selected parent prompts to crossover
    "EvoGA": [
"""Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Simplify the complex text while maintaining its meaning.
2. <prompt>Simplify the complex text while maintaining its meaning.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: {p_1}
Prompt 2: {p_2}
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """
    ],

    # p_init: random prompt in init_pop
    # p_worst: worst prompt in high_pop
    "ContrGA": [
"""Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Worst Prompt: Rewrite the input text into simpler text.
Best Prompt: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Simplify the complex text while maintaining its meaning.
2. <prompt>Simplify the complex text while maintaining its meaning.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Worst Prompt: {p_init}
Best Prompt: {p_worst}
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """
    ],
    
#     # p_contr: mutated from p_worst in high_pop and random p_init
#     # p_best: the prompt is with the highest score
#     # p_i: the prompt to crossover
#     "ContrCoEvo": [
# """"Please follow the instruction step-by-btep to generate a better prompt.
# 1. Identify the same parts and the different parts between Prompt 1 and Prompt 2:
# Prompt 1: Rewrite the input text into simpler text.
# Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
# 2. Randomly mutate the same parts and the different parts.
# 3. Combine the same parts and the different parts with Prompt 3, selectively replace it with the same parts and the different parts in step 2 and generate a new prompt.
# Prompt 3: Rewrite my complex sentence in simpler terms, but keep the meaning.
# 4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
# Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.

# 1.  Identifying the same parts and the different parts between Prompt 1 and Prompt 2:
# Prompt 1: Rewrite the input text into simpler text.
# Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
# Same parts:
# "rewrite"
# "simpler"

# Different parts:
# "input text" vs "my complex sentence"
# "simpler text" vs "simpler terms, but keep the meaning"

# 2. Randomly mutate the same parts and the different parts:
# Same parts:
# "rewrite" -> "transform"
# "simpler" -> "easier"

# Different parts:
# "input text" -> "provided text"
# "my complex sentence" -> "the difficult sentence"
# "simpler text" -> "easier language"
# "simpler terms, but keep the meaning" -> "simpler words while maintaining the meaning"

# 3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
# Prompt 3: Rewrite my complex sentence in simpler terms, but keep the meaning.
# New Prompt: Transform the difficult sentence in easier words while maintaining the meaning.

# 4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
# Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.
# Final Prompt: <prompt>Transform the difficult sentence in easier words while maintaining the meaning to make it easier for people who do not speak English fluently to comprehend.</prompt>

# Please follow the instruction step-by-btep to generate a better prompt.
# 1. Identify the same parts and the different parts between Prompt 1 and Prompt 2:
# Prompt 1: {p_contr}
# Prompt 2: {p_best}
# 2. Randomly mutate the same parts and the different parts.
# 3. Combine the same parts and the different parts with Prompt 3, selectively replace it with the same parts and the different parts in step 2 and generate a new prompt.
# Prompt 3: {p_best}
# 4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
# Basic Prompt: {p_i}

# 1. """
#     ],


}

class GetPrompt:
    """ 各函式的明確用途可以從其使用的 global variable 得知
    """
    
    def __init__(self) -> None:
        pass
    
    def get_final_prompt(self, template, *args):
        if DEBUGGER=="True": print("enter GetPrompt.get_final_prompt")
        
        args = list(args)
        keys = re.findall('{(.*?)}', template)  # 抓出 key
        # kwargs = dict(zip(keys, args))  # 製作字典

        # 製作字典
        kwargs = defaultdict(lambda: None)  # 沒用到的 key 預設值為 None
        for key in keys:
            if kwargs[key] is None:
                kwargs[key] = args.pop(0)
        kwargs = dict(kwargs)
        # print(f"{kwargs=}")
        final_prompt = template.format(**kwargs)   # 代換 prompt 中的變數
        
        if DEBUGGER=="True": print("exit GetPrompt.get_final_prompt")
        return final_prompt
    
    def sum(self, type_os_prompt, *args):
        """ extract the key information from the chunk
        Var
            type_os_prompt:
                "old" or "new"
                
            args
                chunk:
                    The chunk that needs to be summarized
                os_prompt:
                    The prompt to enhance the model's performance
                question_and_options:
                    The question that needs to be answered
        """
        if DEBUGGER=="True": print("enter GetPrompt.sum")
        
        template = PROMPT_4_SUMMARIZE_CHUNK[type_os_prompt][0]
        final_prompt = self.get_final_prompt(template, *args)
        
        if DEBUGGER=="True": print("exit GetPrompt.sum")
        return final_prompt
    
    def exam(self, *args):
        """ answer multichoice
        Var
            args    
                content:
                    summarized chuncks of the corresponding content
                input_question:
                    The question that needs to be answered
        """
        if DEBUGGER=="True": print("enter GetPrompt.exam")
        
        template = PROMPT_4_EXAM_MULTICHOICE
        final_prompt = self.get_final_prompt(template, *args)
        
        if DEBUGGER=="True": print("exit GetPrompt.exam")
        return final_prompt
    
    def create(self, type_update, *args):
        """ create new os prompt
        Var
            type_update:
                "ReSS" or "EvoDE" or "EvoGA" or "CoEvo"
        
            args
                "ReSS"      example
                "EvoDE"     p_best, p_i, p_1, p_2
                "EvoGA"     p_1, p_2
                "CoEvo"     p_best, p_contr,p_i
        """
        if DEBUGGER=="True": print("enter GetPrompt.create")
        
        template = PROMPT_4_CREATE_NEW_OS_PROMPT[type_update][0]
        final_prompt = self.get_final_prompt(template, *args)
        
        if DEBUGGER=="True": print("exit GetPrompt.create")
        return final_prompt

get_prompt = GetPrompt()
