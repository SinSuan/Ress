def get_input_for_reader(chunk, os_prompt, input_question):
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

def get_input_to_llm(new_content, input_question):
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

def get_input_to_temperature_llm(example):
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
