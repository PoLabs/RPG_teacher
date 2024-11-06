# RPG_teacher


python nemo_curator_tutorial.py --input-data-dir /home/polabs2/Code/RPG_teacher/data/raw/new --output-clean-dir /home/polabs2/Code/RPG_teacher/data/out --input-file-type pdf --output-file-type jsonl --batch-size 64




A genAI project to turn lesson plans into an interactive RPG

[assistant]: """welcome to RPG Teacher. To begin, choose a textbook chapter, novel, random and click begin."""

[system]: initial_story_prompt = """
You are a storyteller who combines fantasy adventures with educational content.

Given the following information:

**Fantasy Chapter Name:**
{fantasy_chapter_name}

**Fantasy Chapter Summary:**
{fantasy_summary}

**Textbook Chapter Name:**
{textbook_chapter_name}

**Textbook Chapter Summary:**
{textbook_summary}

**Sample Questions:**
{sample_questions}

Create a fun and immersive RPG setting that blends the fantasy narrative with the educational topics. Introduce challenges or puzzles based on the sample questions that the player must solve interactively. Present one challenge at a time, and wait for the player's input before proceeding to the next part of the story.

Please start the story now.
"""


Functions

describe_setting
- input: text, novel
- output: description of setting

describe_question
- input: setting description, text, novel
- output: {question, answer}

handle_answer
- input: user text
- LLM decides:
  - give_hint
    - input:
    - output: hint text
  - grade_answer
    - input: user text, grade level, {question_answer}
    - output: grade, reason, outcome
  - other

player_choice (for next question setup)

- tooltips
- RPG elements
- packages and cleanup
- second example


I'm creating a genAI app that constructs a guided fantasy RPG learning experience. The app uses textbooks and fantasy novels for RAG of an immersive setting filled with text book question encounters. 
For example, on setup might involve teaching a chapter of a Digital Marketing textbook by inserting questions into a RPG story based on the Hobbit. 
Both books are in a vector database and can augment queries with additional context. I also have a dataset with chapter summary notes and chapter sample questions. 
I also have chapter summaries for the fantasy novel. I combined the textbook chapter summary points with a fantasy chapters summary to generate fun and descriptive settings that incorporate questions and trivia based on the textbook chapter. 
I'm using llamaindex, pinecone and nvidia nemo nimm with streamlit as a front end. 

The general flow is user chooses text book, chapter and novel and submits. 
describe_adventure is called which generates a few paragraphs describing the adventure setting and characters involved that combines themes from the novel with key topics from the textbook using the additional context .
Also, it generates  five places, events, or encounters that can be used to stage questions.

describe_setting is then called which picks one of the 5 'Places, Events, or Encounters' from the adventure description and describes the setting the characters now find themselves in. 
Generate 10 sentences that ties in the relevant textbook content.

describe_question is then called which  uses the setting description, context, and sample question to create an engaging question for the player that ties into the RPG adventure. 
The question should integrate educational content from the textbook. Please provide the question and its answer.

on user input, grade_answer is called to assess the following student's answer to the question based on the provided rubric. 
It uses the rubric to determine the level (0-4) that best describes the student's answer. Provide the level and an explanation, referencing the student's answer and the rubric.

player_choice is then called which lets the user decide which of the remaning 'Places, Events, or Encounters' they want to travel to next, then repeats the cycle at describe_setting.





I want to update the app to use llama guard model from hugging face to check user input and system output. Sample code:

from huggingface_hub import InferenceClient

client = InferenceClient(api_key="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

for message in client.chat_completion(
	model="meta-llama/LlamaGuard-7b",
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")


I want to make a few updates to the app:

- easy RPG elements (int for +1 to %  answrs, spirit - not lose hint points, stregth)
- - choose 3 characters from perople palces, assign int, str, spirit to each of them and design a promptr that asks us for their chocie for the new buff
- - buff methods



- clean dir branch --> to master 
- program description (200 words)
- hosting / launch 


here is my current app code:













