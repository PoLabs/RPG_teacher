# RPG_teacher
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