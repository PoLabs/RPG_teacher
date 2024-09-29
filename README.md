# RPG_teacher
A genAI project to turn lesson plans into an interactive RPG



system

[ast]: welcome to RPG Teacher. To begin, choose a textbook chapter, novel, random and click begin.


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